import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloader_wiki import SimpleWikiPassageLoader
from listener.listener.bertscore_listener import BERTScoreListener
from utils.utils import generate_summary, jaccard_ngrams, make_prompt

from dataclasses import dataclass

@dataclass
class PairBatch:
    #chosen
    ids_c: torch.Tensor
    attn_c: torch.Tensor
    labels_c: torch.Tensor
    #rejected
    ids_r: torch.Tensor
    attn_r: torch.Tensor
    labels_r: torch.Tensor

def _mask_prompt_labels(full_ids, prompt_lens, pad_token_id):
    labels = full_ids.clone()
    for i in range(labels.size(0)):
        labels[i, : int(prompt_lens[i].item())] = -100
    
    # Mask padding tokens as well to fix cuda error
    labels[labels == pad_token_id] = -100
    
    return labels

def collate_pairs(tokenizer, prompts, chosen, rejected, *, max_length):
    prompt_lens = torch.tensor([len(tokenizer(p, truncation=True, max_length=max_length)["input_ids"]) for p in prompts], dtype=torch.long)
    
    tokenizer_chosen = tokenizer([p + c for p, c in zip(prompts, chosen)],
                      padding=True, return_tensors="pt", truncation=True, max_length=max_length)
    tokenizer_rejected = tokenizer([p + r for p, r in zip(prompts, rejected)],
                      padding=True, return_tensors="pt", truncation=True, max_length=max_length)

    labels_c = _mask_prompt_labels(tokenizer_chosen["input_ids"], prompt_lens, tokenizer.pad_token_id)
    labels_r = _mask_prompt_labels(tokenizer_rejected["input_ids"], prompt_lens, tokenizer.pad_token_id)

    # drop the -100 filtered out examples
    has_c = (labels_c != -100).any(dim=1)   
    has_r = (labels_r != -100).any(dim=1)   
    keep = has_c & has_r    

    if not keep.all():
        tokenizer_chosen = {k: v[keep] for k, v in tokenizer_chosen.items()}
        labels_c = labels_c[keep]

        tokenizer_rejected = {k: v[keep] for k, v in tokenizer_rejected.items()}
        labels_r = labels_r[keep]

    return PairBatch(
        tokenizer_chosen["input_ids"], tokenizer_chosen["attention_mask"], labels_c,
        tokenizer_rejected["input_ids"], tokenizer_rejected["attention_mask"], labels_r
    )

def sequential_log_prob(model, ids, attn_mask, labels):
    output = model(input_ids=ids, attention_mask=attn_mask, labels=labels)
    logits = output.logits[:, :-1]
    targets = labels[:, 1:]
    valid = targets != -100

    log_probs = F.log_softmax(logits, dim=-1)
    targets_safe = targets.clone()
    targets_safe[~valid] = 0
    targets_safe = targets_safe.long()

    vocab_size = log_probs.size(-1)
    targets_safe = targets_safe.clamp(0, vocab_size - 1)

    token_log_probs = log_probs.gather(-1, targets_safe.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * valid
    return token_log_probs.sum(dim=1)

def _completion_lengths(labels):
    """
    Helper that removes invalid labels
    """
    return (labels != -100).sum(dim=1)

def _anneal_alpha(epoch, max_epochs, alpha0, k):
    """
    - exp: alpha0 * exp(-k * epoch/(max_epochs-1)) where k is a constant > 1, the larger it is the more extreme the penalty curve
    """
    if max_epochs <= 1:
        return alpha0

    t = epoch / (max_epochs - 1)  # 0 -> 1

    return alpha0 * float(torch.exp(torch.tensor(-k * t)))

def dpo_loss(policy, ref, batch, epoch, max_epochs, alpha0, *, beta): #is a BatchPair
    pi_chosen = sequential_log_prob(policy, batch.ids_c, batch.attn_c, batch.labels_c)
    pi_rejected = sequential_log_prob(policy, batch.ids_r, batch.attn_r, batch.labels_r)

    #when frozen
    with torch.no_grad():
        ref_chosen = sequential_log_prob(ref, batch.ids_c, batch.attn_c, batch.labels_c)
        ref_rejected = sequential_log_prob(ref, batch.ids_r, batch.attn_r, batch.labels_r)
    
    #dpo preference objective
    pref_logits = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)

    with torch.no_grad():
        len_c = _completion_lengths(batch.labels_c).float() 
        len_r = _completion_lengths(batch.labels_r).float()  
        len_adv = (len_r - len_c) / (len_r + len_c + float('1e-8')) # float is to avoid crashes when both produce output of length 0

        alpha_k = 2.0 #default is 2.0 change later, might want to make hyperparam
        modded_alpha = _anneal_alpha(epoch, max_epochs, alpha0, alpha_k) #makes alpha decrease over time exponentially

    logits = beta * pref_logits + modded_alpha * len_adv #combines the preference score and length scoring

    loss = -F.logsigmoid(logits).mean() #negative log likelihood of choosing chosen over rejected

    metrics = {
        "loss": float(loss.item()),
        "pref_logit_mean": float(pref_logits.mean().item()),
        "len_adv_mean": float(len_adv.mean().item()),
        "alpha": float(modded_alpha),
        "len_chosen_mean": float(len_c.mean().item()),
        "len_rejected_mean": float(len_r.mean().item()),
    }
    return loss, metrics

def train_dpo(
        *,
        policy_model,
        reference_model,
        input_path,
        output_path,
        epochs, 
        batch_size, 
        grad_accum,
        lr,
        alpha,
        beta,
        max_length,
        top_p,
        temperature,
        max_new_tokens,
        repetition_penalty,
        no_repeat_ngram_size,
        score_gap_min,
        max_pair_similarity,
        max_resample_tries,
        listener_model_type,
        listener_batch_size,
        wandb_project=None,
        wandb_run_name=None
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize wandb
    if wandb_project is not None:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "policy_model": policy_model,
                "reference_model": reference_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "learning_rate": lr,
                "alpha": alpha,
                "beta": beta,
                "max_length": max_length,
                "top_p": top_p,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "score_gap_min": score_gap_min,
                "max_pair_similarity": max_pair_similarity,
                "max_resample_tries": max_resample_tries,
                "listener_model_type": listener_model_type,
                "listener_batch_size": listener_batch_size,
                "device": device
            }
        )
        print(f"[WandB] Initialized project: {wandb_project}, run: {wandb_run_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(policy_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Model] Loading policy model: {policy_model}")
    policy = AutoModelForCausalLM.from_pretrained(policy_model).to(device)
    
    print(f"[Model] Loading reference model: {reference_model}")
    reference = AutoModelForCausalLM.from_pretrained(reference_model).to(device)
    reference.eval()

    for param in reference.parameters():
        param.requires_grad_(False)

    print(f"[Listener] Initializing BERTScore listener: {listener_model_type}")
    listener = BERTScoreListener(
        model_type=listener_model_type, 
        batch_size=listener_batch_size, 
        device=device,
        rescale_with_baseline=False  #avoid negative scores
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    print(f"[Data] Loading data from: {input_path}")
    loader = SimpleWikiPassageLoader(path=input_path, limit=None)

    policy.train()
    global_step = 0
    total_kept = 0
    total_skipped = 0

    for e in range(epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {e+1}/{epochs}")
        print(f"{'='*50}")
        
        prompts, chosen, rejected = [], [], []
        kept, skipped = 0, 0

        for example in loader:
            utterance = example['passage']
            prompt = make_prompt(utterance)
            reference_text = utterance

            seed_a = torch.randint(0, 2**31 - 1, (1,)).item()
            seed_b = torch.randint(0, 2**31 - 1, (1,)).item()

            candidate_a = generate_summary(
                policy, tokenizer, prompt,
                top_p=top_p, temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                seed=seed_a
            )
            
            # Resample candidate_b if too similar to candidate_a
            for _ in range(max_resample_tries + 1):
                candidate_b = generate_summary(
                    policy, tokenizer, prompt,
                    top_p=top_p, temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    seed=seed_b
                )
                similarity = jaccard_ngrams(candidate_a, candidate_b, n=2)
                if similarity <= max_pair_similarity:
                    break

            # Listener evaluation
            preferred = listener.prefer(candidate_a, candidate_b, reference_text)
            score_a = float(preferred['score_a'])
            score_b = float(preferred['score_b'])
            gap = abs(score_a - score_b)

            # Skip low-quality pairs
            if gap < score_gap_min:
                skipped += 1
                total_skipped += 1
                continue

            # Determine chosen/rejected
            if preferred['preferred'] == 'A':
                c, r = candidate_a, candidate_b
            else:
                c, r = candidate_b, candidate_a

            prompts.append(prompt)
            chosen.append(c)
            rejected.append(r)
            kept += 1
            total_kept += 1

            # Process batch when full
            if len(prompts) == batch_size:
                batch = collate_pairs(tokenizer, prompts, chosen, rejected, max_length=max_length)

                if batch.ids_c.size(0) == 0: #get around edge case where all pairs are filtered out
                    print("Skipped: empty batch after dropping pairs", flush=True)
                    for i in range(min(3, len(prompts))):
                        print("---- PAIR", i, "----")
                        print("PROMPT:", repr(prompts[i]))
                        print("CHOSEN:", repr(chosen[i]))
                        print("REJECTED:", repr(rejected[i]))
                    prompts, chosen, rejected = [], [], []
                    continue

                batch = PairBatch(
                    batch.ids_c.to(device), batch.attn_c.to(device), batch.labels_c.to(device),
                    batch.ids_r.to(device), batch.attn_r.to(device), batch.labels_r.to(device),
                )

                loss, metrics = dpo_loss(policy, reference, batch, e, epochs, alpha, beta=beta)
                loss = loss / grad_accum
                loss.backward()
                global_step += 1

                # Log to wandb
                if wandb_project is not None:
                    wandb.log({
                        "epoch": e,
                        "global_step": global_step,
                        **metrics,
                        "kept_pairs": kept,
                        "skipped_pairs": skipped,
                        "total_kept": total_kept,
                        "total_skipped": total_skipped
                    })

                # Optimizer step with gradient accumulation
                if global_step % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    optimizer_steps = global_step // grad_accum
                    
                    # Save checkpoints
                    save_interval = 100
                    if optimizer_steps % save_interval == 0:
                        checkpoint_path = os.path.join(output_path, f"checkpoint-{optimizer_steps}")
                        os.makedirs(checkpoint_path, exist_ok=True)
                        policy.save_pretrained(checkpoint_path)
                        tokenizer.save_pretrained(checkpoint_path)
                        print(f"[Checkpoint] Saved at step {optimizer_steps}: {checkpoint_path}")
                        
                        # Log checkpoint to wandb
                        if wandb_project is not None:
                            wandb.log({"checkpoint_step": optimizer_steps})

                prompts, chosen, rejected = [], [], []

            # Periodic logging
            if (global_step + 1) % 10 == 0:
                print(f"[Epoch {e+1}] Steps={global_step} | Kept={kept} | Skipped={skipped}")

        print(f"\n[Epoch {e+1} Complete] Total pairs kept: {kept}, skipped: {skipped}")

    # Save final model
    final_path = os.path.join(output_path, "final_model")
    os.makedirs(final_path, exist_ok=True)
    policy.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining Complete. Final model saved to: {final_path}")
    
    if wandb_project is not None:
        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--policy_model", type=str, required=True, default='gpt2')
    parser.add_argument("--reference_model", type=str, required=True, default='gpt2')
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    
    # Generation arguments
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    
    # Preference filtering arguments
    parser.add_argument("--score_gap_min", type=float, default=0.0)
    parser.add_argument("--max_pair_similarity", type=float, default=0.85)
    parser.add_argument("--max_resample_tries", type=int, default=2)
    
    # Listener arguments
    parser.add_argument("--listener_model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--listener_batch_size", type=int, default=8)
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_args()

    train_dpo(
        policy_model=args.policy_model,
        reference_model=args.reference_model,
        input_path=args.input_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        max_length=args.max_length,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        score_gap_min=args.score_gap_min,
        max_pair_similarity=args.max_pair_similarity,
        max_resample_tries=args.max_resample_tries,
        listener_model_type=args.listener_model_type,
        listener_batch_size=args.listener_batch_size,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )

if __name__ == "__main__":
    main()
