import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import wandb #using wandb 

# Add parent directory to path so we can import from data, listener, utils
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

def _mask_prompt_labels(full_ids, prompt_lens):
    labels = full_ids.clone()

    for i in range(labels.size(0)):
        labels[i, : int(prompt_lens[i].item())] = -100
    return labels

def collate_pairs(tokenizer, prompts, chosen, rejected, *, max_length):
    token_p = tokenizer(prompts, padding=True, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_lens = token_p["attention_mask"].sum(dim=1)
    
    tokenizer_chosen = tokenizer([p + c for p, c in zip(prompts, chosen)],
                      padding=True, return_tensors="pt", truncation=True, max_length=max_length)
    tokenizer_rejected = tokenizer([p + r for p, r in zip(prompts, rejected)],
                      padding=True, return_tensors="pt", truncation=True, max_length=max_length)

    labels_c = _mask_prompt_labels(tokenizer_chosen["input_ids"], prompt_lens)
    labels_r = _mask_prompt_labels(tokenizer_rejected["input_ids"], prompt_lens)

    if len(prompts) > 0:
        comp_c = (labels_c[0] != -100).sum().item()
        comp_r = (labels_r[0] != -100).sum().item()

        print(
            "[collate_pairs DEBUG]",
            "prompt_len:", int(prompt_lens[0].item()),
            "seq_len:", tokenizer_chosen["input_ids"].size(1),
            "comp_tokens_chosen:", comp_c,
            "comp_tokens_rejected:", comp_r,
            flush=True
        )

        # If no completion tokens, print the actual text
        if comp_c == 0 or comp_r == 0:
            print("PROMPT:", repr(prompts[0]), flush=True)
            print("CHOSEN:", repr(chosen[0]), flush=True)
            print("REJECTED:", repr(rejected[0]), flush=True)

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
        policy_model, #model being trained
        reference_model, #frozen reference model
        input_path, #path to the input
        output_path, #path to the folder for output
        epochs, 
        batch_size, 
        grad_accum, #number of batches to accumulate gradients before taking an optimizer step
        lr,
        alpha0, #scaling for length penalty
        beta, #strength of DPO, how much chosen is picked over rejected, higher beta more aggressive
        max_length,
        top_p,
        temperature,
        max_new_tokens,
        repetition_penalty,
        no_repeat_ngram_size,
        score_gap_min, #min score gap to treat one candidate as meaningfully better
        max_pair_similarity, #max jaccard similarity between candidates to be considered a valid pair
        max_resample_tries, #max number of times to resample invalid candidates
        listener_model_type, #model type for listener
        listener_batch_size
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(policy_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(policy_model).to(device)
    reference = AutoModelForCausalLM.from_pretrained(reference_model).to(device)
    reference.eval()

    for param in reference.parameters():
        param.requires_grad_(False)

    listener = BERTScoreListener(model_type=listener_model_type, batch_size=listener_batch_size, device=device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    loader = SimpleWikiPassageLoader(path=input_path, limit=500) #set limit to 500 to test small sample

    policy.train()
    global_step = 0

    for e in range(epochs): #eventually use e to alter the max_new_tokens
        prompts, chosen, rejected = [], [], []
        kept, skipped = 0, 0

        for example in loader:
            utterance = example['passage'] #change, utterance is only here for CHILDES dataset
            prompt = make_prompt(utterance)
            reference_text = utterance #should change if wnat to use a different reference

            seed_a = torch.randint(0, 2**31 - 1, (1,)).item()
            seed_b = torch.randint(0, 2**31 - 1, (1,)).item()

            candidate_a = generate_summary(
                policy,
                tokenizer,
                prompt,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens, #add something to incorporate e into max_new_tokens
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                seed=seed_a
            )
            
            for _ in range(max_resample_tries + 1):
                candidate_b = generate_summary(
                    policy,
                    tokenizer,
                    prompt,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens, #add something to incorporate e into max_new_tokens
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    seed=seed_b
                )

                similarity  = jaccard_ngrams(candidate_a, candidate_b, n=2)
                if similarity <= max_pair_similarity:
                    break

            preferred = listener.prefer(candidate_a, candidate_b, reference_text)
            score_a = float(preferred['score_a'])
            score_b = float(preferred['score_b'])
            gap = abs(score_a - score_b)

            if gap < score_gap_min:
                skipped += 1 # counts if the pair was skipped
                continue

            if preferred['preferred'] == 'A':
                c, r = candidate_a, candidate_b
            else:
                c, r = candidate_b, candidate_a

            prompts.append(prompt)
            chosen.append(c)
            rejected.append(r)
            kept += 1

            if len(prompts) == batch_size: #iterated over optimizer, now update gradient
                batch = collate_pairs(tokenizer, prompts, chosen, rejected, max_length=max_length)

                batch = PairBatch(
                    batch.ids_c.to(device),
                    batch.attn_c.to(device),
                    batch.labels_c.to(device),
                    batch.ids_r.to(device),
                    batch.attn_r.to(device),
                    batch.labels_r.to(device),
                )

                loss, metrics = dpo_loss(policy, reference, batch, e, epochs, alpha0, beta=beta)
                loss.backward()
                global_step += 1

                if global_step % grad_accum == 0: #steps out of optimizer if too many global steps taken
                    optimizer.step()
                    optimizer.zero_grad()

                    optimizer_steps = global_step // grad_accum
                    save_interval = 100  
                    if optimizer_steps % save_interval == 0:
                        os.makedirs(output_path, exist_ok=True)
                        policy.save_pretrained(output_path)
                        tokenizer.save_pretrained(output_path)
                        print(f"[saved] optimizer_steps={optimizer_steps} -> {output_path}")

                prompts, chosen, rejected = [], [], []

            if prompts:
                pass

            if (global_step+1) % 10 == 0:
                print(f"[epoch {e}] kept_pairs={kept} skipped_lowgap={skipped} steps={global_step}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy_model", type=str, required=True, default = 'gpt2')
    parser.add_argument("--reference_model", type=str, required=True, default = 'gpt2')
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16) #should bump up to 32 or 64 when we get onto GPU
    parser.add_argument("--grad_accum", type=int, default=8) # large grad_accum training slow, might over fit, small grad_accum training can be unstable
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--alpha0", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    parser.add_argument("--score_gap_min", type=float, default=0.1)
    parser.add_argument("--max_pair_similarity", type=float, default=0.80)
    parser.add_argument("--max_resample_tries", type=int, default=4)

    parser.add_argument("--listener_model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--listener_batch_size", type=int, default=8)

    return parser.parse_args()

def main():
    args = parse_args()

    #overwrite the args with wandb sweep config
    #uncomment when running wandb sweep
    ##########################################################
    # wandb.init(project="telegraphic-dpo", config=vars(args))
    # config = wandb.config

    # args.lr = config.lr
    # args.beta = config.beta #modifier for DPO preference logits
    # args.alpha0 = config.alpha0 #initial scaling for length penalty
    # args.score_gap_min = config.score_gap_min #min score gap for a pair to be "valid" for training
    # args.grad_accum = config.grad_accum

    #not sure if this should be changed, not related to DPO objective
    # args.temperature = config.temperature
    # args.top_p = config.top_p
    ##########################################################

    

    train_dpo(
        policy_model=args.policy_model,
        reference_model=args.reference_model,
        input_path=args.input_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        alpha0=args.alpha0,
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
        listener_batch_size=args.listener_batch_size
    )

if __name__ == "__main__":
    main()