import argparse
import os
import shutil
import sys
import random
from collections import deque
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloader_wiki import SimpleWikiPassageLoader
from listener.listener.bertscore_listener import BERTScoreListener
from utils.utils import generate_summary, jaccard_ngrams, make_prompt, set_global_seed

from dataclasses import dataclass

RANDOM_SEED = 42 #random seed for random initialized weights

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

def split_train_test_examples(examples, test_size, split_seed):
    if not 0.0 <= test_size < 1.0:
        raise ValueError(f"--test_size must be in [0.0, 1.0), got {test_size}")

    if len(examples) == 0:
        return [], []

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(examples), generator=generator).tolist()
    shuffled = [examples[i] for i in indices]

    test_count = int(len(shuffled) * test_size)
    if test_size > 0.0:
        test_count = max(1, test_count)
    test_count = min(test_count, max(0, len(shuffled) - 1))

    train_examples = shuffled[test_count:]
    test_examples = shuffled[:test_count]
    return train_examples, test_examples

def _mask_prompt_labels(full_ids, prompt_lens, pad_token_id):
    labels = full_ids.clone()
    for i in range(labels.size(0)):
        labels[i, : int(prompt_lens[i].item())] = -100
    
    # Mask padding tokens as well to fix cuda error
    labels[labels == pad_token_id] = -100
    
    return labels

def collate_pairs(tokenizer, prompts, chosen, rejected, *, max_length):
    token_p = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    prompt_lens = token_p["attention_mask"].sum(dim=1)
 
    
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

def collate_lm(tokenizer, texts, *, max_length):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    enc["labels"] = labels
    return enc

def nll_loss(policy, batch): # negative log likelihood loss for next token prediction
    out = policy(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.loss

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

def dpo_loss(policy, batch, epoch, max_epochs, alpha0, alpha_k, *, beta): #is a BatchPair
    pi_chosen = sequential_log_prob(policy, batch.ids_c, batch.attn_c, batch.labels_c)
    pi_rejected = sequential_log_prob(policy, batch.ids_r, batch.attn_r, batch.labels_r)
    
    #dpo preference objective
    pref_logits = (pi_chosen - pi_rejected)

    with torch.no_grad():
        len_c = _completion_lengths(batch.labels_c).float() 
        len_r = _completion_lengths(batch.labels_r).float()  
        len_adv = (len_r - len_c) / (len_r + len_c + float('1e-8')) # float is to avoid crashes when both produce output of length 0

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

def _update_ema(prev, value, alpha):
    if prev is None:
        return float(value)
    return float(alpha * value + (1.0 - alpha) * prev)

def _update_sma(window_values, value):
    window_values.append(float(value))
    return float(sum(window_values) / len(window_values))

#function to check what the top p are
def inspect_top_p_tokens(model, tokenizer, prompt, top_p, top_n_to_show=20):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs)
        
        # Get logits for the NEXT token
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1).squeeze()

        # Sort tokens by probability
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Compute cumulative probability
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Keep tokens inside top-p nucleus
        nucleus_mask = cumulative_probs <= top_p
        nucleus_indices = sorted_indices[nucleus_mask]

        print(f"\nTop-p (p={top_p}) nucleus size:", nucleus_indices.shape[0])
        print("Top tokens in nucleus:\n")

        for i in range(min(top_n_to_show, nucleus_indices.shape[0])):
            token_id = sorted_indices[i].item()
            token_str = tokenizer.decode([token_id])
            token_prob = sorted_probs[i].item()
            print(f"{i+1:>2}. '{token_str}'  prob={token_prob:.4f}")
    model.eval()

def quick_generate_sample(
    policy,
    tokenizer,
    prompt,
    *,
    top_p,
    temperature,
    max_new_tokens,
    repetition_penalty,
    no_repeat_ngram_size,
):
    policy.eval()
    with torch.inference_mode():
        text = generate_summary(
            policy,
            tokenizer,
            prompt,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            seed=RANDOM_SEED,
        )
    policy.train()
    return text

def train_dpo(
        *,
        policy_model,
        input_path,
        output_path,
        epochs, 
        batch_size, 
        grad_accum,
        lr,
        alpha,
        alpha_k,
        beta,
        max_length,
        top_p,
        temperature,
        max_new_tokens,
        repetition_penalty,
        no_repeat_ngram_size,
        score_gap_min,
        val_score_gap_min,
        max_pair_similarity,
        max_resample_tries,
        listener_model_type,
        listener_batch_size,
        test_size,
        split_seed,
        run_validation,
        nll_warmup_steps,
        nll_steps_per_cycle,
        dpo_steps_per_cycle,
        nll_batch_size,
        validation_max_examples,
        train_loss_ema_alpha,
        train_loss_sma_window,
        wandb_project=None,
        wandb_run_name=None
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ema_alpha = train_loss_ema_alpha
    train_loss_sma_values = deque(maxlen=train_loss_sma_window)
    train_loss_sma = None
    train_loss_ema = None
    val_loss_ema = None

    sweep_mode = False

    set_global_seed(42) #ensure reproducibility of random weights and data splits for sweeps
    
    # Initialize wandb
    if wandb_project is not None:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "policy_model": policy_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "learning_rate": lr,
                "alpha": alpha,
                "alpha_k": alpha_k,
                "beta": beta,
                "max_length": max_length,
                "top_p": top_p,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "score_gap_min": score_gap_min,
                "val_score_gap_min": val_score_gap_min,
                "max_pair_similarity": max_pair_similarity,
                "max_resample_tries": max_resample_tries,
                "listener_model_type": listener_model_type,
                "listener_batch_size": listener_batch_size,
                "test_size": test_size,
                "split_seed": split_seed,
                "run_validation": run_validation,
                "nll_warmup_steps": nll_warmup_steps,
                "nll_steps_per_cycle": nll_steps_per_cycle,
                "dpo_steps_per_cycle": dpo_steps_per_cycle,
                "nll_batch_size": nll_batch_size,
                "validation_max_examples": validation_max_examples,
                "ema_alpha": ema_alpha,
                "train_loss_sma_window": train_loss_sma_window,
                "device": device
            }
        )

        #creates a folder for each run based on wandb run id to clear up space
        base_output_path = output_path
        sweep_mode = bool(os.environ.get("WANDB_SWEEP_ID")) #true if a sweep

        if sweep_mode and wandb.run is not None:
            run_id = wandb.run.id
            run_output_path = os.path.join(base_output_path, f"run-{run_id}")
        else:
            run_output_path = base_output_path

        os.makedirs(run_output_path, exist_ok=True)
        output_path = run_output_path

        # Use one explicit step metric for all logs to avoid out-of-order warnings.
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        print(f"[WandB] Initialized project: {wandb_project}, run: {wandb_run_name}")

    try:
    
        tokenizer = AutoTokenizer.from_pretrained(policy_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[Model] Initializing policy model from config (random weights): {policy_model}")
        config = AutoConfig.from_pretrained(policy_model)
        policy = AutoModelForCausalLM.from_config(config).to(device)
        
        #training the randomized model
        nll_batch_size = nll_batch_size or batch_size

        optimizer_steps_total = 0 
        cycle_pos = 0
        cycle_len = nll_steps_per_cycle + dpo_steps_per_cycle

        def want_nll_step():
            if optimizer_steps_total < nll_warmup_steps:
                return True
            if cycle_len == 0:
                return False
            return (cycle_pos % cycle_len) < nll_steps_per_cycle

        def want_dpo_step():
            if optimizer_steps_total < nll_warmup_steps:
                return False
            if cycle_len == 0:
                return True
            return (cycle_pos % cycle_len) >= nll_steps_per_cycle

        print(f"[Listener] Initializing BERTScore listener: {listener_model_type}")
        listener = BERTScoreListener(
            model_type=listener_model_type, 
            batch_size=listener_batch_size, 
            device=device,
            rescale_with_baseline=False  #avoid negative scores
        )

        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

        print(f"[Data] Loading data from: {input_path}")
        examples = list(SimpleWikiPassageLoader(path=input_path, limit=None))
        train_examples, test_examples = split_train_test_examples(examples, test_size, split_seed)
        print(
            f"[Data] Split sizes -> total={len(examples)}, "
            f"train={len(train_examples)}, test={len(test_examples)}"
        )

        if len(train_examples) == 0:
            raise ValueError("Train split is empty. Reduce --test_size or provide more data.")

        if wandb_project is not None:
            wandb.log(
                {
                    "data_total": len(examples),
                    "data_train": len(train_examples),
                    "data_test": len(test_examples),
                }
            )
            if wandb.run is not None:
                wandb.run.summary["data_total"] = len(examples)
                wandb.run.summary["data_train"] = len(train_examples)
                wandb.run.summary["data_test"] = len(test_examples)

        sample_eval_prompts = []

        for ex in test_examples[:3]:  # uses fixed test examples to compare change
            sample_eval_prompts.append(make_prompt(ex["passage"]))

        policy.train()
        global_step = 0
        total_kept = 0
        total_skipped = 0

        lm_texts = []

        for e in range(epochs):
            print(f"\n{'='*50}")
            print(f"EPOCH {e+1}/{epochs}")
            print(f"{'='*50}")
            
            prompts, chosen, rejected = [], [], []
            kept, skipped = 0, 0
            dropped_after_collate = 0
            total_dropped_after_collate = 0

            for example in train_examples:
                utterance = example['passage']
                prompt = make_prompt(utterance)
                reference_text = utterance

                #for token pred
                lm_texts.append(utterance)

                if len(lm_texts) >= nll_batch_size and want_nll_step():
                    lm_batch = collate_lm(tokenizer, lm_texts[:nll_batch_size], max_length=max_length)
                    lm_batch = {k: v.to(device) for k, v in lm_batch.items()}
                    loss_nll = nll_loss(policy, lm_batch)

                    loss_value = float(loss_nll.item())
                    loss_scaled = loss_nll / grad_accum
                    loss_scaled.backward()
                    global_step += 1

                    #log for token pred
                    if wandb_project is not None:
                        wandb.log(
                            {
                                "train/nll_loss_raw": loss_value,
                                "epoch": e,
                                "global_step": global_step,
                                "optimizer_steps_total": optimizer_steps_total,
                                "schedule/in_warmup": int(optimizer_steps_total < nll_warmup_steps),
                                "schedule/want_nll": int(True),
                                "schedule/want_dpo": int(False),
                            }
                        )
                    
                    if global_step % grad_accum == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                        optimizer_steps_total += 1
                        cycle_pos += 1

                        # Save checkpoint
                        save_interval = 500
                        if optimizer_steps_total % save_interval == 0:
                            checkpoint_path = os.path.join(output_path, f"checkpoint-{optimizer_steps_total}")
                            os.makedirs(checkpoint_path, exist_ok=True)
                            policy.save_pretrained(checkpoint_path)
                            tokenizer.save_pretrained(checkpoint_path)
                            print(f"[Checkpoint] Saved at step {optimizer_steps_total}: {checkpoint_path}")
                            if wandb_project is not None:
                                wandb.log({"checkpoint_step": optimizer_steps_total, "global_step": global_step})

                    lm_texts = lm_texts[nll_batch_size:]  # drop consumed texts

                if want_dpo_step():

                    seed_a = torch.randint(0, 2**31 - 1, (1,)).item()

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
                        seed_b = torch.randint(0, 2**31 - 1, (1,)).item()
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
                            dropped_after_collate += 1
                            total_dropped_after_collate += 1
                            prompts, chosen, rejected = [], [], []
                            continue

                        batch = PairBatch(
                            batch.ids_c.to(device), batch.attn_c.to(device), batch.labels_c.to(device),
                            batch.ids_r.to(device), batch.attn_r.to(device), batch.labels_r.to(device),
                        )

                        loss, metrics = dpo_loss(policy, batch, e, epochs, alpha, alpha_k, beta=beta)

                        loss_value = loss.item()
                        train_loss_sma = _update_sma(train_loss_sma_values, loss_value)
                        train_loss_ema = _update_ema(train_loss_ema, train_loss_sma, ema_alpha)
                        loss = loss / grad_accum

                        loss.backward()
                        global_step += 1

                        # Log to wandb
                        if wandb_project is not None:
                            wandb.log({
                                "train/loss_raw": loss_value,
                                "train/loss": train_loss_sma,
                                "train/loss_sma": train_loss_sma,
                                "train/loss_ema": train_loss_ema,
                                "epoch": e,
                                "global_step": global_step,
                                **metrics,
                                "kept_pairs": kept,
                                "skipped_pairs": skipped,
                                "dropped_after_collate_pairs": dropped_after_collate,
                                "total_kept": total_kept,
                                "total_skipped": total_skipped,
                                "total_dropped_after_collate_pairs": total_dropped_after_collate
                            })

                        # Optimizer step with gradient accumulation
                        if global_step % grad_accum == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                            optimizer_steps_total += 1
                            cycle_pos += 1
                            
                            # Save checkpoints
                            save_interval = 500
                            if optimizer_steps_total % save_interval == 0:
                                checkpoint_path = os.path.join(output_path, f"checkpoint-{optimizer_steps_total}")
                                os.makedirs(checkpoint_path, exist_ok=True)
                                policy.save_pretrained(checkpoint_path)
                                tokenizer.save_pretrained(checkpoint_path)
                                print(f"[Checkpoint] Saved at step {optimizer_steps_total}: {checkpoint_path}")
                                
                                # Log checkpoint to wandb
                                if wandb_project is not None:
                                    wandb.log({
                                        "checkpoint_step": optimizer_steps_total,
                                        "global_step": global_step,
                                    })

                        prompts, chosen, rejected = [], [], []

                    # Periodic logging
                    if (global_step + 1) % 10 == 0:
                        print(f"[Epoch {e+1}] Steps={global_step} | Kept={kept} | Skipped={skipped}")
            

            print(
                f"\n[Epoch {e+1} Complete] Total pairs kept: {kept}, skipped: {skipped}, "
                f"dropped_after_collate: {dropped_after_collate}"
            )

            if run_validation and len(test_examples) > 0:
                policy.eval()

                val_examples = test_examples
                if validation_max_examples is not None and validation_max_examples > 0:
                    val_examples = test_examples[:validation_max_examples]

                val_rng = random.Random(split_seed + e)
                val_gaps = []
                val_kept = 0
                val_skipped = 0
                val_effective_kept = 0
                val_dropped_after_collate = 0

                val_loss_sum = 0.0
                val_loss_n = 0

                val_surprisal_sum = 0.0
                val_surprisal_n = 0

                with torch.no_grad():
                    for example in val_examples:
                        utterance = example["passage"]
                        prompt = make_prompt(utterance)
                        reference_text = utterance

                        enc = tokenizer(
                            utterance,
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_length,
                        ).to(device)
                        out = policy(**enc, labels=enc["input_ids"])
                        val_surprisal_sum += float(out.loss.item())
                        val_surprisal_n += 1
                        
                        seed_a = val_rng.randrange(0, 2**31 - 1)

                        candidate_a = generate_summary(
                            policy, tokenizer, prompt,
                            top_p=top_p, temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            repetition_penalty=repetition_penalty,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            seed=seed_a
                        )
                        for _ in range(max_resample_tries + 1):
                            seed_b = val_rng.randrange(0, 2**31 - 1)
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

                        preferred = listener.prefer(candidate_a, candidate_b, reference_text)
                        gap = abs(float(preferred["score_a"]) - float(preferred["score_b"]))
                        val_gaps.append(gap)

                        if gap < val_score_gap_min:
                            val_skipped += 1
                        else:
                            val_kept += 1

                            if preferred["preferred"] == "A":
                                c, r = candidate_a, candidate_b
                            else:
                                c, r = candidate_b, candidate_a

                            vb = collate_pairs(tokenizer, [prompt], [c], [r], max_length=max_length)
                            if vb.ids_c.size(0) == 0:
                                val_dropped_after_collate += 1
                                continue

                            val_effective_kept += 1
                            vb = PairBatch(
                                vb.ids_c.to(device), vb.attn_c.to(device), vb.labels_c.to(device),
                                vb.ids_r.to(device), vb.attn_r.to(device), vb.labels_r.to(device),
                            )
                            vloss, _ = dpo_loss(policy, vb, e, epochs, alpha, alpha_k, beta=beta)
                            val_loss_sum += float(vloss.item())
                            val_loss_n += 1

                val_total = val_kept + val_skipped
                val_avg_gap = float(sum(val_gaps) / len(val_gaps)) if val_gaps else 0.0
                val_keep_rate = (float(val_kept) / val_total) if val_total > 0 else 0.0
                val_loss = (val_loss_sum / val_loss_n) if val_loss_n > 0 else None
                val_surprisal = (val_surprisal_sum / val_surprisal_n) if val_surprisal_n > 0 else None

                print(
                    f"[Epoch {e+1}] Validation: total={val_total}, kept={val_kept}, "
                    f"skipped={val_skipped}, keep_rate={val_keep_rate:.4f}, avg_gap={val_avg_gap:.4f}, "
                    f"effective_kept={val_effective_kept}, dropped_after_collate={val_dropped_after_collate}, "
                    f"val_loss_batches={val_loss_n}, val_loss={(val_loss if val_loss is not None else 'NA')}",
                    f"val_surprisal={(val_surprisal if val_surprisal is not None else 'NA')}",
                    flush=True
                )

                if wandb_project is not None:
                    payload = {
                        "epoch": e,
                        "global_step": global_step,
                        "val_total_pairs": val_total,
                        "val_kept_pairs": val_kept,
                        "val_skipped_pairs": val_skipped,
                        "val_keep_rate": val_keep_rate,
                        "val_avg_score_gap": val_avg_gap,
                        "val_effective_kept_pairs": val_effective_kept,
                        "val_dropped_after_collate_pairs": val_dropped_after_collate,
                        "val_loss_batches": val_loss_n,
                    }
                    if val_loss is not None:
                        val_loss_ema = _update_ema(val_loss_ema, val_loss, ema_alpha)
                        payload["val/loss"] = val_loss
                        payload["val/loss_ema"] = val_loss_ema
                    if val_surprisal is not None:
                        payload["val/surprisal"] = val_surprisal
                    wandb.log(payload)

                policy.train()
            
            #check qualitative output on sample eval
            for i, prompt in enumerate(sample_eval_prompts):
                inspect_top_p_tokens(policy, tokenizer, prompt, top_p=top_p, top_n_to_show=20)
                sample = quick_generate_sample(
                    policy,
                    tokenizer,
                    prompt,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=32,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )

                print(f"\n[Sample {i}]")
                print(sample)

        # Save final model
        final_path = os.path.join(output_path, "final_model")
        os.makedirs(final_path, exist_ok=True)
        policy.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"\nTraining Complete. Final model saved to: {final_path}")
    
    finally:
        if sweep_mode:
            print(f"[Cleanup] Deleting run folder: {output_path}")
            shutil.rmtree(output_path, ignore_errors=True)

    if wandb_project is not None:
        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--policy_model", type=str, required=True, default='gpt2')
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--alpha_k", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_loss_ema_alpha", type=float, default=0.03)
    parser.add_argument("--train_loss_sma_window", type=int, default=25)
    
    # Generation arguments
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    
    # Preference filtering arguments
    parser.add_argument("--score_gap_min", type=float, default=1e-4)
    parser.add_argument("--val_score_gap_min", type=float, default=0.0)
    parser.add_argument("--max_pair_similarity", type=float, default=0.85)
    parser.add_argument("--max_resample_tries", type=int, default=2)
    
    # Listener arguments
    parser.add_argument("--listener_model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--listener_batch_size", type=int, default=8)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--run_validation", action="store_true")
    parser.add_argument("--validation_max_examples", type=int, default=512)
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    #training next token predict arguments
    parser.add_argument("--nll_warmup_steps", type=int, default=0)
    parser.add_argument("--nll_steps_per_cycle", type=int, default=2)
    parser.add_argument("--dpo_steps_per_cycle", type=int, default=1)
    parser.add_argument("--nll_batch_size", type=int, default=8)

    return parser.parse_args()

def main():
    args = parse_args()

    if not 0.0 < args.train_loss_ema_alpha <= 1.0:
        raise ValueError(f"--train_loss_ema_alpha must be in (0, 1], got {args.train_loss_ema_alpha}")
    if args.train_loss_sma_window < 1:
        raise ValueError(f"--train_loss_sma_window must be >= 1, got {args.train_loss_sma_window}")

    train_dpo(
        policy_model=args.policy_model,
        input_path=args.input_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        alpha=args.alpha,
        alpha_k=args.alpha_k,
        beta=args.beta,
        max_length=args.max_length,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        score_gap_min=args.score_gap_min,
        val_score_gap_min=args.val_score_gap_min,
        max_pair_similarity=args.max_pair_similarity,
        max_resample_tries=args.max_resample_tries,
        listener_model_type=args.listener_model_type,
        listener_batch_size=args.listener_batch_size,
        test_size=args.test_size,
        split_seed=args.split_seed,
        run_validation=args.run_validation,
        validation_max_examples=args.validation_max_examples,
        train_loss_ema_alpha=args.train_loss_ema_alpha,
        train_loss_sma_window=args.train_loss_sma_window,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        nll_warmup_steps=args.nll_warmup_steps,
        nll_steps_per_cycle=args.nll_steps_per_cycle,
        dpo_steps_per_cycle=args.dpo_steps_per_cycle,
        nll_batch_size=args.nll_batch_size
    )

if __name__ == "__main__":
    main()
