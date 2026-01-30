import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from speaker_listener_rl.data.dataloader_wiki import SimpleWikiPassageLoader


def build_prompt(passage: str) -> str:
    return (
        passage
        + "\n\nSummary:"
    )


# def compute_length_penalty(num_gen_tokens: int, k: int, lam: float = 0.2) -> float:
#     """
#     Soft penalty for exceeding K tokens.
#     - If <=K: 0 penalty
#     - If >K: penalty grows linearly with overflow 
#     """
#     if num_gen_tokens <= k:
#         return 0.0
#     overflow = num_gen_tokens - k
#     return lam * (overflow / max(k, 1))
def compute_length_penalty(num_gen_tokens: int, k: int, lam: float = 0.2) -> float:
    """
    Soft penalty for exceeding K tokens (linear).
    penalty = lam * max(0, num_gen_tokens - k)
    """
    overflow = max(0, num_gen_tokens - k)
    return lam * overflow


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, "data")

    in_path = os.path.join(data_dir, "simple_wiki_passages.jsonl")
    out_path = os.path.join(repo_root, "outputs", "speaker_summaries_simplewiki_K10.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    MODEL_NAME = "gpt2"
    K = 10

    STRICT_LIMIT = False
    # BUFFER = 10  # only used if STRICT_LIMIT=False (allows going beyond K, then penalize)

    NUM_BEAMS = 4
    LENGTH_PENALTY = 2.0 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    loader = SimpleWikiPassageLoader(in_path)
    MAX_EXAMPLES = 500

    with open(out_path, "w", encoding="utf-8") as out:
        for i, ex in enumerate(loader):

            if i >= MAX_EXAMPLES:
                break

            passage = ex["passage"]
            prompt = build_prompt(passage)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            max_new = K if STRICT_LIMIT else 100
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    top_p=0.9,            
                    temperature=0.8,    
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            new_token_ids = gen_ids[0][prompt_len:]
            num_gen_tokens = int(new_token_ids.shape[0])

            summary = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

            overflow_pen = compute_length_penalty(num_gen_tokens, K, lam=0.2)

            if not STRICT_LIMIT and num_gen_tokens > K:
                truncated_ids = new_token_ids[:K]
                summary_truncated = tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()
            else:
                summary_truncated = summary

            record = {
                "id": ex.get("id", i),
                "source": ex.get("source", "simplewiki"),
                # "k": K,
                # "strict_limit": STRICT_LIMIT,
                "passage": passage,
                "speaker_summary": summary_truncated,
                # "raw_generated_tokens": num_gen_tokens,
                "length_penalty_value": overflow_pen,
                # "generation_config": {
                #     "model": MODEL_NAME,
                #     "max_new_tokens": max_new,
                #     "num_beams": NUM_BEAMS,
                #     "hf_length_penalty": LENGTH_PENALTY,
                #     "no_repeat_ngram_size": 3,
                # },
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                print(f"[{i+1}] wrote summaries -> {out_path}")

    print(f"Done. Wrote summaries to {out_path}")


if __name__ == "__main__":
    main()