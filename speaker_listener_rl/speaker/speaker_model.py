import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.utils import generate_summary, jaccard_ngrams, make_prompt
#from listener.listener.bertscore_listener import BERTScoreListener
# preference_dataset = load_dataset("json", data_files="data/preferences_train.jsonl")["train"]


def load_data(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): #filters out empty lines
                continue
            rows.append(json.loads(line))
    return rows

def generate_summary_pairs(
        speaker_model_name,
        dataset_path, #json input path
        output_path, #output path
        *,
        top_p,
        temperature,
        max_new_tokens,
        repetition_penalty,
        no_repeat_ngram_size,
        max_resample_tries, #how many times the model will try to generate a different summary
        max_jaccard_similarity #between summaries so that they are "different" enough
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(speaker_model_name) #should be gpt2 to start
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token #make sure that pad_token is specified

    model = AutoModelForCausalLM.from_pretrained(speaker_model_name).to(device)
    model.eval()

    examples = load_data(dataset_path)

    text_field = 'source_text' # is the field that the json file stores the source

    total = 0
    resampled = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for example in examples:
            total += 1
            source = example[text_field]
            prompt = make_prompt(source)

            a, b = '', ''
            for attempt in range(max_resample_tries):
                seed_a = torch.randint(0, 2**31 - 1, (1,)).item() # generate random seeds 
                seed_b = torch.randint(0, 2**31 - 1, (1,)).item()

                a = generate_summary(
                    model,
                    tokenizer,
                    prompt,
                    top_p,
                    temperature,
                    max_new_tokens,
                    repetition_penalty,
                    no_repeat_ngram_size,
                    seed=seed_a
                )

                b = generate_summary(
                    model,
                    tokenizer,
                    prompt,
                    top_p,
                    temperature,
                    max_new_tokens,
                    repetition_penalty,
                    no_repeat_ngram_size,
                    seed=seed_b
                )

                sim = jaccard_ngrams(a, b, n=2)
                if sim <= max_jaccard_similarity:
                    break
                resampled += 1

            record = { #logs for reproducibility
                "prompt": prompt,
                "cand_a": a,
                "cand_b": b,
                "speaker_model": speaker_model_name,
                "top_p": top_p,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print (f'speaker wrote {total} examples to {output_path}, resampled: {resampled} times')

    return None

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--speaker_model", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    parser.add_argument("--max_resample_tries", type=int, default=2)
    parser.add_argument("--max_jaccard_similarity", type=float, default=0.85)

    return parser.parse_args()

def main():
    args = parse_args()

    generate_summary_pairs(
        speaker_model_name=args.speaker_model,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_resample_tries=args.max_resample_tries,
        max_jaccard_similarity=args.max_jaccard_similarity,
    )

if __name__ == "__main__":
    main()