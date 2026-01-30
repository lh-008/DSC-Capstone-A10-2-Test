import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="output")
    parser.add_argument("--prompt", type=str, default="Write a one-sentence summary of: The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(args.prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
            num_return_sequences=args.num_return_sequences
        )

    for i, seq in enumerate(outputs):
        text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"=== SAMPLE {i} ===")
        print(text)
        print()


if __name__ == "__main__":
    main()
