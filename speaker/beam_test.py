import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    prompt = (
        "PASSAGE:\n"
        "The city council approved a new policy to reduce emissions by expanding public transit.\n\n"
        "SUMMARY (keywords, very short):\n"
        "Example: emissions, transit, policy\n"
        "Now: \n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Beam search parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        num_beams=7,              # number of tracked sequences
        early_stopping=True,
        no_repeat_ngram_size=3,   
        length_penalty=-20,       # <1.0 favors shorter outputs, >1.0 favors longer
        num_return_sequences=3,
        do_sample=False,
    )

    # Decode
    prompt_len = inputs["input_ids"].shape[1]
    for i, out in enumerate(outputs):
        gen_ids = out[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"[{i}] {text}")

if __name__ == "__main__":
    main()
