from speaker_listener_rl.data.dataloader import PassageQuestionLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

def build_prompt(passage, question):
    return (
        "PASSAGE: " + passage + "\n"
        "QUESTION: " + question + "\n"
        "SUMMARY:"
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    # model = AutoModelForCausalLM.from_pretrained("speaker_listener_rl/outputs/dpo-speaker")
# 

    loader = PassageQuestionLoader()

    out_path = "speaker_listener_rl/outputs/generations_K2.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:

        for ex in loader:
            prompt = build_prompt(ex["passage"], ex["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=2,   # <-- K = 2 (hard constraint)
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = full_text.split("SUMMARY:")[-1].strip()

            record = {
                "id": ex["id"],
                "K": 2,
                "summary": summary,
                "passage": ex["passage"],
                "question": ex["question"],
            }

            out.write(json.dumps(record) + "\n")
            print(record)

if __name__ == "__main__":
    main()
