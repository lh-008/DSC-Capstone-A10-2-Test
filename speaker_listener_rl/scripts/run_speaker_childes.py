from speaker_listener_rl.data.dataloader_childes import ChildesUtteranceLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

def build_prompt(utterance: str) -> str:
    return (
        "UTTERANCE:\n" + utterance + "\n\n"
        "TELEGRAPHIC (2 tokens max):\n"
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    loader = ChildesUtteranceLoader("childes_utterances.jsonl")

    out_path = "speaker_listener_rl/outputs/generations_childes_K2.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        # for ex in loader:
        for i, ex in enumerate(loader):
            if i >= 500:
                break
   
            prompt = build_prompt(ex["utterance"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # outputs = model.generate(
            #     **inputs,
            #     max_new_tokens=2,     # K = 2 hard constraint
            #     do_sample=True,
            #     top_p=0.9,
            #     temperature=0.7,
            #     pad_token_id=tokenizer.eos_token_id,
            # )
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=True,
                top_p=0.6,         
                temperature=0.5,   
                top_k=50,           
                repetition_penalty=1.1,
                bad_words_ids=[
                    [tokenizer.encode("1", add_special_tokens=False)[0]],
                    [tokenizer.encode("-", add_special_tokens=False)[0]],
                ],
                pad_token_id=tokenizer.eos_token_id,
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = decoded.split("TELEGRAPHIC (2 tokens max):")[-1].strip()

            record = {
                "id": ex.get("id"),
                "K": 2,
                "summary": summary,
                "utterance": ex["utterance"],
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(record)

    print(f"\nSaved generations to: {out_path}")

if __name__ == "__main__":
    main()
