import json

IN_PATH = "speaker_listener_rl/outputs/generations_childes_K2.jsonl"
OUT_PATH = "speaker_listener_rl/outputs/childes_pairs_K2.jsonl"

def main():
    n = 0
    with open(IN_PATH, "r", encoding="utf-8") as f, open(OUT_PATH, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            pair = {
                "candidate": ex["summary"],
                "reference": ex["utterance"],
                "meta": {"id": ex.get("id"), "K": ex.get("K", 2)}
            }
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} pairs to {OUT_PATH}")

if __name__ == "__main__":
    main()
