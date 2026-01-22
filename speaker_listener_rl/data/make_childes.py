import json
import re

IN_PATH = "speaker_listener_rl/data/train_10M/childes.train"
OUT_PATH = "speaker_listener_rl/data/childes_utterances.jsonl"

def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line

def main(max_lines=5000, min_words=2, max_words=30):
    n = 0
    with open(IN_PATH, "r", encoding="utf-8") as f, open(OUT_PATH, "w", encoding="utf-8") as out:
        for line in f:
            line = clean_line(line)
            if not line:
                continue

            wc = len(line.split())
            if wc < min_words or wc > max_words:
                continue

            ex = {"id": n, "source": "childes", "utterance": line}
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1

            if n >= max_lines:
                break

    print(f"Wrote {n} utterances to {OUT_PATH}")

if __name__ == "__main__":
    main()
