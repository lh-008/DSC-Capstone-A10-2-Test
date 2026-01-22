import json
import re
import os

# IN_PATH = "../speaker_listener_rl/data/train_10M/childes.train"
# OUT_PATH = "speaker_listener_rl/data/childes_utterances.jsonl"
BASE_DIR = os.path.dirname(__file__)  # speaker_listener_rl/data
IN_PATH = os.path.join(BASE_DIR, "train_10M", "childes.train")
OUT_PATH = os.path.join(BASE_DIR, "childes_utterances.jsonl")

def clean_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    # if not (line.startswith("*MOT:") or line.startswith("*CHI:")):
    #     return ""

    line = re.sub(r"^\*[A-Z]{3}:\s*", "", line)

    line = re.sub(r"\[[^\]]*\]", " ", line)

    line = re.sub(r"\s+", " ", line).strip()

    if not re.search(r"[A-Za-z]", line):
        return ""

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
