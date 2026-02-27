import json
import os

from make_wiki import IN_PATH, chunk_sentences, clean_line


BASE_DIR = os.path.dirname(__file__)
OUT_PATH = os.path.join(BASE_DIR, "simple_wiki_passages_8k.jsonl")


def main(
    min_words: int = 20,
    max_words: int = 120,
    target_passage_words: int = 80,
    max_instances: int = 8000,
):
    """
    Reads simple_wiki.train from train_100M, applies the same cleaning/chunking
    logic as make_wiki.py, and writes exactly up to 8k records by default.
    """
    n = 0

    with open(IN_PATH, "r", encoding="utf-8") as f, open(OUT_PATH, "w", encoding="utf-8") as out:
        for raw_line in f:
            line = clean_line(raw_line)
            if not line:
                continue

            for passage in chunk_sentences(line, target_words=target_passage_words):
                wc = len(passage.split())
                if wc < min_words or wc > max_words:
                    continue

                ex = {"id": n, "source": "simplewiki", "passage": passage}
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
                if n >= max_instances:
                    break

            if n >= max_instances:
                break

    print(f"Wrote {n} passages to {OUT_PATH} from {IN_PATH}")


if __name__ == "__main__":
    main()
