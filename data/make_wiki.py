import json
import os
import re
from typing import Optional


BASE_DIR = os.path.dirname(__file__)
IN_PATH = os.path.join(BASE_DIR, "train_100M", "simple_wiki.train")
OUT_PATH = os.path.join(BASE_DIR, "simple_wiki_passages.jsonl")


def clean_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    # Remove common wiki-ish markup / artifacts (keep it conservative)
    line = re.sub(r"\[[^\]]*\]", " ", line)          # [brackets]
    line = re.sub(r"\{\{[^}]*\}\}", " ", line)       # {{templates}}
    line = re.sub(r"<[^>]*>", " ", line)             # <tags>
    line = re.sub(r"={2,}.*?={2,}", " ", line)       # == headings ==
    line = re.sub(r"\s+", " ", line).strip()

    # Filter out non-linguistic lines
    if not re.search(r"[A-Za-z]", line):
        return ""

    return line


def chunk_sentences(text: str, target_words: int) -> list[str]:
    """
    Split into rough sentences and pack into passages ~target_words.
    Keeps it simple and robust for a .train file that's already sentence-ish.
    """
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return []

    passages = []
    cur = []
    cur_wc = 0

    for s in sents:
        wc = len(s.split())
        # If one sentence is huge, allow it alone (it'll get filtered later)
        if cur and (cur_wc + wc) > target_words:
            passages.append(" ".join(cur).strip())
            cur = [s]
            cur_wc = wc
        else:
            cur.append(s)
            cur_wc += wc

    if cur:
        passages.append(" ".join(cur).strip())

    return passages


def main(
    min_words: int = 20,
    max_words: int = 120,
    target_passage_words: int = 80,
    max_instances: Optional[int] = None,
):
    """
    Reads simple_wiki.train from train_100M, cleans text, chunks into
    multi-sentence passages, and writes jsonl records: {id, source, passage}.

    `max_instances=None` writes all matching passages.
    """
    n = 0

    with open(IN_PATH, "r", encoding="utf-8") as f, open(OUT_PATH, "w", encoding="utf-8") as out:
        for raw_line in f:
            line = clean_line(raw_line)
            if not line:
                continue

            # Turn each cleaned line into 1+ passages (in case lines are long)
            for passage in chunk_sentences(line, target_words=target_passage_words):
                wc = len(passage.split())
                if wc < min_words or wc > max_words:
                    continue

                ex = {"id": n, "source": "simplewiki", "passage": passage}
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
                if max_instances is not None and n >= max_instances:
                    break

            if max_instances is not None and n >= max_instances:
                break

    print(f"Wrote {n} passages to {OUT_PATH} from {IN_PATH}")


if __name__ == "__main__":
    main()
