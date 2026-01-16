import argparse
import json
from tqdm import tqdm

from listener.bertscore_listener import BERTScoreListener


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--model_type", default="roberta-large")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    listener = BERTScoreListener(
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=args.device,
    )

    rows = []
    candidates = []
    references = []

    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rows.append(ex)
            candidates.append(ex["candidate"])
            references.append(ex["reference"])

    scores = []
    bs = args.batch_size
    for i in tqdm(range(0, len(rows), bs), desc="Scoring"):
        c_batch = candidates[i:i + bs]
        r_batch = references[i:i + bs]
        scores.extend(listener.score_batch(c_batch, r_batch))

    with open(args.outfile, "w", encoding="utf-8") as out:
        for r, s in zip(rows, scores):
            r["bertscore_f1"] = s
            out.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
