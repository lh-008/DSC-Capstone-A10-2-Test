import json
import os
from typing import Iterator, Dict, Any

class PassageQuestionLoader:
    """
    Simple dataloader that yields (passage, question) pairs
    from data/passage_question_train.jsonl
    """

    def __init__(self, path: str = "passage_question_train.jsonl"):
        base_dir = os.path.dirname(__file__)
        if os.path.isabs(path):
            self.path = path
        elif os.path.exists(path):
            self.path = path
        else:
            self.path = os.path.join(base_dir, path)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yields:
            {
              "id": int,
              "passage": str,
              "question": str,
              "source": str (optional)
            }
        """
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)

                if "passage" not in ex or "question" not in ex:
                    raise ValueError(f"Missing keys in example: {ex}")

                yield ex
