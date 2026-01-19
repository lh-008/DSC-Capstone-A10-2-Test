import json
from typing import Iterator, Dict, Any

class PassageQuestionLoader:
    """
    Simple dataloader that yields (passage, question) pairs
    from data/passage_question_train.jsonl
    """

    def __init__(self, path: str = "speaker_listener_rl/data/passage_question_train.jsonl"):
        self.path = path

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
