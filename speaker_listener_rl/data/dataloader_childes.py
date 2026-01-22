import json
import os
from typing import Iterator, Dict, Any

class ChildesUtteranceLoader:
    """
    Loads CHILDES-only utterances from speaker_listener_rl/data/childes_utterances.jsonl
    Yields dicts with keys: id, utterance (and optional source)
    """

    def __init__(self, path: str = "childes_utterances.jsonl"):
        base_dir = os.path.dirname(__file__)  
        self.path = path if os.path.isabs(path) else os.path.join(base_dir, path)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if "utterance" not in ex:
                    raise ValueError(f"Missing 'utterance' key in: {ex}")
                yield ex
