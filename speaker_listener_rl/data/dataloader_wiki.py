import json
from typing import Dict, Iterator, Optional


class SimpleWikiPassageLoader:
    """
    Iterates over simplewiki_passages.jsonl and yields dict examples.

    Expected jsonl schema:
      {"id": int, "source": "simplewiki", "passage": str}
    """

    def __init__(self, path: str, limit: Optional[int] = None):
        self.path = path
        self.limit = limit

    def __iter__(self) -> Iterator[Dict]:
        n = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)

                if "passage" not in ex:
                    continue

                yield ex
                n += 1
                if self.limit is not None and n >= self.limit:
                    break