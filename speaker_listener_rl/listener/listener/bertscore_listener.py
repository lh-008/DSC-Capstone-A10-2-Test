import torch
from bert_score import BERTScorer


class BERTScoreListener:
    """
    Listener that scores candidate text against a reference text using BERTScore.:
      - reference = original passage (or target meaning text)
      - candidate = speaker summary
      - reward = BERTScore F1
    """

    def __init__(
        self,
        model_type="roberta-large",
        lang="en",
        device=None,
        batch_size=16,
        use_idf=False,
        rescale_with_baseline=True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_type = model_type
        self.lang = lang
        self.device = device
        self.batch_size = batch_size
        self.use_idf = use_idf
        self.rescale_with_baseline = rescale_with_baseline

        self.scorer = BERTScorer(
            model_type=self.model_type,
            lang=self.lang,
            device=self.device,
            batch_size=self.batch_size,
            idf=self.use_idf,
            rescale_with_baseline=self.rescale_with_baseline,
        )

    @torch.inference_mode()
    def score_pair(self, candidate, reference):
        """
        Score ONE candidate against ONE reference.

        Parameters:
            candidate : str
                The text produced by the speaker (e.g., a summary).
            reference : str
                The text we compare against (e.g., passage or gold summary).

        Returns:
            float:
                BERTScore F1 for this pair (higher = more similar).
        """
        P, R, F1 = self.scorer.score([candidate], [reference])
        return float(F1[0].item())

    @torch.inference_mode()
    def score_batch(self, candidates, references):
        if len(candidates) != len(references):
            raise ValueError("candidates and references must have same length")

        P, R, F1 = self.scorer.score(candidates, references)
        return [float(x) for x in F1.detach().cpu().tolist()]

    def prefer(self, cand_a, cand_b, reference):
        scores = self.score_batch([cand_a, cand_b], [reference, reference])
        score_a, score_b = scores[0], scores[1]

        preferred = "A" if score_a >= score_b else "B"
        return {"preferred": preferred, "score_a": score_a, "score_b": score_b}
