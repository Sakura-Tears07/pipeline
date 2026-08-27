"""Qwen3-Embedding listwise predictor（A/B/C 统一；CPU 预取）。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_FIND_MODEL = _PIPELINE_ROOT.parent / "find_model"
if str(_FIND_MODEL) not in sys.path:
    sys.path.insert(0, str(_FIND_MODEL))

from common.find_model_paths import MAX_CANDIDATES, QWEN3_EMB_CKPT
from common.sample_to_listwise import pipeline_sample_to_listwise
from common.uncertainty import compute_uncertainty


@dataclass
class PredictResult:
    raw: str
    function_call: Optional[str]
    backend: str
    scores: List[float] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


def _top1_from_scores(names: List[str], scores: List[float]) -> str:
    if not names or not scores:
        return ""
    idx = max(range(len(scores)), key=lambda i: scores[i])
    return names[idx]


class Qwen3EmbListwisePredictor:
    """微调 Qwen3-Embedding-0.6B bi-encoder（默认 CPU 预取）。"""

    def __init__(self, *, device: torch.device, checkpoint: str = ""):
        from src.finetune.bge_listwise import (
            BgeListwiseConfig,
            _forward_batch,
            _make_collator,
            _make_dataset,
            _prepare_tokenizer,
            _resolve_tokenizer_path,
        )
        from src.finetune.candidate_encoder import BiEncoder
        from src.finetune.bge_state import QWEN_QUERY_INSTRUCT

        ckpt = str(checkpoint or QWEN3_EMB_CKPT)
        if not Path(ckpt).is_dir() or not (Path(ckpt) / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"Qwen3-Embedding checkpoint missing: {ckpt} "
                "(expected /data/model/Qwen3-Embedding-0.6B-finetuned)"
            )
        self.device = device
        fields = BgeListwiseConfig.__dataclass_fields__
        cfg_kw = {
            "model_path": ckpt,
            "padding_side": "left",
            "query_instruct": QWEN_QUERY_INSTRUCT,
            "unique_functions": True,
            "gradient_checkpointing": False,
            "max_query_length": 512,
            "max_tool_length": 256,
            "max_candidates": MAX_CANDIDATES,
            "temperature": 0.05,
            "hard_neg_lambda": 0.0,
        }
        self.cfg = BgeListwiseConfig(**{k: v for k, v in cfg_kw.items() if k in fields})
        if device.type == "cpu":
            torch.set_num_threads(max(1, min(8, int(torch.get_num_threads() or 8))))
        dtype = torch.float32 if device.type == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        tok_path = _resolve_tokenizer_path(ckpt, self.cfg)
        self.tokenizer = _prepare_tokenizer(tok_path, self.cfg)
        self.model = BiEncoder(
            ckpt,
            logit_scale=1.0 / float(self.cfg.temperature),
            dtype=dtype,
            hard_neg_lambda=0.0,
        ).to(device)
        self.model.eval()
        self._collate = _make_collator(self.tokenizer, self.cfg)
        self._make_dataset = _make_dataset
        self._forward_batch = _forward_batch
        self.backend = f"qwen3_emb:{Path(ckpt).name}"

    @torch.inference_mode()
    def predict_sample(self, sample: Dict[str, Any]) -> PredictResult:
        lw = pipeline_sample_to_listwise(sample, max_candidates=MAX_CANDIDATES)
        names = [c["name"] for c in (lw.get("candidates") or [])]
        if not names:
            return PredictResult(raw="{}", function_call=None, backend=self.backend)
        if int(lw.get("label_index", -1) or -1) < 0:
            lw["label_index"] = 0
        ds = self._make_dataset([lw], self.cfg)
        if not ds.samples:
            return PredictResult(raw="{}", function_call=None, backend=self.backend)
        batch = self._collate([ds.samples[0]])
        out, _ = self._forward_batch(self.model, batch, self.device)
        k = int(ds.samples[0]["k"])
        scores = [float(x) for x in out["logits"][0, :k].detach().cpu().tolist()]
        names = list(ds.samples[0]["names"])
        pred = _top1_from_scores(names, scores)
        unc = compute_uncertainty(scores)
        payload = {
            "function_call": pred,
            "scores": scores,
            "candidates": names,
            "uncertainty": unc,
        }
        return PredictResult(
            raw=json.dumps(payload, ensure_ascii=False),
            function_call=pred or None,
            backend=self.backend,
            scores=list(scores),
            candidates=names,
            extras={"uncertainty": unc},
        )

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def make_predictor(backend: str, device: torch.device):
    b = (backend or "qwen3_emb").strip().lower()
    if b in ("none", "no", "off"):
        return None
    if b in ("qwen3_emb", "qwen3-emb", "qwen3_embedding", "embedding", "qwen"):
        return Qwen3EmbListwisePredictor(device=device)
    raise ValueError(f"unknown predictor backend (supported: none, qwen3_emb): {backend}")
