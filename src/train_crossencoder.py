"""Semantic matcher: Ditto-style Transformer cross-encoder (default backbone
DeBERTa-v3), PDF "Deep Semantic Matching with Transformer Cross-Encoders".

Serialisation (Ditto):
    [CLS] COL name VAL <name> COL address VAL <address> [SEP]
          COL name VAL <name> COL address VAL <address> [SEP]
The tokenizer inserts [CLS]/[SEP]; the [CLS] hidden state feeds a linear
head producing one match logit.

Training techniques from the PDF:
 1. Domain-knowledge injection: offline regexes + the parsed address wrap
    discriminative alphanumerics in special tags - postcodes as
    "[PC] 95113 [/PC]", other digit-bearing tokens (house numbers, suites) as
    "[NUM] 500 [/NUM]". The tags are added to the tokenizer vocabulary.
 2. MixDA augmentation (Ditto): an augmented copy of each training pair is
    produced with one operator - span deletion (del), span shuffle (swap),
    attribute drop (drop_col), token deletion (token_del), attribute
    shuffle (attr_shuffle) or entity swap (entry_swap) - and the [CLS]
    encodings of original and augmented input are interpolated,
    h = lam*h_orig + (1-lam)*h_aug, lam ~ Beta(a, a), lam = max(lam, 1-lam),
    before the classification head; the label is the original label.
 3. TF-IDF summarisation: values longer than `tfidf_summarize_max_words`
    keep only their highest-IDF words (original order preserved).
 4. Focal loss FL = -alpha_t (1 - p_t)^gamma log(p_t) against the extreme
    class imbalance of candidate pairs.
Hard negatives: the highest-scoring LightGBM negatives of each training
entity (plus a few random ones) form the negative training pairs.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import random
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from .train_biencoder import linear_warmup_schedule, require_torch, resolve_device
from .utils import ensure_dir, get_logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None

SPECIAL_TOKENS = ["[PC]", "[/PC]", "[NUM]", "[/NUM]"]
_UNIT_RE = re.compile(r"\[(PC|NUM)\] \S+ \[/\1\]|\S+")
_DIGIT_RE = re.compile(r"\d")


class DittoSerializer:
    def __init__(self, cfg: dict, idf: dict | None = None):
        ccfg = cfg.get("crossencoder", {})
        self.text_source = ccfg.get("text_source", "raw")
        self.domain_tags = bool(ccfg.get("domain_tags", True))
        self.max_words = int(ccfg.get("tfidf_summarize_max_words", 40))
        self.idf = idf or {}

    def _summarize(self, words: list[str]) -> list[str]:
        if len(words) <= self.max_words or not self.idf:
            return words
        default = max(self.idf.values()) if self.idf else 1.0
        scores = [self.idf.get(w.lower(), default) for w in words]
        keep = set(np.argsort(scores)[::-1][: self.max_words].tolist())
        return [w for i, w in enumerate(words) if i in keep]

    def _tag(self, words: list[str], postcode: str) -> list[str]:
        """Domain-knowledge injection: tag postcodes and digit-bearing tokens."""
        if not self.domain_tags:
            return words
        out = []
        pc = postcode.replace(" ", "").lower()
        for w in words:
            core = re.sub(r"[^\w\-/]", "", w).lower()
            is_pc = bool(pc and core) and (core == pc or core.startswith(pc + "-")
                                           or (len(core) >= 3 and core in pc and not core.isalpha()))
            if is_pc:
                out.append(f"[PC] {w} [/PC]")
            elif _DIGIT_RE.search(w):
                out.append(f"[NUM] {w} [/NUM]")
            else:
                out.append(w)
        return out

    def serialize_frame(self, df: pd.DataFrame) -> list[str]:
        if self.text_source == "raw":
            names, addrs = df["name"], df["address"]
        else:
            names, addrs = df["name_clean"], df["addr_clean"]
        out = []
        for n, a, pc in zip(names, addrs, df["addr_postcode"]):
            aw = self._tag(self._summarize(str(a).split()), pc)
            nw = self._tag(self._summarize(str(n).split()), "")
            out.append(f"COL name VAL {' '.join(nw)} COL address VAL {' '.join(aw)}")
        return out


# ------------------------------------------------------------------ MixDA
def _split_attrs(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"COL (\w+) VAL ?", text)
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def _join_attrs(attrs: list[tuple[str, str]]) -> str:
    return " ".join(f"COL {k} VAL {val}".strip() for k, val in attrs)


def augment_pair(a: str, b: str, op: str, rng: random.Random) -> tuple[str, str]:
    """One Ditto augmentation operator on a serialised pair."""
    if op == "entry_swap":
        return b, a
    side = rng.randint(0, 1)
    attrs = _split_attrs(a if side == 0 else b)
    if not attrs:
        return a, b
    if op == "attr_shuffle":
        rng.shuffle(attrs)
    elif op == "drop_col":
        i = rng.randrange(len(attrs))
        attrs[i] = (attrs[i][0], "")
    else:
        i = rng.randrange(len(attrs))
        units = [m.group(0) for m in _UNIT_RE.finditer(attrs[i][1])]
        if units:
            if op == "token_del" and len(units) > 1:
                del units[rng.randrange(len(units))]
            elif op in ("del", "swap") and len(units) > 1:
                span = rng.randint(1, min(4, len(units) - 1 if op == "del" else len(units)))
                start = rng.randrange(0, len(units) - span + 1)
                if op == "del":
                    del units[start:start + span]
                else:
                    seg = units[start:start + span]
                    rng.shuffle(seg)
                    units[start:start + span] = seg
            attrs[i] = (attrs[i][0], " ".join(units))
    new = _join_attrs(attrs)
    return (new, b) if side == 0 else (a, new)


if torch is not None:
    class CrossEncoder(nn.Module):
        def __init__(self, path: str, local_files_only: bool = True, dropout: float = 0.1):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(path, local_files_only=local_files_only)
            hidden = self.backbone.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)
            self._accepts = set(inspect.signature(self.backbone.forward).parameters)

        def encode(self, enc: dict) -> "torch.Tensor":
            kw = {k: val for k, val in enc.items() if k in self._accepts}
            return self.backbone(**kw).last_hidden_state[:, 0]

        def forward(self, enc: dict, aug_enc: dict | None = None, lam: float | None = None):
            h = self.encode(enc)
            if aug_enc is not None and lam is not None:
                h = lam * h + (1.0 - lam) * self.encode(aug_enc)
            return self.head(self.dropout(h)).squeeze(-1)

    def focal_loss(logits, targets, gamma: float = 2.0, alpha: float | None = None):
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * (1 - p_t) ** gamma
        if alpha is not None:
            loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
        return loss.mean()


def _amp_dtype(device):
    """bf16 where supported (DeBERTa-v3 is prone to fp16 overflow), else fp16."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _tokenize(tok, a: list[str], b: list[str], max_len: int, device):
    enc = tok(a, b, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return {k: val.to(device) for k, val in enc.items()}


def _load_tokenizer(path: str, local_only: bool):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=local_only)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


def train_crossencoder(train_a: list[str], train_b: list[str], train_y: np.ndarray, val_a: list[str],
                       val_b: list[str], val_y: np.ndarray, cfg: dict, out_dir: str, seed: int = 42) -> dict:
    require_torch()
    log = get_logger()
    ccfg = cfg.get("crossencoder", {})
    device = resolve_device(ccfg.get("device", "auto"))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    local_only = ccfg.get("local_files_only", True)
    tok = _load_tokenizer(ccfg["pretrained"], local_only)
    model = CrossEncoder(ccfg["pretrained"], local_only)
    model.backbone.resize_token_embeddings(len(tok))
    model.to(device)
    max_len = int(ccfg.get("max_length", 128))
    bs = int(ccfg.get("batch_size", 32))
    epochs = int(ccfg.get("epochs", 3))
    gamma = float(ccfg.get("focal_gamma", 2.0))
    alpha = ccfg.get("focal_alpha")
    alpha = None if alpha is None else float(alpha)
    mix = ccfg.get("mixda", {})
    use_mix = bool(mix.get("enabled", True))
    ops = list(mix.get("ops", ["del", "swap", "drop_col"]))
    mix_alpha = float(mix.get("alpha", 0.8))
    use_amp = bool(ccfg.get("fp16", True)) and device.type == "cuda"
    amp_dtype = _amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None
    n = len(train_y)
    steps = math.ceil(n / bs) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=float(ccfg.get("lr", 2e-5)),
                            weight_decay=float(ccfg.get("weight_decay", 0.01)))
    sched = linear_warmup_schedule(opt, int(float(ccfg.get("warmup_ratio", 0.1)) * steps), steps)
    ensure_dir(out_dir)
    best_ap, history = -1.0, []
    np_rng = np.random.default_rng(seed)
    log.info("Cross-encoder: %d training pairs (%d pos), %d validation pairs, device=%s, MixDA=%s",
             n, int(train_y.sum()), len(val_y), device, use_mix)
    for epoch in range(epochs):
        model.train()
        perm = np_rng.permutation(n)
        losses = []
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            a = [train_a[i] for i in idx]
            b = [train_b[i] for i in idx]
            y = torch.tensor(train_y[idx], dtype=torch.float32, device=device)
            enc = _tokenize(tok, a, b, max_len, device)
            aug_enc, lam = None, None
            if use_mix and ops:
                aug = [augment_pair(x, z, rng.choice(ops), rng) for x, z in zip(a, b)]
                aug_enc = _tokenize(tok, [p[0] for p in aug], [p[1] for p in aug], max_len, device)
                lam = float(np_rng.beta(mix_alpha, mix_alpha))
                lam = max(lam, 1 - lam)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(enc, aug_enc, lam)
            loss = focal_loss(logits.float(), y, gamma, alpha)
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            losses.append(float(loss.detach().cpu()))
        val_p = _predict(model, tok, val_a, val_b, max_len, device, int(ccfg.get("eval_batch_size", 128)), use_amp)
        rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else None}
        if len(np.unique(val_y)) == 2:
            rec["val_average_precision"] = float(average_precision_score(val_y, val_p))
            rec["val_logloss"] = float(log_loss(val_y, np.clip(val_p, 1e-7, 1 - 1e-7)))
        history.append(rec)
        log.info("cross-encoder epoch %d: %s", epoch + 1, rec)
        ap = rec.get("val_average_precision", -rec["train_loss"] if rec["train_loss"] is not None else 0)
        if ap > best_ap:
            best_ap = ap
            save_crossencoder(model, tok, out_dir, max_len, ccfg)
    report = {"history": history, "best_val_average_precision": best_ap, "train_pairs": int(n),
              "train_positives": int(train_y.sum()), "mixda_ops": ops if use_mix else [], "focal_gamma": gamma,
              "focal_alpha": alpha}
    with open(os.path.join(out_dir, "crossencoder_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def save_crossencoder(model, tok, out_dir: str, max_len: int, ccfg: dict):
    ensure_dir(out_dir)
    model.backbone.save_pretrained(os.path.join(out_dir, "backbone"))
    tok.save_pretrained(os.path.join(out_dir, "backbone"))
    torch.save(model.head.state_dict(), os.path.join(out_dir, "head.pt"))
    with open(os.path.join(out_dir, "crossencoder_config.json"), "w") as fh:
        json.dump({"max_length": max_len, "text_source": ccfg.get("text_source", "raw"),
                   "domain_tags": bool(ccfg.get("domain_tags", True))}, fh)


def _predict(model, tok, a, b, max_len, device, bs, use_amp=False) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(a), bs):
            enc = _tokenize(tok, a[s:s + bs], b[s:s + bs], max_len, device)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=use_amp):
                logits = model(enc)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


class CrossEncoderPredictor:
    def __init__(self, model_dir: str, cfg: dict):
        require_torch()
        ccfg = cfg.get("crossencoder", {})
        with open(os.path.join(model_dir, "crossencoder_config.json")) as fh:
            self.meta = json.load(fh)
        self.device = resolve_device(ccfg.get("device", "auto"))
        bdir = os.path.join(model_dir, "backbone")
        self.tok = AutoTokenizer.from_pretrained(bdir, local_files_only=True)
        self.model = CrossEncoder(bdir, local_files_only=True)
        self.model.head.load_state_dict(torch.load(os.path.join(model_dir, "head.pt"), map_location="cpu"))
        self.model.to(self.device)
        self.bs = int(ccfg.get("eval_batch_size", 128))
        self.use_amp = bool(ccfg.get("fp16", True)) and self.device.type == "cuda"

    def predict(self, a: list[str], b: list[str]) -> np.ndarray:
        return _predict(self.model, self.tok, a, b, int(self.meta["max_length"]), self.device, self.bs, self.use_amp)
