"""Semantic blocking: supervised-contrastive bi-encoder + ANN retrieval
(PDF: "Supervised Contrastive Blocking for Semantic Alignment", SC-Block).

* Encoder: a pre-trained transformer (default MiniLM, PDF: "a RoBERTa variant
  or a distilled MiniLM") with mean pooling and L2 normalisation, shared by
  Source-1 and vendor records (siamese towers; records never interact, so all
  vendor embeddings can be pre-computed and indexed).
* Loss: batch-wise Supervised Contrastive loss (Khosla et al. 2020)

      L = sum_i  -1/|P(i)| sum_{p in P(i)} log( exp(z_i.z_p/t) / sum_{a in A(i)} exp(z_i.z_a/t) )

  Labels come from the ground truth: a Source-1 entity and all of its
  labelled Source-2/3 records share one label; singletons and unlabelled
  vendor records get unique labels and act as in-batch negatives only.
* Hard batching: entities sharing a blocking key (name prefix) are placed in
  the same batch so in-batch negatives are look-alikes; extra vendor records
  with the same keys are added as hard negatives.
* Retrieval: FAISS inner-product index (exact `flat` or `hnsw`) per source,
  top-k neighbours per Source-1 entity; numpy fallback if faiss is missing.

AdaFlood (named in the PDF) is NOT implemented: the PDF gives no formula and
its per-sample flood levels require auxiliary models whose training protocol
for SupCon is not specified. `flood_level` implements only the original,
constant Flooding regulariser (Ishida et al. 2020, |L - b| + b); it is
disabled by default and is not a substitute for AdaFlood.

Pre-trained weights are loaded from a LOCAL directory (`local_files_only`);
nothing is downloaded during training or matching.
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import pandas as pd

from .utils import ensure_dir, get_logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def require_torch():
    if torch is None:
        raise ImportError("Full mode needs torch and transformers: pip install -r requirements-full.txt")


def resolve_device(name: str = "auto"):
    require_torch()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def record_texts(df: pd.DataFrame, text_source: str = "normalized") -> list[str]:
    if text_source == "raw":
        return [f"{n} | {a}".strip(" |") for n, a in zip(df["name"], df["address"])]
    return [f"{n} | {a}".strip(" |") for n, a in zip(df["name_clean"], df["addr_clean"])]


def linear_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


if torch is not None:
    class BiEncoder(nn.Module):
        def __init__(self, path: str, pooling: str = "mean", local_files_only: bool = True):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(path, local_files_only=local_files_only)
            self.pooling = pooling

        def forward(self, input_ids, attention_mask, **kw):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kw).last_hidden_state
            if self.pooling == "cls":
                emb = out[:, 0]
            else:
                m = attention_mask.unsqueeze(-1).to(out.dtype)
                emb = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
            return F.normalize(emb, dim=-1)

    def supcon_loss(z: "torch.Tensor", labels: "torch.Tensor", temperature: float = 0.07) -> "torch.Tensor":
        """Supervised contrastive loss; anchors without positives are ignored."""
        n = z.shape[0]
        sim = z @ z.T / temperature
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, -1e9)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        n_pos = pos.sum(1)
        valid = n_pos > 0
        if not valid.any():
            return (z * 0).sum()
        mean_log_prob_pos = (log_prob * pos).sum(1)[valid] / n_pos[valid]
        return -mean_log_prob_pos.mean()


def _batches(groups: list[list[int]], keys: list[str], batch_size: int, rng: random.Random,
             hard: bool) -> list[list[int]]:
    """Pack whole groups (S1 record + its positives) into batches."""
    order = list(range(len(groups)))
    rng.shuffle(order)
    if hard:
        order.sort(key=lambda g: keys[g])  # stable: random within equal keys
    batches, cur, cur_n = [], [], 0
    for g in order:
        if cur and cur_n + len(groups[g]) > batch_size:
            batches.append(cur)
            cur, cur_n = [], 0
        cur.append(g)
        cur_n += len(groups[g])
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def train_biencoder(s1: pd.DataFrame, v: pd.DataFrame, true_pairs: pd.DataFrame, split: np.ndarray,
                    cfg: dict, out_dir: str, seed: int = 42) -> dict:
    """Train on `fit` entities, early-stop on recall@k of `fit_val` entities."""
    require_torch()
    log = get_logger()
    bcfg = cfg.get("biencoder", {})
    device = resolve_device(bcfg.get("device", "auto"))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(bcfg["pretrained"], local_files_only=bcfg.get("local_files_only", True))
    model = BiEncoder(bcfg["pretrained"], bcfg.get("pooling", "mean"),
                      bcfg.get("local_files_only", True)).to(device)
    text_src = bcfg.get("text_source", "normalized")
    t1, tv = record_texts(s1, text_src), record_texts(v, text_src)
    max_pos = int(bcfg.get("max_positives_per_entity", 4))

    pos_by_s1 = true_pairs.groupby("s1_idx")["v_idx"].apply(list).to_dict()
    fit_ents = np.where(split == "fit")[0]
    val_ents = np.where(split == "fit_val")[0]
    # items: ("s", idx) for S1 records, ("v", idx) for vendor records
    items: list[tuple[str, int]] = []
    groups, keys = [], []
    for e in fit_ents:
        g = [len(items)]
        items.append(("s", int(e)))
        for vi in pos_by_s1.get(e, [])[:max_pos]:
            g.append(len(items))
            items.append(("v", int(vi)))
        groups.append(g)
        keys.append(s1["name_nospace"].iat[e][:3] + "|" + s1["addr_postcode"].iat[e])
    labels_all = np.empty(len(items), dtype=np.int64)
    for gi, g in enumerate(groups):
        labels_all[g] = gi
    positive_v = set(true_pairs["v_idx"].tolist())
    neg_pool = [i for i in range(len(v)) if i not in positive_v]
    neg_by_key: dict[str, list[int]] = {}
    for i in neg_pool:
        neg_by_key.setdefault(v["name_nospace"].iat[i][:3], []).append(i)
    extra_n = int(bcfg.get("extra_negatives_per_batch", 32))

    bs = int(bcfg.get("batch_size", 128))
    epochs = int(bcfg.get("epochs", 5))
    steps_per_epoch = max(1, len(_batches(groups, keys, bs, random.Random(0), False)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(bcfg.get("lr", 3e-5)),
                            weight_decay=float(bcfg.get("weight_decay", 0.01)))
    total = steps_per_epoch * epochs
    sched = linear_warmup_schedule(opt, int(float(bcfg.get("warmup_ratio", 0.1)) * total), total)
    temp = float(bcfg.get("temperature", 0.07))
    flood = bcfg.get("flood_level")
    max_len = int(bcfg.get("max_length", 64))
    k_eval = int(cfg.get("blocking", {}).get("methods", {}).get("semantic", {}).get("top_k", 30))
    best_recall, history = -1.0, []
    ensure_dir(out_dir)

    def encode_batch(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        return model(enc["input_ids"], enc["attention_mask"])

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_groups in _batches(groups, keys, bs, rng, bool(bcfg.get("hard_batching", True))):
            idx = [i for g in batch_groups for i in groups[g]]
            texts = [t1[items[i][1]] if items[i][0] == "s" else tv[items[i][1]] for i in idx]
            labels = list(labels_all[idx])
            if extra_n and neg_pool:
                bkeys = {s1["name_nospace"].iat[items[groups[g][0]][1]][:3] for g in batch_groups}
                cand = [j for kk in bkeys for j in neg_by_key.get(kk, [])]
                picks = rng.sample(cand, min(len(cand), extra_n // 2))
                picks += rng.sample(neg_pool, min(len(neg_pool), extra_n - len(picks)))
                texts += [tv[j] for j in picks]
                labels += list(range(-1, -1 - len(picks), -1))  # unique negative labels
            z = encode_batch(texts)
            loss = supcon_loss(z, torch.tensor(labels, device=device), temp)
            if flood is not None:
                loss = (loss - float(flood)).abs() + float(flood)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach().cpu()))
        recall = _recall_at_k(model, tok, t1, tv, val_ents, pos_by_s1, v, k_eval, max_len, device,
                              int(bcfg.get("encode_batch_size", 256)), seed)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else None,
                        "val_recall_at_k": recall})
        log.info("bi-encoder epoch %d: loss=%.4f  fit_val recall@%d=%.4f", epoch + 1,
                 history[-1]["loss"] or float("nan"), k_eval, recall)
        if recall > best_recall:
            best_recall = recall
            model.backbone.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            with open(os.path.join(out_dir, "biencoder_config.json"), "w") as fh:
                json.dump({"pooling": model.pooling, "max_length": max_len, "text_source": text_src}, fh)
    report = {"history": history, "best_fit_val_recall_at_k": best_recall, "k": k_eval,
              "train_entities": int(len(fit_ents)), "adaflood": "not implemented (see module docstring)",
              "flood_level": flood}
    with open(os.path.join(out_dir, "biencoder_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def _recall_at_k(model, tok, t1, tv, val_ents, pos_by_s1, v, k, max_len, device, bs, seed) -> float:
    """Recall@k of fit_val positives against a vendor pool (all positives of
    the val entities + a random sample of other vendor records)."""
    ents = [e for e in val_ents if pos_by_s1.get(e)]
    if not ents:
        return 0.0
    rng = np.random.default_rng(seed)
    pool = set(vi for e in ents for vi in pos_by_s1[e])
    others = rng.choice(len(v), size=min(len(v), 20000), replace=False)
    pool = np.array(sorted(pool | set(others.tolist())))
    ze = _encode(model, tok, [t1[e] for e in ents], max_len, device, bs)
    zv = _encode(model, tok, [tv[i] for i in pool], max_len, device, bs)
    src = v["source"].to_numpy()[pool]
    hits = total = 0
    for s in (2, 3):
        m = src == s
        if not m.any():
            continue
        sims = ze @ zv[m].T
        kk = min(k, int(m.sum()))
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        pool_s = pool[m]
        for row, e in enumerate(ents):
            truth = {vi for vi in pos_by_s1[e] if v["source"].iat[vi] == s}
            total += len(truth)
            hits += len(truth & set(pool_s[top[row]].tolist()))
    return hits / total if total else 0.0


def _encode(model, tok, texts, max_len, device, bs) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            out.append(model(enc["input_ids"], enc["attention_mask"]).float().cpu().numpy())
    if not out:
        return np.zeros((0, 1), dtype=np.float32)
    return np.vstack(out).astype(np.float32)


class SemanticRetriever:
    """Encode records with the trained bi-encoder and search FAISS indices."""

    def __init__(self, model_dir: str, cfg: dict):
        require_torch()
        bcfg = cfg.get("biencoder", {})
        with open(os.path.join(model_dir, "biencoder_config.json")) as fh:
            meta = json.load(fh)
        self.device = resolve_device(bcfg.get("device", "auto"))
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = BiEncoder(model_dir, meta.get("pooling", "mean"), local_files_only=True).to(self.device)
        self.max_len = int(meta.get("max_length", 64))
        self.text_source = meta.get("text_source", "normalized")
        self.bs = int(bcfg.get("encode_batch_size", 256))
        self.index_type = bcfg.get("faiss_index", "flat")
        self._key = None
        self.indices: dict[int, object] = {}

    def prepare(self, s1: pd.DataFrame, v: pd.DataFrame):
        key = (len(s1), len(v), s1["id"].iat[0] if len(s1) else "", v["id"].iat[0] if len(v) else "")
        if key == self._key:
            return
        log = get_logger()
        self.z1 = _encode(self.model, self.tok, record_texts(s1, self.text_source), self.max_len, self.device, self.bs)
        self.zv = _encode(self.model, self.tok, record_texts(v, self.text_source), self.max_len, self.device, self.bs)
        self.src = v["source"].to_numpy()
        self.indices = {}
        for s in (2, 3):
            rows = np.where(self.src == s)[0]
            self.indices[s] = (self._build_index(self.zv[rows]), rows)
        self._key = key
        log.info("Semantic retriever: encoded %d S1 / %d vendor records (index=%s, faiss=%s)",
                 len(s1), len(v), self.index_type, faiss is not None)

    def _build_index(self, emb: np.ndarray):
        if faiss is None or len(emb) == 0:
            return emb
        d = emb.shape[1]
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 128
        else:
            index = faiss.IndexFlatIP(d)
        index.add(np.ascontiguousarray(emb, dtype=np.float32))
        return index

    def search(self, s1: pd.DataFrame, v: pd.DataFrame, source: int, k: int) -> pd.DataFrame:
        self.prepare(s1, v)
        index, rows = self.indices[source]
        if len(rows) == 0:
            return pd.DataFrame({"s1_idx": [], "v_idx": [], "sim": []})
        kk = min(k, len(rows))
        if faiss is not None and not isinstance(index, np.ndarray):
            sims, idx = index.search(np.ascontiguousarray(self.z1, dtype=np.float32), kk)
        else:
            sims_full = self.z1 @ index.T
            idx = np.argpartition(-sims_full, kk - 1, axis=1)[:, :kk]
            sims = np.take_along_axis(sims_full, idx, axis=1)
        valid = idx >= 0
        s1_idx = np.repeat(np.arange(len(self.z1)), kk).reshape(-1, kk)[valid]
        return pd.DataFrame({"s1_idx": s1_idx.astype(np.int64), "v_idx": rows[idx[valid]].astype(np.int64),
                             "sim": sims[valid].astype(np.float32)})

    def save_indices(self, out_dir: str):
        ensure_dir(out_dir)
        for s, (index, rows) in self.indices.items():
            np.save(os.path.join(out_dir, f"semantic_rows_s{s}.npy"), rows)
            if faiss is not None and not isinstance(index, np.ndarray):
                faiss.write_index(index, os.path.join(out_dir, f"semantic_s{s}.faiss"))
            else:
                np.save(os.path.join(out_dir, f"semantic_emb_s{s}.npy"), index)
