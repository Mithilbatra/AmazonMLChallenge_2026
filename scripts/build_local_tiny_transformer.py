#!/usr/bin/env python
"""Build TINY, randomly initialised transformer checkpoints on local disk so
the full-mode code path (bi-encoder, DeBERTa-architecture cross-encoder) can
be smoke-tested fully offline - e.g. in CI or when the HuggingFace hub is
unreachable. The WordPiece tokenizer is trained only on the provided training
TSVs, so nothing external is used.

These models are NOT pre-trained and are no substitute for the real
DeBERTa-v3 / MiniLM weights (see scripts/download_pretrained.py).

    python scripts/build_local_tiny_transformer.py --config configs/synthetic_full.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get, load_config  # noqa: E402
from src.data_io import read_tsv  # noqa: E402


def corpus(cfg):
    for s in (1, 2, 3):
        path = get(cfg, f"data.train.source{s}")
        df = read_tsv(path, cfg)
        sch = get(cfg, f"schema.source{s}")
        addr_cols = sch["address"] if isinstance(sch["address"], list) else [sch["address"]]
        for _, row in df.iterrows():
            yield " ".join([str(row[sch["name"]])] + [str(row[c]) for c in addr_cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synthetic_full.yaml")
    ap.add_argument("--out", default="models/pretrained")
    ap.add_argument("--vocab-size", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config(args.config)

    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import (BertConfig, BertModel, DebertaV2Config, DebertaV2Model,
                              PreTrainedTokenizerFast)

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=args.vocab_size, special_tokens=special)
    tk.train_from_iterator(list(corpus(cfg)), trainer)
    from tokenizers import processors
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tk.token_to_id("[CLS]")), ("[SEP]", tk.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                   model_max_length=256)
    vocab = len(fast)
    h = args.hidden
    builds = {
        "tiny-bert": BertModel(BertConfig(vocab_size=vocab, hidden_size=h, num_hidden_layers=2,
                                          num_attention_heads=2, intermediate_size=2 * h,
                                          max_position_embeddings=256)),
        "tiny-deberta": DebertaV2Model(DebertaV2Config(vocab_size=vocab, hidden_size=h, num_hidden_layers=2,
                                                       num_attention_heads=2, intermediate_size=2 * h,
                                                       max_position_embeddings=256, relative_attention=True,
                                                       position_buckets=64, pos_att_type=["p2c", "c2p"],
                                                       type_vocab_size=0)),
    }
    for name, model in builds.items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        model.save_pretrained(d)
        fast.save_pretrained(d)
        print(f"saved {d} ({sum(p.numel() for p in model.parameters()):,} params, vocab {vocab})")


if __name__ == "__main__":
    main()
