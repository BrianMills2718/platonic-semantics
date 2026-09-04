#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, gc
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# transformers renamed the dtype kwarg at 5.0. Resolve it once, loudly, rather
# than letting a notebook monkey-patch the call site at run time.
_TF_MAJOR = int(transformers.__version__.split(".")[0])
DTYPE_KWARG = "dtype" if _TF_MAJOR >= 5 else "torch_dtype"

def read_concepts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def masked_mean(h, mask):
    mask = mask.to(h.dtype).unsqueeze(-1)
    return (h * mask).sum(1) / mask.sum(1).clamp_min(1)

# bloom-560m produces NaN activations in float16 on CUDA: 80% of a run's hidden
# states came back NaN on 2026-09-03, while float32 on the same GPU and inputs
# was clean with max|h| = 1380. Extraction here is 212 short prompts against
# ~0.5B models, so float32 costs seconds and buys a representation you can
# trust. Override with --dtype only for a model you have checked.
DEFAULT_DTYPE = "float32"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_model(name, device, dtype=DEFAULT_DTYPE):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    kwargs = {"low_cpu_mem_usage": True}
    if device.startswith("cuda"):
        kwargs[DTYPE_KWARG] = DTYPES[dtype]
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval().to(device)
    return tok, model

@torch.inference_mode()
def encode_all(tok, model, texts, batch_size, max_length, device):
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt", return_special_tokens_mask=True)
        special = enc.pop("special_tokens_mask", None)
        enc = {k:v.to(device) for k,v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        # hidden_states includes embedding output + transformer block outputs.
        mask = enc["attention_mask"].clone()
        if special is not None:
            content = mask * (1-special.to(mask.device))
            # If a tokenizer marks everything special for some edge case, fall back.
            empty = content.sum(1) == 0
            if empty.any():
                content[empty] = mask[empty]
            mask = content
        layer_reps = [masked_mean(h, mask).float().cpu() for h in out.hidden_states]
        # [batch, layers, dim]
        rep = torch.stack(layer_reps, dim=1)
        chunks.append(rep)
        del out, enc, layer_reps, rep
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    reps = torch.cat(chunks, dim=0).numpy()
    bad = int(np.count_nonzero(~np.isfinite(reps)))
    if bad:
        raise RuntimeError(
            f"{bad} of {reps.size} extracted values are NaN or Inf. This is a dead "
            "representation, not a weak result -- saving it would feed silent zeros "
            "into every downstream correlation. Re-run with --dtype float32 if you "
            "used a reduced precision."
        )
    return reps

@torch.inference_mode()
def encode_term_in_context(tok, model, template, terms, batch_size, max_length, device):
    """Mean-pool only the term's own tokens inside a rendered prompt.

    The bare-prompt path pools every content token, which is correct when the
    prompt *is* the term. Once a template is added, pooling everything would mix
    the template's words into the representation and make "The concept is X"
    partly a representation of "the concept is". Offsets let the term be
    isolated, so what varies across templates is the context the term sits in,
    not what is being measured.
    """
    texts, spans = [], []
    for term in terms:
        rendered = template.format(term=term)
        start = rendered.index(term)
        texts.append(rendered)
        spans.append((start, start + len(term)))

    chunks = []
    for begin in range(0, len(texts), batch_size):
        batch = texts[begin:begin + batch_size]
        batch_spans = spans[begin:begin + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)

        mask = torch.zeros_like(enc["attention_mask"])
        for i, (lo, hi) in enumerate(batch_spans):
            for j, (a, b) in enumerate(offsets[i].tolist()):
                if a == b:          # special token
                    continue
                if a < hi and b > lo:   # token overlaps the term's characters
                    mask[i, j] = 1
        empty = mask.sum(1) == 0
        if empty.any():
            raise RuntimeError(
                f"{int(empty.sum())} prompt(s) produced no token overlapping the term span; "
                "the tokenizer offsets cannot locate the term, so its representation "
                "would silently become the whole sentence."
            )
        layer_reps = [masked_mean(h, mask).float().cpu() for h in out.hidden_states]
        chunks.append(torch.stack(layer_reps, dim=1))
        del out, enc, layer_reps
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    reps = torch.cat(chunks, dim=0).numpy()
    bad = int(np.count_nonzero(~np.isfinite(reps)))
    if bad:
        raise RuntimeError(f"{bad} non-finite values extracted for template {template!r}")
    return reps


def main():
    ap=argparse.ArgumentParser(description="Extract layer-wise concept representations.")
    ap.add_argument("--config",default="experiment_config.json")
    ap.add_argument("--concepts",default="benchmark/concepts.csv")
    ap.add_argument("--outdir",default="outputs/representations")
    ap.add_argument("--models",nargs="*",default=None,help="Config model keys; default all.")
    ap.add_argument("--languages",nargs="*",default=None)
    ap.add_argument("--prompt-mode",default=None,choices=["bare","neutral"])
    ap.add_argument("--device",default=None,help="e.g. cuda, cuda:0, mps, cpu")
    ap.add_argument("--dtype",default=DEFAULT_DTYPE,choices=sorted(DTYPES),
                    help="CUDA compute dtype; float32 is the safe default (see load_model)")
    ap.add_argument("--average-modes",default=None,
                    help="comma-separated prompt modes to average over, or 'config' to use "
                         "averaged_modes from experiment_config.json. Saves as __averaged.")
    args=ap.parse_args()

    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows=read_concepts(args.concepts)
    models=args.models or list(cfg["models"])
    langs=args.languages or cfg["languages"]
    prompt_mode=args.prompt_mode or cfg["default_prompt_mode"]
    templates=cfg["prompt_modes"][prompt_mode]
    outdir=Path(args.outdir);outdir.mkdir(parents=True,exist_ok=True)

    if args.device:
        device=args.device
    elif torch.cuda.is_available():
        device="cuda"
    elif getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
        device="mps"
    else:
        device="cpu"

    print(f"device={device}; prompt_mode={prompt_mode}; concepts={len(rows)}")
    ids=np.array([r["concept_id"] for r in rows])

    average_modes = None
    if args.average_modes:
        average_modes = (cfg["averaged_modes"] if args.average_modes == "config"
                         else [m.strip() for m in args.average_modes.split(",")])
        unknown = [m for m in average_modes if m not in cfg["prompt_modes"]]
        if unknown:
            raise SystemExit(f"unknown prompt mode(s): {unknown}")
        prompt_mode = "averaged"
        print(f"averaging over {len(average_modes)} templates: {', '.join(average_modes)}")

    for key in models:
        name=cfg["models"][key]
        print(f"\n=== {key}: {name} ===")
        tok,model=load_model(name,device,args.dtype)
        for lang in langs:
            if average_modes:
                # Each template contributes an L2-normalised representation, so a
                # long frame cannot dominate the mean through its norm alone.
                acc=None
                for mode in average_modes:
                    tmpl=cfg["prompt_modes"][mode][lang]
                    r=encode_term_in_context(
                        tok,model,tmpl,[row[lang] for row in rows],
                        cfg["batch_size"],cfg["max_length"],device)
                    r=r/np.maximum(np.linalg.norm(r,axis=-1,keepdims=True),1e-12)
                    acc=r if acc is None else acc+r
                reps=acc/len(average_modes)
            else:
                texts=[templates[lang].format(term=r[lang]) for r in rows]
                reps=encode_all(tok,model,texts,cfg["batch_size"],cfg["max_length"],device)
            path=outdir/f"{key}__{lang}__{prompt_mode}.npz"
            np.savez_compressed(path, concept_ids=ids, reps=reps.astype(np.float16),
                                model_key=key, model_name=name, language=lang,
                                prompt_mode=prompt_mode)
            print(f"saved {path}  shape={reps.shape}  dtype={args.dtype}")
        del model,tok
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

if __name__=="__main__":
    main()
