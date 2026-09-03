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

def load_model(name, device):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    kwargs = {"low_cpu_mem_usage": True}
    if device.startswith("cuda"):
        kwargs[DTYPE_KWARG] = torch.float16
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
    return torch.cat(chunks, dim=0).numpy()

def main():
    ap=argparse.ArgumentParser(description="Extract layer-wise concept representations.")
    ap.add_argument("--config",default="experiment_config.json")
    ap.add_argument("--concepts",default="benchmark/concepts.csv")
    ap.add_argument("--outdir",default="outputs/representations")
    ap.add_argument("--models",nargs="*",default=None,help="Config model keys; default all.")
    ap.add_argument("--languages",nargs="*",default=None)
    ap.add_argument("--prompt-mode",default=None,choices=["bare","neutral"])
    ap.add_argument("--device",default=None,help="e.g. cuda, cuda:0, mps, cpu")
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

    for key in models:
        name=cfg["models"][key]
        print(f"\n=== {key}: {name} ===")
        tok,model=load_model(name,device)
        for lang in langs:
            texts=[templates[lang].format(term=r[lang]) for r in rows]
            reps=encode_all(tok,model,texts,cfg["batch_size"],cfg["max_length"],device)
            path=outdir/f"{key}__{lang}__{prompt_mode}.npz"
            np.savez_compressed(path, concept_ids=ids, reps=reps.astype(np.float16),
                                model_key=key, model_name=name, language=lang,
                                prompt_mode=prompt_mode)
            print(f"saved {path}  shape={reps.shape}")
        del model,tok
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

if __name__=="__main__":
    main()
