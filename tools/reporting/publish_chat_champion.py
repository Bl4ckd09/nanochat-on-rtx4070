#!/usr/bin/env python3
"""Convert the mixv2 step-300 .pt checkpoint to safetensors and push it to the Hub.

Used by .github/workflows/publish-chat-champion.yml (sparse-checkout of this
directory only — do not clone artifacts/). Also runnable locally:

    python tools/reporting/publish_chat_champion.py --ckpt path/to/model_000300.pt --publish
    python tools/reporting/publish_chat_champion.py --ckpt-dir ~/ckpt --publish

Does not touch README.md on the Hub. The card already calls this checkpoint
provisional.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = "Marcolini/nanochat-d24-chat-champion"
EXPECTED_PARAMS = 910_691_760
FALLBACK_CONFIG = {
    "model_type": "nanochat_gpt",
    "architectures": ["GPT"],
    "transformers_version": None,
    "sequence_len": 2048,
    "vocab_size": 32768,
    "n_layer": 24,
    "n_head": 9,
    "n_kv_head": 9,
    "n_embd": 1152,
    "window_pattern": "L",
    "torch_dtype": "bfloat16",
    "num_parameters": EXPECTED_PARAMS,
    "_nanochat": {
        "step": 300,
        "model_tag": "d24_r32_adamw_partial_fr20_mixv2_s768_s300",
        "provisional": True,
        "replication": "failed (direct replica + 3-seed sweep 0/3)",
    },
}


def resolve_ckpt(ckpt: Path | None, ckpt_dir: Path | None) -> Path:
    if ckpt is not None:
        if not ckpt.is_file():
            sys.exit(f"--ckpt is not a file: {ckpt}")
        return ckpt
    roots = []
    if ckpt_dir is not None:
        roots.append(ckpt_dir)
    env_dir = os.environ.get("CKPT_DIR")
    if env_dir:
        roots.append(Path(env_dir))
    roots.append(Path.cwd())
    pts: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                pts.append(p)
    if not pts:
        sys.exit(f"no .pt under {roots}")
    preferred = [p for p in pts if p.name == "model_000300.pt"]
    if not preferred:
        preferred = [p for p in pts if "000300" in p.name]
    chosen = preferred[0] if preferred else sorted(pts)[-1]
    if len(pts) > 1:
        print(f"found {len(pts)} .pt files; using {chosen}")
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="search this tree for model_000300.pt")
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("./chat_champion_export"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--delete-source", action="store_true",
                    help="delete the .pt after conversion to free disk")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        sys.exit("HF_TOKEN is not set")

    ckpt = resolve_ckpt(args.ckpt, args.ckpt_dir)

    import torch
    from huggingface_hub import HfApi
    from safetensors.torch import save_file

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"loading {ckpt}", flush=True)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob
    del blob
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in state.items()}
    state = {k: v.detach().clone().contiguous() for k, v in state.items()}
    n_params = sum(v.numel() for v in state.values())
    print(f"{len(state)} tensors, {n_params:,} params", flush=True)
    if n_params != EXPECTED_PARAMS:
        sys.exit(f"param count {n_params} != {EXPECTED_PARAMS}")

    save_file(state, str(args.out / "model.safetensors"), metadata={"format": "pt"})
    dtypes = sorted({str(v.dtype).replace("torch.", "") for v in state.values()})
    del state
    print(f"wrote {args.out / 'model.safetensors'} dtypes={dtypes}", flush=True)

    meta_src = args.meta
    if meta_src is None:
        cands = sorted(ckpt.parent.glob("meta_*.json"))
        meta_src = cands[-1] if cands else None
    cfg = dict(FALLBACK_CONFIG)
    if meta_src and meta_src.exists():
        meta = json.loads(meta_src.read_text())
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        extracted = meta.get("model_config") or meta.get("config") or {}
        if extracted:
            cfg.update(extracted)
        print(f"copied metadata from {meta_src.name}", flush=True)
    else:
        print("no meta_*.json; writing architecture fallback config.json", flush=True)
    cfg["num_parameters"] = EXPECTED_PARAMS
    cfg.setdefault("_nanochat", {})
    if isinstance(cfg["_nanochat"], dict):
        cfg["_nanochat"].setdefault("step", 300)
        cfg["_nanochat"].setdefault("provisional", True)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if args.delete_source:
        ckpt.unlink(missing_ok=True)
        print(f"deleted {ckpt}", flush=True)

    api = HfApi()
    print(f"uploading to {args.repo} ...", flush=True)
    api.upload_folder(
        folder_path=str(args.out),
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add chat champion weights (step 300)",
        allow_patterns=["model.safetensors", "config.json", "meta.json"],
    )
    if args.publish:
        api.update_repo_settings(repo_id=args.repo, repo_type="model", private=False)
        print("repo is now public", flush=True)
    else:
        print("repo left private — re-run with --publish", flush=True)
    print(f"done: https://huggingface.co/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
