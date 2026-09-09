"""
Modal harness for nanochat-on-rtx4070, cloud side of the dual-operating-point fork.

The repo's code dirs (nanochat/, scripts/, tasks/) are mounted from the local
sparse clone, so local edits run in the cloud without pushing to GitHub. Heavy
state (checkpoints, tokenizer, eval bundle) lives on the `nanochat-cache` volume
under /vol/nanochat, which is what NANOCHAT_BASE_DIR points at.

Usage:
    modal run tools/modal/modal_app.py --action setup    # pull d24@820230 from GitHub into the volume (CPU, one-time)
    modal run tools/modal/modal_app.py --action smoke    # CORE on a few examples per task (~minutes)
    modal run tools/modal/modal_app.py --action core     # full CORE, skip5120 policy — compare vs report_v3.md (0.1514)
    modal run tools/modal/modal_app.py --action pack --model-tag <sft_tag> --step <n>   # GSM8K pass@1/8 + MMLU|SpellingBee

The tokenizer must exist at /vol/nanochat/tokenizer. If `setup` reports it was
not inside the checkpoint export, upload it from the training box:
    modal volume put nanochat-cache <local>/tokenizer /nanochat/tokenizer
"""
import os
import subprocess
from pathlib import Path

import modal

# GPU for eval functions. A100-40GB needs a payment method on the workspace;
# hackathon-credit workspaces typically allow smaller GPUs. Ampere (A10G/A100)
# and Ada (L4) all take the SDPA fallback path, matching the local 4070 runs.
GPU = os.environ.get("NANOCHAT_MODAL_GPU", "A10G")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_URL = "https://github.com/Bl4ckd09/nanochat-on-rtx4070.git"

MODEL_TAG = "d24_asp48_track"
STEP = 820230
ARTIFACT_DIR = f"artifacts/{MODEL_TAG}_s{STEP}_2026-03-19_122717"
META_DIR = f"archive/nanochat-learn-parent-2026-03-28/exports/{MODEL_TAG}_s{STEP}_2026-03-19_120117"

VOL_NAME = "nanochat-cache"
BASE_DIR = "/vol/nanochat"  # NANOCHAT_BASE_DIR inside containers

app = modal.App("nanochat-eval")
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

fetch_image = modal.Image.debian_slim(python_version="3.11").apt_install("git")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "datasets>=4.0.0",
        "kernels>=0.11.7",
        "matplotlib>=3.10.8",
        "psutil>=7.1.0",
        "python-dotenv>=1.2.1",
        "pyyaml>=6.0",
        "regex>=2025.9.1",
        "rustbpe>=0.1.0",
        "scipy>=1.15.3",
        "tabulate>=0.9.0",
        "tiktoken>=0.11.0",
        "tokenizers>=0.22.0",
        "torchao==0.15.0",
        "transformers>=4.57.3",
        "wandb>=0.21.3",
        "zstandard>=0.25.0",
    )
    .env({"NANOCHAT_BASE_DIR": BASE_DIR, "PYTHONPATH": "/repo"})
    .add_local_dir(REPO_ROOT / "nanochat", "/repo/nanochat")
    .add_local_dir(REPO_ROOT / "scripts", "/repo/scripts")
    .add_local_dir(REPO_ROOT / "tasks", "/repo/tasks")
)


def _run(cmd, **kw):
    print(f"+ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


@app.function(image=fetch_image, volumes={"/vol": vol}, timeout=3600, cpu=4, memory=8192)
def setup_checkpoint(force: bool = False) -> dict:
    """Fetch the d24@820230 export from GitHub (blobless sparse clone, cloud-to-cloud),
    verify sha256, extract, and place it on the volume where load_model expects it."""
    import hashlib
    import shutil
    import tarfile

    ckpt_dir = Path(BASE_DIR) / "base_checkpoints" / MODEL_TAG
    model_pt = ckpt_dir / f"model_{STEP:06d}.pt"
    if model_pt.exists() and not force:
        listing = sorted(str(p.relative_to(BASE_DIR)) for p in Path(BASE_DIR).rglob("*") if p.is_file())
        return {"status": "already_present", "volume_files": listing[:50]}

    work = Path("/tmp/fetch")
    shutil.rmtree(work, ignore_errors=True)
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", REPO_URL, work])
    _run(["git", "-C", work, "sparse-checkout", "set", ARTIFACT_DIR, META_DIR])
    _run(["git", "-C", work, "checkout", "master"])

    art = work / ARTIFACT_DIR
    tar_path = art / f"model_{STEP}.tar.gz"
    with open(tar_path, "wb") as out:
        for part in sorted(art.glob(f"model_{STEP}.tar.gz.part-*")):
            out.write(part.read_bytes())

    # verify combined sha256
    expected = (art / "SHA256SUM.combined.txt").read_text().split()[0]
    h = hashlib.sha256()
    with open(tar_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise RuntimeError(f"sha256 mismatch: {h.hexdigest()} != {expected}")
    print(f"sha256 OK: {expected}", flush=True)

    staging = Path("/tmp/extract")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        contents = tf.getnames()
        tf.extractall(staging)

    # place everything under the checkpoint dir; surface tokenizer if bundled
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_present = False
    for p in staging.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(staging)
        if "tokenizer" in str(rel):
            dst = Path(BASE_DIR) / "tokenizer" / rel.name
            tokenizer_present = True
        else:
            dst = ckpt_dir / rel.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), dst)

    meta_src = work / META_DIR / f"meta_{STEP}.json"
    shutil.copy(meta_src, ckpt_dir / f"meta_{STEP:06d}.json")

    vol.commit()
    placed = sorted(str(p.relative_to(BASE_DIR)) for p in Path(BASE_DIR).rglob("*") if p.is_file())
    return {"status": "ok", "tar_contents": contents, "placed": placed, "tokenizer_present": tokenizer_present}


@app.function(image=image, volumes={"/vol": vol}, timeout=2 * 3600, cpu=16, memory=32768)
def tok_train(
    shards: int = 12,
    max_chars: int = 2_000_000_000,
    vocab_size: int = 32768,
    shards_visible: int = 0,
) -> list:
    """Reproduce the tokenizer with stock tok_train.py settings (deterministic:
    fixed pre-shuffled FineWeb-edu shards, sorted iteration order).
    Validated downstream: a mismatched tokenizer collapses CORE LM tasks to ~0.

    shards_visible > 0 replicates the speedrun.sh race: tok_train sees exactly
    that many shards on disk (train split = all but the last), regardless of how
    many live on the volume. speedrun.sh at the fork point downloaded 8 shards
    then trained immediately, so the original likely saw 8 -> trained on 0-6."""
    import os as _os
    import shutil

    _run(["python", "-m", "nanochat.dataset", "-n", str(shards), "-w", "8"], cwd="/repo")

    env = dict(_os.environ)
    if shards_visible > 0:
        staged = Path("/tmp/tokbase/base_data")
        shutil.rmtree(staged.parent, ignore_errors=True)
        staged.mkdir(parents=True)
        for i in range(shards_visible):
            name = f"shard_{i:05d}.parquet"
            _os.symlink(Path(BASE_DIR) / "base_data" / name, staged / name)
        env["NANOCHAT_BASE_DIR"] = str(staged.parent)

    _run(
        ["python", "-m", "scripts.tok_train",
         "--max-chars", str(max_chars), "--vocab-size", str(vocab_size)],
        cwd="/repo",
        env=env,
    )

    tok_dir = Path(BASE_DIR) / "tokenizer"
    if shards_visible > 0:
        # tok_train wrote into the staged base dir; move results to the volume
        tok_dir.mkdir(parents=True, exist_ok=True)
        for f in (Path("/tmp/tokbase") / "tokenizer").iterdir():
            shutil.move(str(f), tok_dir / f.name)
    vol.commit()
    return sorted(f"{p.name} ({p.stat().st_size} bytes)" for p in tok_dir.iterdir())


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=4 * 3600)
def eval_base(
    evals: str = "core",
    model_tag: str = MODEL_TAG,
    step: int = STEP,
    max_per_task: int = -1,
    core_max_seq_len: int = 5120,
    overflow_policy: str = "skip",
    device_batch_size: int = 32,
):
    """CORE (and optionally bpb/sample) on a base checkpoint. Defaults replicate the
    report_v3.md skip5120 reference run for d24_asp48_track@820230 (CORE 0.1514).
    A100 (sm80) uses the same SDPA fallback path as the local RTX 4070 runs."""
    _run(
        [
            "python", "-m", "scripts.base_eval",
            "--eval", evals,
            "--model-tag", model_tag,
            "--step", str(step),
            "--max-per-task", str(max_per_task),
            "--core-max-seq-len", str(core_max_seq_len),
            "--core-overflow-policy", overflow_policy,
            "--device-batch-size", str(device_batch_size),
        ],
        cwd="/repo",
    )
    vol.commit()  # persist eval_bundle download for next time


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=4 * 3600)
def eval_pack(model_tag: str, step: int, max_problems: int = 300, wandb_run: str = "dummy"):
    """The fork's standard chat eval pack (run_chat_eval_pack.sh): GSM8K pass@1,
    GSM8K pass@8 (t=0.7), MMLU|SpellingBee. Needs an SFT checkpoint on the volume
    under chatsft_checkpoints/<model_tag>/."""
    common = ["python", "-m", "scripts.chat_eval", "-i", "sft", "-g", model_tag, "-s", str(step), "--run", wandb_run]
    _run(common + ["-a", "GSM8K", "-t", "0", "-n", "1", "-m", "1024", "-k", "50", "-x", str(max_problems)], cwd="/repo")
    _run(common + ["-a", "GSM8K", "-t", "0.7", "-n", "8", "-m", "1024", "-k", "50", "-x", str(max_problems)], cwd="/repo")
    _run(common + ["-a", "MMLU|SpellingBee", "-x", "200"], cwd="/repo")
    vol.commit()


@app.local_entrypoint()
def main(
    action: str = "setup",
    model_tag: str = MODEL_TAG,
    step: int = STEP,
    shards: int = 12,
    max_chars: int = 2_000_000_000,
    shards_visible: int = 0,
):
    if action == "setup":
        result = setup_checkpoint.remote()
        print(f"status: {result['status']}")
        if "tar_contents" in result:
            print("tar contents:", *result["tar_contents"], sep="\n  ")
            print("tokenizer bundled in export:", result["tokenizer_present"])
        print("volume files:", *result.get("placed", result.get("volume_files", [])), sep="\n  ")
    elif action == "tok-train":
        files = tok_train.remote(shards=shards, max_chars=max_chars, shards_visible=shards_visible)
        print("tokenizer files on volume:", *files, sep="\n  ")
    elif action == "smoke":
        eval_base.remote(max_per_task=20)
    elif action == "bpb":
        # train bpb fingerprint: reference log says 0.801367 (first ~21M train tokens = shard 0)
        eval_base.remote(evals="bpb", device_batch_size=8)
    elif action == "core":
        eval_base.remote()
    elif action == "pack":
        eval_pack.remote(model_tag=model_tag, step=step)
    else:
        raise SystemExit(f"unknown action: {action}")
