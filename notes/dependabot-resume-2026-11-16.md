# Dependabot resume note, 2026-11-16

Related: [issue #3](https://github.com/Bl4ckd09/nanochat-on-rtx4070/issues/3). Leave both PRs open until Sun asks for a decision. Do not merge either PR yet.

## Open PRs

- [PR #2](https://github.com/Bl4ckd09/nanochat-on-rtx4070/pull/2) bumps gitpython 3.1.45 to 3.2.1. It is mergeable, and CodeQL is neutral. Dependabot's Gitpython Updates job on `master` succeeded on 2026-11-06. The same gitpython 3.2.1 pin is already in [PR #1](https://github.com/Bl4ckd09/nanochat-on-rtx4070/pull/1)'s `uv.lock`, so PR #2 is redundant if PR #1 lands. If PR #2 is merged on its own, first verify `uv sync --extra gpu` on the RTX 4070: its lock also rewrites torch wheel URLs, not only the gitpython line.
- [PR #1](https://github.com/Bl4ckd09/nanochat-on-rtx4070/pull/1) is a draft on `security/compatible-lock-updates` with green CodeQL. It raises `requires-python` to `>=3.11`, updates `uv.lock` and `.python-version`, and bumps dependencies such as datasets. It keeps `torch==2.9.1` and the CUDA 12.8 extra. In the same lock it already moves pillow 12.1.0 to 13.0.0, tornado 6.5.4 to 6.6, filelock 3.19.1 to 4.2.9, and gitpython 3.1.45 to 3.2.1. The author requires `uv sync --extra gpu` and tests on the Linux RTX 4070 before a merge decision. A Mac cannot verify GPU runtime.

## Why default-branch updates are red

Dependabot Updates jobs for Pillow and tornado last failed on 2026-11-14 KST. The last torch and filelock failures were on 2026-11-12 KST. On `master`, `requires-python = ">=3.10"` makes Dependabot resolve with CPython 3.10.22.

- Pillow 13.0.0 needs Python >=3.11. Version 12.1.0 is the latest resolvable on 3.10. Its job ended `security_update_not_possible`.
- tornado 6.6.1 needs Python >=3.11. Version 6.5.4 is the latest resolvable on 3.10. Same error type.
- filelock 4.2.x needs Python >=3.11. Version 3.19.1 is the latest resolvable on 3.10, which is why those jobs failed (for example run 43237331593).
- The torch job ended `dependency_file_not_resolvable`, not `security_update_not_possible`. Dependabot filtered torch 2.15.0 as needing CPython >=3.11, then failed to resolve a torch==2.13.0 candidate against the project's extras and supported Python range. The current pin is `torch==2.9.1`. Remaining Torch advisories need a verified replacement, and some list no patched version. PR #1 does not bump torch. Do not bump torch here.

## Next steps

1. Prefer landing PR #1 over PR #2. PR #2 is only useful if PR #1 is not merged, and even then the 4070 `uv sync --extra gpu` check should come first.
2. On the Linux RTX 4070, check out PR #1, run `uv sync --extra gpu`, `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`, and `uv run pytest tests/`, then decide whether to merge it.
3. After PR #1 lands, check that the open alerts for pillow, tornado, and filelock close. Do not expect Dependabot to need a further bump for those three. Do not bump torch merely to silence advisories without RTX 4070 verification. The torch resolvability error is a separate issue from those three and is not shown to be fixed by the 3.11 floor alone.
