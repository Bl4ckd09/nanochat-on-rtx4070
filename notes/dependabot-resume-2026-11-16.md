# Dependabot resume note, 2026-11-16

Related: [issue #3](https://github.com/Bl4ckd09/nanochat-on-rtx4070/issues/3). Leave both PRs open until Sun asks for a decision. Do not merge either PR yet.

## Open PRs

- [PR #2](https://github.com/Bl4ckd09/nanochat-on-rtx4070/pull/2) bumps gitpython 3.1.45 to 3.2.1. It is mergeable, and CodeQL is neutral. Dependabot's Gitpython Updates job on `master` succeeded on 2026-11-06.
- [PR #1](https://github.com/Bl4ckd09/nanochat-on-rtx4070/pull/1) is a draft on `security/compatible-lock-updates` with green CodeQL. It raises `requires-python` to `>=3.11`, updates `uv.lock` and `.python-version`, and bumps dependencies such as datasets. It keeps `torch==2.9.1` and the CUDA 12.8 extra. The author requires `uv sync --extra gpu` and tests on the Linux RTX 4070 before a merge decision. This Intel Mac cannot verify GPU runtime.

## Why default-branch updates are red

As last seen on 2026-11-14 KST, Dependabot Updates jobs for Pillow, tornado, torch, and filelock failed. On `master`, `requires-python = ">=3.10"` makes Dependabot resolve with CPython 3.10.22.

- Pillow 13.0.0 needs Python >=3.11. Version 12.1.0 is the latest resolvable on 3.10. Its job ended `security_update_not_possible`.
- tornado 6.6.1 needs Python >=3.11. Version 6.5.4 is the latest resolvable on 3.10.
- torch 2.15.0 needs CPython >=3.11. The current pin is `torch==2.9.1`. The remaining Torch advisories need a verified replacement, and some have no listed fix (see PR #1). Do not bump torch here.
- The filelock jobs also failed. Inspect them after the Python 3.11 move.

## Next steps

1. When Sun is ready, merge or close PR #2 as an independent gitpython bump.
2. On the Linux RTX 4070, check out PR #1, run `uv sync --extra gpu` and the usual tests, then decide whether to merge it.
3. After PR #1 lands with `requires-python >=3.11`, re-run or wait for Dependabot. Pillow and tornado security updates should become resolvable. Do not bump torch merely to silence advisories without RTX 4070 verification.
