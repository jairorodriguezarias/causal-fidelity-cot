# Evidence files

These files preserve the provenance of the study and dated diagnostic snapshots.
They do not form a live dashboard or a completed research release.

| File | Meaning |
|---|---|
| `dataset.json`, `DATA_LICENSE.md` | Selected SNLI pairs, revision, checksum and attribution |
| `protocol-freeze.json` | Local protocol freeze before held-out execution, 2026-09-25 |
| `environment.txt` | Original installed environment; not a portable lockfile |
| `judge-validation.json` | Synthetic evaluator fixtures |
| `judge-validation-results.json` | Stored synthetic validation output; not human validation |
| `test-results.json` | Historical test evidence from 2026-09-25 (29 tests at that time) |
| `release-audit.json` | Historical failed release check on a smoke run; not current campaign status |
| `diagnostics/20261102/` | Interim evidence, assistant review and verified figure tables |

The historical test and release reports are retained as evidence. Do not mistake
their old test count or stale-source messages for a fresh check of this checkout.
Run `env/bin/python -m pytest -q` for current tests. The GitHub preparation review
is recorded in [docs/github-review-20261102.md](../docs/github-review-20261102.md).

The diagnostic snapshots refer to raw records under `runs/`, which are excluded
from Git. Recomputing analyses or independently checking the quoted traces requires
those original records. Evaluator review proposals are not human gold labels and
do not replace the original stored judgments.
