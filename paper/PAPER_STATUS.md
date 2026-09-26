# PAPER_STATUS — Working Draft v0 (+ analysis update), 2026-09-26

| Item | Status |
|---|---|
| Manuscript | `paper/main.tex` + `sections/`, `appendix.tex`, `references.bib`; all sections in full sentences |
| Title | *Separating Optimizer-State Interventions from Update-Scale Effects in Continual Learning* ("Without Relearning Features" retired: every branch updates its weights) |
| Submission state | **INTERNAL WORKING DRAFT — NOT SUBMITTED.** Anonymous official style (no `\iclrfinalcopy`, no submission ID, no URL). The "Under review" header comes from the unmodified style; a banner under the title states the true status. |
| Format | **Official ICLR 2027 style, unmodified** (zip sha256 `0d940dfa…`, per-file hashes in `iclr_style_official/SHA256SUMS`) |
| Build | **BUILT**: `paper/main.pdf`, 10 pages total; main text (Sec. 1–7) ends on page 7 (limit 9); references p. 8; appendix pp. 8–10. Final log: no undefined refs/citations, no overfull boxes, labels stable. Pages rendered and inspected (pdftoppm). |
| Numbers | 279 macros generated from raw files by `tools/export_numbers.py`; provenance per macro in `tables/provenance.tsv` |
| Human review | **HUMAN_REVIEW_PENDING** (code, analysis, citations, text) |
| Science | STATE_REPAIR_BRANCH_ON_HOLD, NOT_READY_FOR_PILOT (unchanged) |

## Commits
- v0 (existing results only): `f586ad5`.
- Then: counterexample + raw-trace analysis (`analysis/counter_equivalence.py` -> `runs/p4_analysis_counter/`), manuscript update, build.

## Corrections made during this pass (both kept visible)
- Earlier internal report said reset_t == norm control "exactly" and ratio "0.3162": corrected to conditional (eps=0, v>0; finite-age kappa) — manuscript App. G; pointer appended to `docs/STAGE1_REPORT.md`.
- A draft sentence claimed "every exploding tensor has ratio ~1"; raw values show 17/20 within 1% and max 1.54 — text replaced by exact macros before the build.

## Not run (App. F)
LR-matched control for artifact arms; probe-objective LR tuning; more probes / fresh seeds / real data; coordinate-level traces; any trigger/predictor.

## Budgets (paper/cpu_ledger.tsv)
analysis 31.5 s / 900 s; build 22.4 s / 600 s (includes failed build attempts). Downloads recorded in BUILD.md.
