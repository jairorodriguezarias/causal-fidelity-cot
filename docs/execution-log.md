# Execution log

## 2026-09-25

- Inspected project state and preserved historical v3 outputs. No valid previous
  comparison between distilled and RL-post-trained checkpoints exists.
- Verified NVIDIA GB10, functional CUDA 13/PyTorch and NF4 CUDA forward computation.
  Existing unrelated GPU services remain active; models are loaded within available memory.
- Built and validated 500 construction / 50 calibration / 50 test contrast pairs
  from pinned official SNLI splits; no normalized-premise overlap.
- Implemented batched generation and source/config/dataset provenance, prefill-aware
  interventions, equal-norm controls, truncation/substitution/shuffle tests, conservative
  parsing, blinded assessment and missingness-aware analysis.
- Completed `runs/smoke-1.5b`: 51 records covering all configured conditions; independent
  audit passed. Four calibration entailment examples produced 0 correct baseline
  answers, with 2 unfinished thoughts, 1 invalid label and 1 incorrect contradiction.
  This engineering run demonstrates task-competence and completion limitations;
  it provides no estimate of successful baseline-entailment flips (denominator zero).
- Assessor synthetic validation: 12/12 cases. On 20 repeated real smoke assessments,
  agreement 0.85 and kappa 0.7222. This is not human validation.
- Larger 1.5B and 7B baseline calibration runs launched on 20 calibration pairs each.
  The 32B checkpoints are downloading at pinned revisions. The proposed held-out
  schedule remains a draft until calibration is inspected.
- Draft manuscript and bibliography prepared; first PDF compilation is in progress.
  Author name(s), affiliation(s) and contact email requested and still pending.

- 29 automated tests passed, including a real tiny-Transformer hook check and a gate that rejects the engineering smoke run as a held-out study.
- Draft PDF compiled successfully and first-page layout inspected; source archive generated, explicitly marked incomplete.
- The first 7B calibration process exited with SIGTERM after saving 8 records. Its matching source/config identity was verified and it resumed in `causal-fidelity-calibration-7b.service`.
- `causal-fidelity-study.service` waits for that live calibration service, checks its completed audit, then freezes the protocol and executes all 7B/32B study stages sequentially with logs.
- Extended 1.5B calibration completed: entailment 6/20 correct, contradiction 15/20 correct; 16/20 parseable answers in each class.
- 7B calibration completed and independently audited: 16/20 entailment and 17/20 contradiction answers correct; all other responses hit the thought-length limit. The frozen study uses 2048 tokens, batch size 16, eight layers and four intensities with both controls.
- Full campaign `runs/study-20260925` launched under the persistent study service. Check its live PID and stage before continuing; do not restart a live job.

## 2026-09-29

- Session log recorded before a planned machine power-off; no source, config or
  dataset files were modified in this session.
- Diagnosed the interrupted campaign: the machine was rebooted on 2026-09-28
  14:57 CEST and the campaign process died with the session; the systemd unit the
  README referenced no longer existed. The kill left a truncated write in
  `runs/study-20260925/distill-7b/judgments.jsonl` (line 330, NUL padding), which
  made `rejudge.py --run` fail with `JSONDecodeError` before it could resume.
- Verified campaign identity before relaunch: all eight frozen source files and
  `configs/study.json` match the SHA-256 values recorded in `campaign.json`.
- Repaired `judgments.jsonl` by keeping only valid JSON records (329 kept, 0
  duplicate keys, last key `gaussian_L0_a1.0_r0/644c6a7cbd62d68f/entail`). The
  pre-repair file is preserved at
  `runs/study-20260925-repair-backups/judgments.jsonl.corrupt`. `records.jsonl`
  verified intact: 5,229 valid records.
- Recreated the campaign as a systemd user unit
  `~/.config/systemd/user/causal-fidelity-study.service` running
  `scripts/run_campaign.py --config configs/study.json --name study-20260925`
  with `Restart=on-failure` (2 attempts / 300 s), enabled with linger. Resume
  semantics confirmed in code: completed stages are skipped and the LLM judge
  resumes from `judgments.jsonl` by record key rather than restarting.
- Verified resume in practice: after repair the judge continued from record 330
  and reached 1,020/4,850 (21%) at about 36 assessments/minute with no new
  errors. The two 7B-judge rows with `returncode=1` in `campaign.json` date from
  this recovery attempt and are expected, not a new failure.
- Remaining schedule after the 7B judge: 32B download (about 21 of ~65 GB cached
  at the pinned revision), then 32B and QwQ generation, audit, judge per model
  and the final model comparison. The 7B generate stage alone took ~15 h on this
  machine, so the full held-out campaign is a multi-day run.
- Recovery after any unexpected shutdown: `systemctl --user status
  causal-fidelity-study.service`; linger restarts it at boot. If a JSONL is
  truncated again, drop only trailing invalid lines and keep a backup copy first.
- Pilot observations inspected this session, descriptive only and pre-judge (7B,
  50 test examples): baseline accuracy 0.80 (Wilson CI 0.67-0.89), mean 417
  generated tokens; steering flip rates 0.05-0.13 across layers and alphas with
  orthogonal/Gaussian controls in the same range, so 7B flip evidence is not
  distinguishable from control noise except at L0 a=2.0 (0.128 vs 0.05/0.0, with
  overlapping intervals). No research claim follows from these numbers.
- Judge categories while the 7B judge ran at n=1,020 assessments: 753
  `NO_OBSERVED_SHIFT`, 204 `BASELINE_NOT_ENTAILMENT`, 39 `ALIGNED_SHIFT`, 18
  `UNASSESSABLE`, 7 `AMBIGUOUS`, 5 `ANSWER_REASONING_MISMATCH`, 2
  `REVERSE_MISMATCH`. Counts change while the judge runs; report final counts
  only from the completed run.
- Two trace contrasts inspected manually: example
  `54928c608d47f9ab/entail`, where `steer_L0_a0.5_r0` produced an
  `ALIGNED_SHIFT` (trace and answer both moved to CONTRADICTION and the trace
  re-derived the supporting distinction), and
  `steer_L0_a2.0_r0/2ebc92fd4bda7938/entail`, where trace and final answer
  disagree (`ANSWER_REASONING_MISMATCH`). Case studies only; see the judge
  statistics above before drawing any conclusion.
- Session discussion (not yet a planned stage): extend the same causal machinery
  to banking decision tasks to test whether the stated trace drives the decision
  and whether reasoning models are worth using there. Literature leads to verify
  before any citation: FaithCoT-Bench (arXiv 2510.04040), FinChain (ACL
  2026.acl-long.662), FinCoT (arXiv 2506.16123), FinanceReasoning (arXiv
  2506.05828) and a September 2026 Zenodo preprint on counterfactual
  self-explanation testing in credit adjudication that neutralises only the
  named field at n=50 across two gpt-oss models. The steering-with-equal-norm
  controls and the distilled-versus-RL comparison would be the missing pieces
  there; a future study needs rule-documented ground truth and a contamination
  statement.


## 2026-11-01

- Inspected the inactive campaign at the user's request. The last attempt failed
  on 2026-10-20 while resuming Distill-32B: `records.jsonl` line 901 contained
  2,176 NUL bytes. The previous status file claimed 916 records, but only 900
  complete JSON records were present on disk. There was no valid record after
  the corrupted tail.
- With authorization to continue, preserved the original records, status,
  campaign history and generation log under
  `runs/study-20260925-repair-backups/20261101T201652Z/`. The repair report
  records SHA-256 checksums for the originals and repaired records.
- Removed only the trailing NUL bytes. All 900 complete records were preserved
  byte-for-byte; keys are unique and stored token counts match token-ID arrays.
  Corrected the interrupted status to the actual record count.
- Verified all eight frozen scientific source hashes, the study configuration,
  and the Distill-32B generation source snapshot before resuming. No scientific
  source, model configuration, prompts, sampling settings or seeds were changed.
- Started `causal-fidelity-study.service` at 21:16:53 Europe/Madrid. The existing
  exclusive-vLLM resource policy pauses the local evaluator during generation
  and restores it for judging or when the service stops. The completed 7B
  stages are skipped; Distill-32B resumes from its 900 preserved records.
- Verified resumed generation at 21:25:41 Europe/Madrid: the first new batch
  saved 16 records (`orthogonal_L9_a1.0_r0`), bringing the count to 916.
  Every original valid byte remains unchanged; all record keys are unique
  and token counts match token-ID arrays. The service is active with no
  restarts; frozen source/configuration hashes still match. Verification
  is saved beside the repair report as `resume-verification.json`.

## 2026-11-02

- At the user's request, ran three agent tasks in parallel with the existing
  Distill-32B generation: evaluator audit, CPU-only statistical verification and
  figures, and a standalone interim manuscript. See
  `docs/parallel-work-20261102.md` for outputs and operational details.
- Verified the eight frozen scientific source hashes and study configuration;
  7B and 32B use identical evaluation IDs. No original generations or judgments
  were changed. Supplementary 7B robustness output was generated from its complete
  5,229-record run. Figure construction independently recomputed all 13 text and
  64 activation contrasts and verified their counts against existing reports.
- Assistant review covered all 42 automatically flagged forward/reverse
  mismatches and 30 sampled agreement records. Its exploratory proposals are
  26 agreement and 16 ambiguous cases among the flags; these are not human gold
  labels or a corrected whole-run error rate. All 72 evidence quotes and their
  source hashes were checked. The interpretation of
  `steer_L0_a2.0_r0/2ebc92fd4bda7938/entail` as a mismatch in the September 29
  entry is contradicted by its actual trace and answer; the stored evaluator
  verdict is erroneous. Details: `docs/judge-audit-20261102.md`.
- Built and visually checked `paper/interim-20261102.pdf` (six pages), explicitly
  marked as an incomplete working report. It includes verified figures, complete
  7B results and the assistant-audit limitations; the main manuscript is preserved.
- Started QwQ-32B prefetch at 09:44:38 CET as
  `causal-fidelity-qwq-prefetch.service`, using its frozen revision and the existing
  shared model cache. The download uses reduced priority, one CPU quota and a
  4 GiB memory ceiling; it does not load a GPU model. This transient unit persists
  after the conversation but is not installed to restart at machine boot.
- At 09:46:37 CET both generation and prefetch services were active with zero
  restarts. Distill-32B had just saved a new batch, reaching 2,082/5,236 records
  (39.8%). The prefetch cache contained growing partial downloads. QwQ generation
  remains scheduled after Distill under the existing single-GPU resource policy.
