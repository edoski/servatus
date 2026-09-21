# Implementation ledger (temporary)

Original main/worktree: /Users/edo/dev/python/servatus, main at 5efa7dd97bb7a46a8a02d8f2abcb7ba736ad35e9.
Original untracked file: docs/research/2026-09-21-audit-assessment.md (our preceding report).
Run worktree: /Users/edo/.codex/worktrees/servatus-clean-break/servatus.
Run branch: codex/servatus-clean-break.
User authorized full clean break, no legacy/stale compatibility code.

Approved spec: .scratch/clean-break-spec.md; evidence: .scratch/audit-assessment.md.
Execution stdin decision: checked private temporary file; early close permitted.
Campaign decision: explicit create/load/append/seal, fixed roster default; attempts own execution;
compact plans and bounded planned batches, structured partial outcomes; clean schema/API replacement.
No deployment or external service action authorized by this implementation run.

Slice 1: implementing; baseline aa339641948323395545a8850adcb10cc6d24d27;
worker /root/implement_execution. Independent read-only campaign design: /root/design_campaign.
Slice 2: next. Slice 3: pending. Final integration: pending.

Slice 1 implementation 898ab90df9ce00951e34f1925123141f57780997; reviewer /root/review_execution
reviewing aa33964...898ab90 with separate Standards/Spec agents. Worker gates: 509 passed, 1 skipped;
Ruff, Pyright, Vulture, builds and installed smokes passed.
Slice 1 GREEN LIGHT: reviewer Standards 0, Spec 0; 125 focused tests independently passed.
No correction round required.

Filesystem design clarification: retain one deep publication transaction owner unless direct
workflows actually reduce complexity without duplicating its exception/commit sequencing. No
mechanical callback deletion required. Root/child use immutable location + optional live lease;
ordinary root construction followed by one private location replacement avoids magic allocation
and nonexistent child-parent validation without adding public constructor knobs.

Read-only campaign design complete: four internal responsibilities (typed model/codecs, store,
observation/policy, orchestration). Plan gathers evidence internally; view authenticity protocol
removed. create uses appendable=False by default; append receives new suffix. Partial outcomes
distinguish observed-but-not-durable receipt. Interruptions propagate. These choices added to spec.
