# Servatus clean-break implementation

Approved by the user on 21 September 2026: implement the assessment in full, with no legacy shims,
stale remainders, or backward compatibility. Baseline: 5efa7dd97bb7a46a8a02d8f2abcb7ba736ad35e9.
The accompanying audit-assessment.md supplies evidence, not a requirement to retain old interfaces.

## Destination

One ordinary typed campaign model and explicit atomic store transactions; one transient observation
projection and one eligibility policy; explicit roster authoring and read-only planning; each
attempt owns its original resolved execution configuration and explicit intent/outcome chronology.
No compatibility decoder, legacy aliases, duplicate engine, ORM, service, scheduler registry, or
speculative framework. Configuration remains strict and dependency-free. Application meaning stays
opaque. Preserve useful existing operations, not old representations.

Fixed rosters are sealed at creation by default. Expose create/load/append/seal explicitly; remove
Campaign.open and implicit CLI plan authoring. A plan is a compact reviewed intent, not a serialized
observation graph. It carries the fields needed to reconstruct exact selected allocations against
the campaign revision, with one useful integrity digest. Inspect remains ephemeral. All historical
attempts count toward retry safety; each is observed on its own original route. Only selected profile
semantics need validation, while whole TOML syntax and unknown-key validation remain strict.

Concrete workflow: Campaign.create(path, tasks, appendable=False), Campaign.load(path), and
campaign.append(new_tasks). Append accepts only the new suffix. Campaign.plan(profile, probe=None,
retry=(), allow_duplicate_risk=(), tasks_per_allocation=None) gathers its own current evidence;
remove the need to authenticate a caller-built CampaignView. inspect remains independent diagnostics.
Saved result-aware plans retain probe_required so restoring them cannot lose the re-probe obligation.
Submission interruption by KeyboardInterrupt/SystemExit still propagates, preserving durable intent;
ordinary operational failures return complete structured partial outcomes. Include an observed but
not durably recorded receipt separately when persistence fails after Slurm acceptance.

Planning limits its batch to max_allocations_per_submit and exposes deferred tasks. Submit attempts
the entire reviewed batch and reports receipts plus unresolved/unattempted work when interrupted by
an operational failure or concurrent change. Keep conservative submission ambiguity and explicit
operator reconciliation/resolution. Do not silently truncate or automatically retry uncertain work.
An identical receipt is idempotent; conflicting outcomes fail. Unrelated authoring must not discard
an observed receipt, but must stop further allocations from a stale reviewed plan.

No hidden self-authentication of frozen Python objects. Validate external JSON/TOML and coherent
current state once; retain meaningful schema/reference/uniqueness/chronology checks. Persist explicit
intent/outcome revisions instead of reconstructing histories never stored. Keep a simple redacted
operational record only as a nonauthoritative projection, without redundant digest protocol.

## Protected behavior

Persist unresolved intent durably before scheduler contact. Local deterministic render/command
validation happens before intent. Unknown launch outcome never proves rejection. Overlapping
unresolved work blocks submit; active or held work blocks retry; unknown accepted work requires
explicit retry plus duplicate-risk acknowledgement. Refresh safety evidence before atomic claim.
Accepted jobs, terminal scheduler evidence, valid application results, result readiness, and
quiescence remain distinct. Remote work stays outside store locks.

Preserve exact per-task resources, balanced packing, binary stdin and argv quoting, strict receipts,
job identity/cluster/accounting-window/requeue checks, private state/log redaction, bounded external
input, owner-only/no-follow filesystem checks, atomic publication and commit-before-cleanup ordering.
Keep cooperative Linux directory fallback and independent child workspace concurrency. Never hold
the shared parent coordination lock during fsync. No live external services or real output paths
from tests.

## Slices

### 1. Execution correction

Expected outcome: every SSH operation is bounded and locally cleaned up, generated steps preserve
GPU selection and payload correctness, and retained scheduler work cannot be mistaken for terminal
work. Inspection uses bounded job batches instead of two SSH launches per historical attempt.

Own transport/script/scheduler implementation and focused tests. Decode payloads into private
temporary files with checked decoder status before passing stdin to steps; workers may close stdin
early. Preserve exact binary input, checked scratch creation, cleanup after all siblings and on
interrupt, and wait for all started siblings. No new remote Python dependency. Explicitly forward
step-local CUDA visibility, address CUDA_DEVICE_ORDER consistently with Slurm's documented ordering,
and keep restricted submission environment. Correct held/requeued states and retain positive queue
evidence independently from accounting. Batch by target/cluster with bounded command/output chunks,
matching immutable identity; preserve history anchoring and fail closed on contradictions. Keep
local command bounds preflight before durable claims. Lifecycle guards own all post-spawn failures.

### 2. Filesystem correction and simplification

Expected outcome: durable workspace initialization includes the parent entry; entry cannot expose
redundant private work after compliant publication; root/child construction and cleanup express only
the documented safety guarantees.

Own Workspace/POSIX/tests and their docs. Sync parent before authoritative identity, outside shared
coordination. Recheck root/child destination absence after leases. Replace object.__new__ with one
immutable location descriptor and a live lease session. Remove per-file cleanup hard-link pins and
same-fd identity tautologies under trusted quiescent cleanup, retaining meaningful path checks and
commit hard links. Simplify callback factories into direct workflows where that improves clarity,
sharing deep commit/cleanup invariants. Preserve fallback lock lifetime and child concurrency.

### 3. Campaign clean replacement

Expected outcome: one typed state authority makes receipt persistence robust to unrelated mutations,
resource adjustments preserve attempt history, plans are compact decisions, and public authoring,
planning, submission, inspection, recovery and diagnostics have one coherent clean interface.

Own campaign/config/store/model/CLI/public exports and their tests/docs. Implement destination above
as one coherent schema/API break. Remove obsolete implementations and tests of deleted representation
details; preserve or rewrite behavioral safety coverage. Final tests should be lean, focused public
workflows plus genuine primitive fault tests. No transition-check tests for absence of old APIs.
Update version to 0.9.0 and lock metadata. Rewrite README/context/security/ADRs as necessary to state
current behavior; remove stale implementation narration and superseded normative architecture text.
Keep useful proven native-lane/fallback assumptions without claiming new live acceptance.

### 4. Final integration and cleanup

Expected outcome: independently reviewed slices compose into an installable coherent package, with
no obsolete schema machinery, temporary investigation/spec/ledger files, misleading docs, or failing
repository gates. Integrate onto original main and remove this run's branch/worktree.

Run pytest, Ruff lint/format, Pyright, Vulture, wheel and sdist builds and isolated installed-wheel
import, both CLI entrypoints, and a synthetic public workflow. Verify source distributions too.
No push, release publication, live cluster interaction, or external service mutation.

## Review protocol

Implementers read implement skill; independent reviewers read code-review skill and run separate
Standards/Spec agents over immutable baseline/head. Orchestrator owns ledger/integration only.
One writer per slice. Implementers commit scoped changes with conventional prefixes and report
checks; no self-review. Reviewer reports actionable findings; same implementer corrects them and
same reviewer checks correction deltas. Advance only after both review axes are clear.
