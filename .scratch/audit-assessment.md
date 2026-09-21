# Servatus audit assessment

Reviewed 21 September 2026 against commit `5efa7dd97bb7a46a8a02d8f2abcb7ba736ad35e9`.
Input: `/Users/edo/Downloads/servatus-audit`. Four independent investigations covered campaign
state, execution, filesystem safety, and public workflow. The primary investigation reproduced
the bundle and checked both patches against the complete repository in a temporary checkout.

## Decision

The audit is substantially valid. Its five headline findings justify corrections, and its central
architectural recommendation is supported by the code: replace duplicated campaign authority with
typed state, explicit transactions, one observation projection, and one eligibility policy.
Fundamental changes to the campaign schema and public workflow are justified.

The supplied patches are useful evidence, but are not ready to integrate unchanged. They omit
several correctness gaps, change shell behavior, and require coordinated test/documentation work.
The audit's proposed filesystem simplifications deserve selective adoption; cancellation, transfer,
and deployment features are separate product decisions.

No production code was changed during this investigation. This report is the repository deliverable.

## What was independently verified

| Check | Current checkout | Temporary checkout with both patches |
| --- | --- | --- |
| Source identity | Exact audit commit and both source blob hashes match | Both patches apply cleanly |
| Full repository pytest, Python 3.11.15 on macOS | 481 passed, 1 skipped | 477 passed, 4 failed, 1 skipped |
| Ruff lint | Pass | Pass |
| Ruff format | Pass | One Slurm formatting failure |
| Pyright strict | Pass | Pass |
| Vulture | Pass | Pass |
| Wheel and source distribution build | Pass | Pass |
| Isolated wheel import and both CLI help entry points | Pass | Pass |
| Additional installed Python 3.11 smoke | Import passes | Campaign create/seal/load and Workspace entry pass |

The bundle's independent harness reproduced exactly: **35 passed** with its modified sources;
**15 failed, 20 passed** with its originals. Those are test cases, not fifteen independent bugs.
The bundle tests isolated modules and a five-line Workspace function; it is not an installed-package
or full-Workspace certificate.

The four patched-suite failures have identifiable causes:

1. An assertion requires an unanchored running job to become `UNKNOWN`; the patch deliberately
   changes that policy.
2. An operational-record test hardcodes a digest derived from generated script bytes; the script
   changed.
3. A UTC/reconciliation test mocks `subprocess.run`; the unified transport uses `Popen`, so its
   observation mock supplies the wrong scheduler row shape. This is a test-seam mismatch, not
   evidence of a new UTC defect.
4. A Workspace test explicitly requires the parent directory not to be synced. That assertion
   contradicts the documented durability contract.

All execution tests used local synthetic processes and temporary directories. No live SSH,
Slurm, Apptainer, GPU, external storage, or application outputs were contacted. Linux CI,
power-loss durability, and cluster behavior were not verified in this investigation.

## Findings and integration decisions

### Execution and scheduler evidence

**Accept: bounded transport for every SSH operation.**
[The current submission transport](/Users/edo/dev/python/servatus/src/servatus/_slurm.py:215)
has neither a deadline nor streaming output limits. Validation and reconciliation share it.
Use one bounded transport, with binary stdin, simultaneous stdout/stderr draining, and local child
cleanup. A temporary stdin file is a reasonable implementation. Keep the existing rule that an
uncertain result after submission may represent an accepted job; never automatically replay it.

**Improve the patch: validate local command limits before recording an intent.** The patch adds
4 KiB field and 16 KiB command bounds to submission, but planning does not enforce them. A
synthetic target accepted by the Python interface with a 4,097-character host produces an
`UNRESOLVED` attempt and `AmbiguousSubmission` even though `Popen` was never called. Complete
deterministic rendering and transport validation before the durable claim. Failures after launch
must retain conservative ambiguity handling.

**Additional defect: child cleanup starts too late.**
[Selector creation](/Users/edo/dev/python/servatus/src/servatus/_slurm.py:243) occurs after spawning
the child but before the cleanup guard. Injecting selector-construction failure leaves the child
alive and its stdout/stderr open, in both original and patched implementations. Resource ownership
must cover every fallible operation after process creation.

**Accept: preserve each step's GPU selection explicitly.** The wrapper must run inside `srun`,
after step allocation, and forward that step's `CUDA_VISIBLE_DEVICES` through Apptainer's explicit
environment mechanism. `--nv` supplies NVIDIA support but does not establish the scheduler's
CUDA selection. Missing required visibility should fail clearly. This follows
[Apptainer's GPU guidance](https://apptainer.org/docs/user/main/gpu.html) and
[Slurm's step GRES behavior](https://slurm.schedmd.com/gres.html).
The proposed `SLURM_EXPORT_ENV=ALL` is consistent with
[Slurm's export guidance](https://slurm.schedmd.com/sbatch.html).
Actual device isolation still depends on site configuration.
The `CUDA_DEVICE_ORDER` contract also needs review: Slurm documents `PCI_BUS_ID` ordering when
aligning CUDA enumeration with NVML numbering. This remains an environment question for controlled
cluster acceptance, not a reproduced hardware defect.

**Accept, but broaden: retain positive evidence of live work without accounting.**
[The current merge](/Users/edo/dev/python/servatus/src/servatus/_slurm.py:595)
erases verified queue evidence when accounting is absent. That documented policy can permit an
acknowledged retry of work just observed running. Evidence sufficient to block retry need not be
sufficient to prove terminal completion. Keep job identity checks and accounting/requeue rules.

**Additional scheduler defect: held work is not terminal failure.**
[The state table](/Users/edo/dev/python/servatus/src/servatus/_slurm.py:103) classifies
`SPECIAL_EXIT` as `FAILED`. Slurm defines it as a special form of `REQUEUE_HOLD`; the job is retained
and can be released. Treating it as terminal can incorrectly establish quiescence and allow explicit
retry without duplicate-risk acknowledgement. `REQUEUE_HOLD`, `REQUEUE_FED`, and `RESV_DEL_HOLD`
also need retained-work handling; the patch still loses their unanchored queue evidence.
Use an explicit retry-blocking fact alongside the displayed normalized state, rather than forcing
all safety information into a small display enum.
[Source: Slurm job states and flags](https://slurm.schedmd.com/job_state_codes.html).
The synthetic public workflow reports `FAILED`, `quiescent=True`, and one retry allocation without
duplicate acknowledgement for `SPECIAL_EXIT`, on both original and patched code.

**Accept bounded query batching after the state redesign.** Current inspection starts two SSH
processes per accepted historical attempt. Group requests by each attempt's target/cluster, then
chunk by command and output limits. Match replies by allocation identity as well as job number;
do not rely on response order. Preserve duplicate accounting records, original submission-window
anchoring, and later requeue incarnations. Both
[squeue](https://slurm.schedmd.com/squeue.html) and
[sacct](https://slurm.schedmd.com/sacct.html) support job lists. A single unbounded query is not a
replacement for two bounded queries per job.

**Accept the decoder-failure finding; decide the stdin contract before adopting `pipefail`.**
The current shell pipeline can report success after decoder failure. The supplied Bash fix detects
that, but also fails a successful worker that closes stdin early when an upstream producer gets
SIGPIPE. A synthetic large-payload case reproduces that behavior. This is a contract change, not
a harmless shell substitution. For a generic opaque-task library, prefer checked decoding into
private temporary storage followed by stdin redirection if early close is allowed; account for
cleanup and remote scratch availability. Alternatively, deliberately require complete payload
consumption and test/document that requirement. In either design, wait for every started sibling.

### Campaign state and public workflow

**Accept and prioritize: record receipts against the attempt, not the global revision.**
[Receipt persistence](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:1530)
rejects a valid receipt after an unrelated append or seal. A synthetic submission that appends one
task while returning `4242;alpha` leaves durable acceptance `UNRESOLVED`. Resolve the matching
immutable intent under lock, accept an identical existing receipt idempotently, and reject conflicts.

Deleting that revision check alone is unsafe. The
[history validator](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:2195)
assumes immediate intent/outcome chronology within a plan. A fault-model with only relaxed receipt
persistence, two allocations, and an intervening seal submits the second allocation then leaves a
state that `Campaign.load` rejects. This is an integration trap introduced by the naive fix, not
an additional baseline defect. Store explicit intent/outcome revisions and stop further submission
when unrelated changes invalidate the remaining reviewed plan.

**Accept: a clean replacement of campaign representation.**
[Inspection](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:950) and
[view verification](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:1637)
independently reconstruct task evidence, ambiguity, readiness, and quiescence.
[Planning eligibility](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:1724) and
[pre-submit refresh](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:1393)
repeat policy. The
[transaction context](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:1553)
temporarily installs a descriptor on the Campaign object. These are concrete sources of coupling.
Splitting the existing dictionary machinery into more files would preserve them.

**Accept: explicit authoring, read-only planning.**
[CLI plan](/Users/edo/dev/python/servatus/src/servatus/cli.py:157) authors the roster before loading
the profile or publishing output. Both malformed configuration and an occupied plan destination
were reproduced leaving new tasks durably registered after command failure. Separate create,
load, and append from planning; this removes the need for cross-file rollback. Preserve explicit
append/seal capabilities, which are documented product behavior. Fixed rosters can be sealed at
creation by default.

**Accept: make reviewed submission scope explicit.**
[Submission](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:1268) slices allocations by
the submit cap and returns only receipts. A three-allocation plan with cap one returns one receipt;
replaying the same plan is stale. Prefer enforcing the cap during planning, showing deferred tasks,
and submitting the complete reviewed batch. Report accepted, unresolved, and unattempted work
when execution stops partway. A refusal-only fix is insufficient for large rosters unless planning
also supports bounded batches.

**Accept: reject NUL keys at construction.**
[Task validation](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:123) permits a key that
[saved-plan parsing](/Users/edo/dev/python/servatus/src/servatus/_campaign.py:2335) rejects.
The public plan/serialize/restore roundtrip fails. Use one key contract at both input seams.

**Adopt as a deliberate redesign: attempt-specific execution configuration.** The current
campaign-wide target/resource freeze is documented policy. Moving immutable resolved execution
values into each attempt enables memory/time adjustments without splitting history. Inspection,
logs, reconciliation, and retry safety must use each historical attempt's original route. The
application still owns whether changing an image or work root preserves task meaning; hashing a
pathname does not pin its contents.

**Qualify the audit's configuration advice.** Unknown keys are already rejected at every TOML
level. Whole-document profile validation is deliberate and tested. Selected-profile semantic
validation is a reasonable usability change, not a correctness repair; TOML syntax remains
document-wide regardless.

### Workspace and publication

**Accept the parent-directory fsync correction.**
[Initialization](/Users/edo/dev/python/servatus/src/servatus/_workspace.py:414) syncs the private
children but omits the parent entry that names the new container. The creation helper supplies no
equivalent sync. Sync the parent before identity becomes authoritative. The requirement follows
[fsync semantics](https://man7.org/linux/man-pages/man2/fsync.2.html).
A full-Workspace synthetic failure check confirmed that parent-sync failure propagates, no identity
becomes authoritative, and a later entry can recover.

Keep this fsync outside the shared parent coordination lock. The
[repository history](/Users/edo/dev/python/servatus/docs/research/campaign-engine-implrevloop.md:1626)
records a CephFS convoy from holding that lock across stalled synchronization. The supplied patch's
placement preserves the shorter lock lifetime. Correct the contradictory test and stale ADR wording.

**Additional defect: recheck destination absence after acquiring the workspace lease.**
[Root entry](/Users/edo/dev/python/servatus/src/servatus/_workspace.py:199) and
[child entry](/Users/edo/dev/python/servatus/src/servatus/_workspace.py:263) check too early.
A competing compliant Workspace can publish and clean its private tree between the check and
lease acquisition. The late opener then recreates private state and permits redundant application
work despite an existing canonical result. Reproduced for root and child. Canonical contents remain
protected from overwrite; the defect is unnecessary execution and stale private state.
Check again after acquiring the relevant leases and before returning access to the caller.

**Adopt selective simplification, preserving the actual safety contract.** The
[per-file cleanup hard-link pin](/Users/edo/dev/python/servatus/src/servatus/_posix.py:518)
does not make pathname verification and unlink atomic. Under the documented trusted, quiescent
cleanup model, direct descriptor-relative unlink can remove this machinery. Keep no-follow traversal,
name/inode checks, root substitution handling, and final synchronization. Publication hard links have
a different purpose and must remain where they provide create-if-absent semantics.

Replace [Workspace.child construction](/Users/edo/dev/python/servatus/src/servatus/_workspace.py:176)
with one immutable location/identity description and a separate live descriptor/lease session.
Simplify same-descriptor identity comparisons, while preserving live permission checks and
pathname-to-descriptor checks. Two explicit file/directory publication workflows are worth comparing
with the callback-heavy transaction, but this is lower priority than campaign ownership.

Retain the documented cooperative Linux directory-publication fallback. Repository evidence shows
it supports the actual cluster filesystem. Removing it would require a deployment capability
decision; it is not an undisclosed atomicity defect.

## Recommended clean-break destination

Keep the package dependency-free unless evidence establishes a need otherwise. Use ordinary typed
values and functions with these responsibilities; the exact filenames are secondary:

| Module responsibility | Owns | Must not own |
| --- | --- | --- |
| Configuration and task inputs | Strict external decoding; immutable task definitions and execution values | Application result meaning |
| Campaign store | Typed state, pinned directory, lock, atomic commit, explicit transition revisions | Scheduler contact or transient observations |
| Observation and eligibility | One projection and one decision over all historical attempts | A second verifier that recomputes the same policy |
| Execution adapter | Rendering, bounded transport, identity-aware observation and submission | Campaign persistence |
| Campaign orchestration | Review, fresh checks, durable claim, receipt resolution, bounded batches | Hidden transaction descriptors or parallel authorities |
| Workspace/publication | Resumable private state and durable no-clobber publication | Scheduler or scientific completion |
| CLI | Input/output adaptation and diagnostics | Implicit roster authoring or duplicate retry rules |

Persist one typed State containing tasks, roster phase, revision, and Attempts. Each Attempt retains
its immutable execution configuration, selected keys, explicit intent revision/time, retry choices,
acceptance outcome, and outcome revision. Keep one canonical reviewed-intent/script digest where it
has an external integrity use. Remove transient-object self-digests, overlapping hashes of the same
facts, and reconstruction of history that was never stored.

A Snapshot is transient scheduler/result evidence projected from state. A saved Plan is a compact
reviewed decision containing campaign identity/revision, selected work, execution values, retry
choices, and rendered-allocation identity. It does not need to serialize and authenticate an entire
Snapshot graph. Decoding still validates schema, references, uniqueness, and consistency. Submission
must refresh safety evidence, check the reviewed decision, and claim work atomically.

Preserve these distinctions: acceptance versus completion; result validity versus scheduler state;
results ready versus quiescent; latest displayed attempt versus every attempt relevant to retry.
An unresolved intent blocks overlap. Positive retained-job evidence blocks retry. Unknown accepted
work requires explicit retry and duplicate-risk acknowledgement. No redesign should weaken these
properties to save code.

Create a single replacement schema and interface. Reject old unsupported state and saved plans
clearly; regenerate unsubmitted plans. Do not add legacy decoders, compatibility engines, or shims.
Do not overwrite or delete existing campaign data as part of the code change. Deployment must
account for any outstanding old campaigns before newly authored work can safely be replayed.

## Implementation and review sequence

1. **Execution and filesystem correctness, independent workers.** Execution owns transport,
   scheduler state handling, GPU forwarding, and payload failure semantics. Filesystem owns parent
   durability and post-lease absence checks. Add focused public-flow/fault tests for these defects.
2. **Replace campaign state and transitions as one coherent slice.** One owner handles typed
   decoding, transaction ownership, explicit chronology, attempt-local receipt persistence, shared
   projection/eligibility, compact plans, and attempt-specific execution. Avoid a half-migrated state
   engine whose old history validator rejects new transitions.
3. **Complete the public workflow against that interface.** Explicit create/load/append, fixed-roster
   default, bounded plans with deferred work, structured partial outcomes, coherent key validation,
   selected-profile policy, CLI and documentation. A separate reviewer checks the real public
   workflow and concurrency/ambiguity cases.
4. **Simplify filesystem internals separately.** Replace duplicated Workspace construction and
   justified cleanup machinery after the correctness tests exist. Preserve independent sibling
   concurrency and cooperative fallback support.

Each substantial slice gets an independent review followed by corrections by its original owner.
Review observable behavior and required invariants, not preservation of deleted representations.
Run pytest, Ruff lint/format, Pyright, Vulture, builds, and isolated installed-artifact smokes at each
handoff. Linux/macOS CI and a separate controlled cluster acceptance exercise remain release gates.
Tests must continue to use synthetic fixtures; live acceptance does not belong in pytest.

Keep cancellation, data transfer, image deployment, generic scheduler plugins, services, and database
frameworks outside this correction effort. The
[current documented scope](/Users/edo/dev/python/servatus/README.md:426) explicitly excludes several
of those operations; their absence does not invalidate Servatus's execution/publication purpose.

## Reproduction artifacts

Temporary artifacts are available for this investigation; they are not permanent test dependencies:

- Patched source checkout: `/var/folders/y1/h6b6vjm114v6yhtrbzr877kc0000gn/T/servatus-audit-validation-xz04hq7j`.
- Full patched-suite failure log: `/tmp/servatus-patched-suite.txt`.
- Campaign receipt/NUL repro and naive-fix fault model: `/tmp/servatus-campaign-audit-repro.py`.
- CLI mutation/partial-submission/configuration repro: `/tmp/servatus-workflow-audit.py`.
- Root/child workspace race repro: `/private/tmp/servatus-workspace-race-repro-dzr1hdr8.py`.
- Retained Slurm state repro: `/tmp/servatus-special-exit-repro.py`.
- Transport cleanup, premature ambiguity, and early-stdin-close repro:
  `/tmp/servatus-execution-audit-repro.py`.
- Baseline and patched wheels/sdists: `/tmp/servatus-audit-baseline-dist` and
  `/tmp/servatus-audit-patched-dist`.

The temporary full-suite run used `PYTHONPATH="$PWD/src"` and verified `servatus.__file__` before
testing, so results refer to the patched sources rather than the main checkout's editable install.
