# Campaign Engine Implementation and Review Ledger

Status: architecture and amended complete ledger independently reviewed GREEN; implementation
authorized. This is the active plan.

Date: 2026-08-13

## Objective

Deepen Servatus into a small, daemonless campaign engine for finite or appendable ML/HPC work on
Slurm. Servatus will own the generic execution facts shared by KAIROS and similar projects:

- one ordered immutable Task roster with append-only authoring and an irreversible seal;
- every submission attempt, scheduler acceptance, ambiguity, and explicit retry;
- read-only observation of accepted Slurm allocations;
- bounded diagnostic access to exact accepted allocation and packed-Task logs;
- one caller-supplied, in-memory result probe over opaque Tasks;
- result-aware planning and derived readiness without application schemas;
- one repository-local named execution-profile document that replaces paired client target/resource
  files;
- a redacted, canonical execution-provenance record;
- the existing Workspace and publication transactions, unchanged in responsibility.

KAIROS will adopt that deeper interface and delete its duplicate pre-publication roster. Scientific
objects, validation, grouping, manifests, metrics, and identifiers remain KAIROS-owned.

Client simplicity is a release criterion. The normal KAIROS path must read as direct composition of
one typed envelope, one domain result probe, public Campaign operations, and one scientific
publication function. No adapter class hierarchy, service object, repository layer, lifecycle
facade, generic helper package, duplicate status type, or compatibility branch is allowed. A
Servatus feature that cannot remove or materially simplify client responsibility fails the deletion
test and is redesigned before integration.

Servatus is not becoming a DAG framework, experiment tracker, model registry, artifact store,
scheduler daemon, or generic ML domain model. The target is one deep Campaign abstraction, not a
larger framework.

## Prior program closure

The historical KAIROS lifecycle-extraction ledger is complete for every authorized implementation,
review, release, integration, deployment, and cleanup action. It remains historical evidence and is
not extended by this program.

The completed baseline is:

- Servatus `0.6.0`, exact public source head
  `281c381548489c1dcf7a6ca8d045908d0b50ba3f`;
- KAIROS `main`, exact published head
  `ff2a9e26ba67a7b4b58cc9e389a4eb4e81ff7b95` on `origin` and `research`;
- KAIROS compact-CUDA, exact published head
  `45f27ef9ce345c59e4c469522e2aa184f613a9e6` on `origin` and `research`;
- production image `kairos-cuda-352cc96.sif`, built from compact source
  `352cc968853bda69b1a9131db709d48d8f1d2043`, SHA-256
  `4b56487c8ea046abd42c6e2bdd186a01b0d029997364f918779d99f101432ad5`;
- Blockweaver `0.3.4` and only `datasets/<uuid>` KAIROS inputs;
- the three legacy KAIROS corpus directories removed from active local and research storage, with
  the local recovery copy retained in the macOS Trash as separately recorded.

The first real production Servatus Campaign is current scientific work, not unfinished extraction
work. It is protected by the clean-break gate below. The separate inference cost/time benchmark
still has its own scientific readiness gate: before it runs, KAIROS must verify one canonical full
nine-chain-family-group by four-horizon roster, totaling 36 Artifacts and 36 Evaluations. That gate
does not belong to this implementation program.

The old ledger's stale introductory wording does not reopen completed work. It should be treated as
an archived historical record. This new Servatus-owned ledger is the sole implementation authority
for the Campaign-engine program.

## Fixed starting state

At plan creation:

- Servatus has one clean worktree on `main` at `281c381548489c1dcf7a6ca8d045908d0b50ba3f`;
- KAIROS has one clean worktree on `main` at
  `ff2a9e26ba67a7b4b58cc9e389a4eb4e81ff7b95`;
- the run-owned Servatus branch is `codex/campaign-engine-0.7`;
- the run-owned Servatus worktree is `/private/tmp/servatus-campaign-engine-0.7`;
- no KAIROS implementation branch or worktree exists for this program;
- no output, image, scheduler, campaign, remote checkout, or canonical object has been touched.

At implementation authorization, KAIROS `main` remained at the same exact commit but contained one
unrelated user-owned untracked document, `docs/app-ownership-simplification-implrevloop.md`. This run
must not read, edit, stage, remove, or claim that file. A separate task owns App simplification in an
isolated worktree. Servatus S1-S4 may proceed independently. KAIROS K1 must re-pin the then-current
accepted `main` after that task integrates; no current `app/` product edit is expected from this
program, while `docs/CONTEXT.md`, `docs/KAIROS.md`, ADR 0006, and ADR 0008 remain possible later
overlap surfaces.

Every slice records its exact baseline, head, worktree, commit range, status, and gates before the
next slice starts. Later baselines are the preceding independently accepted heads, never an
unreviewed moving branch.

## Protected completed Campaign

The Runner task owns the completed K-study evidence:

- K-study UUID `b0e6d421-86e9-4ef6-8d81-d02f355b2da0`;
- Servatus Campaign `9afb18b38088a3dfa88147011945ba7f`;
- exactly 24 non-`K=5` tasks, with three frozen `K=5` Artifacts referenced and not retrained;
- initial jobs `45085` through `45090`;
- never-started `BadConstraints` attempts `45085` and `45086`, whose exact eight keys were
  explicitly retried as jobs `45091` through `45094`;
- its retired hourly Runner heartbeat was named `kairos-final-k-study`.

This program must not inspect, edit, migrate, reopen, resolve, retry, cancel, reprioritize, or delete
that Campaign or its jobs. Servatus `0.7` state is a clean break and must reject `0.6` state. The
Runner has attested that every intended K-study Task has valid canonical evidence, no retry or
authoring remains, and no job or heartbeat still needs this schema-3 Campaign. The exact manifest is
`outputs/experiments/k_study/b0e6d421-86e9-4ef6-8d81-d02f355b2da0/manifest.json`. Current held-out
and future inference work remain outside this program.

Servatus implementation, local synthetic tests, and CI may proceed without touching the protected
Campaign. The isolated E0 live gate, stable `0.7.0` release, KAIROS adoption, dependency repinning,
new KAIROS Campaign creation, image construction, and deployment remain dependency-gated below.

## Domain boundary

### Identities

Servatus owns infrastructure identities whose sameness it defines:

- Campaign identity;
- Campaign revision;
- allocation and attempt identity;
- Slurm receipt identity;
- allocation, stage, inode, lock, and publication transaction identity.

KAIROS owns scientific identities whose sameness it defines:

- experiment UUID;
- Study UUID;
- Artifact UUID;
- Evaluation UUID;
- Dataset UUID association;
- the opaque semantic identity bytes supplied to each Workspace;
- cell labels and scientific group membership.

Servatus hashes, persists, binds, locks, and verifies Workspace identity bytes and filesystem inode
pins. It does not define which KAIROS requests are the same work. KAIROS supplies those semantic
identity bytes.

`Task.key` remains a caller-supplied stable opaque string. Servatus validates that keys are safe,
unique, immutable, and correctly bound to durable Task bytes. It never mandates UUIDs, parses a key,
or creates scientific IDs. There will be no `servatus.new_id()`, generic entity registry, universal
Task-to-result mapping, or Task metadata bag.

### Facts and authority

The implementation must keep these facts separate:

1. **Roster membership:** the Task belongs to the Campaign.
2. **Submission attempt:** Servatus recorded intent before contacting Slurm.
3. **Scheduler acceptance:** Slurm returned or reconciliation proved one job identity.
4. **Scheduler execution:** a current bounded query reports an allocation state.
5. **Application result validity:** the caller's probe validates the Task's canonical result.
6. **Aggregate publication:** the application validates and publishes its scientific manifest.

No one fact implies another. A receipt proves acceptance, not execution or a valid result. Slurm
`COMPLETED` proves process outcome, not scientific validity. A valid canonical result may override
stale scheduler information for completion selection, but it does not rewrite scheduler history.

### Ownership after this program

Servatus owns:

- Task roster integrity and ordered suffix growth;
- the open/sealed roster phase;
- strict repository-local execution-profile loading and exact selected-profile lineage;
- target/resource lineage;
- deterministic packing and rendered Slurm scripts;
- attempts, acceptance receipts, ambiguity, and explicit resolution;
- current scheduler observation for exact accepted allocations;
- result-observation orchestration around an opaque caller probe;
- retry eligibility derived from scheduler and result facts;
- derived Campaign result readiness and execution quiescence;
- redacted execution provenance;
- Workspace, Draft, file publication, no-clobber commit, and cleanup reporting.

KAIROS owns:

- strict request and execution-envelope schemas;
- Study, Artifact, Evaluation, Dataset, experiment, and cell meaning;
- task keys and worker bytes;
- canonical object addresses and association checks;
- result probes implemented through KAIROS loaders;
- Study fan-in and candidate completeness;
- HPO extension rules and scientific roster construction;
- manifest construction and publication;
- metrics, figures, benchmark protocols, and scientific readiness;
- profile selection, committed profile values, one-GPU policy, image selection, and external queue
  policy.

## Accepted Campaign model

### One deep public authority

`Campaign` remains the public execution authority. Do not add `Workflow`, `RunSet`, `JobGroup`,
`WorkspaceGroup`, `Pipeline`, or a second runner facade.

The intended Python path is:

```python
profile = Profile.load(Path("SERVATUS.toml"), name=selected_profile)
campaign = Campaign.open(path, tasks)
campaign.seal()  # immediately for fixed rosters; later for appendable authoring
view = campaign.inspect(result_probe)  # includes scheduler evidence for planning
plan = campaign.plan(
    profile,
    view=view,
    retry=explicit_retry_keys,
    tasks_per_allocation=cap,
)
campaign.validate(plan)  # optional bounded Slurm --test-only check
receipts = campaign.submit(plan, probe=result_probe)
```

Exact public type and property names may change during a slice only when the implementation and
independent reviewer show a smaller, clearer interface with the same single path and semantics.
Adding a second equivalent path is not allowed.

### Repository-local execution profiles

One public immutable `Profile` groups one complete execution lane: its opaque label, one
`SlurmTarget`, and one homogeneous `ResourceRequest`. `Campaign.plan()` consumes the Profile rather
than a separate target/resource pair. Campaign compatibility binds only to the exact resolved target
and resource values. The selected label is retained as nonbinding plan, attempt, and operational
provenance so humans can identify the authored lane. Two labels resolving to identical values are
compatible; changing the resolved values behind one label fails ordinary Campaign lineage
comparison.

Profiles live in one client-owned, version-controlled `SERVATUS.toml` at the repository root:

```toml
default_profile = "KAIROS"

[profiles.KAIROS.target]
host = "research"
# complete target values

[profiles.KAIROS.resources]
cpus_per_task = 24
memory_mib_per_task = 65536
gpus_per_task = 1
time_limit = "3-00:00:00"
```

The document root allows exactly `profiles` plus optional `default_profile`. `profiles` is a
nonempty table. Every declared label must be a nonempty string key and every declared profile must
contain exactly one complete `target` table and one complete `resources` table. Loading validates the
whole document once, including unselected profiles; one malformed profile invalidates the document.
Profiles do not inherit, merge, reference one another, or split target and resource ownership.
Selection is exactly:

1. an explicit profile name supplied by the caller;
2. otherwise the document's optional `default_profile`;
3. otherwise one direct configuration error.

An explicit name always overrides `default_profile`. A missing document, missing selected label, or
default label absent from `profiles` fails directly. Existing target/resource typed validation owns
the nested raw values exactly once. There is no environment fallback, repository-name inference,
parent-directory search, global store, path-to-project registry, per-profile default boolean, or
implicit single-profile selection.

The library interface loads one explicit path supplied by its caller. Both Servatus and KAIROS CLIs
read exactly `Path.cwd() / "SERVATUS.toml"`; there is no config-path option and no parent-directory
search. They permit `--profile NAME` and otherwise use the file's declared default. Invocation from
another directory therefore fails on the missing exact file rather than guessing a project root.
Git owns creation, editing, review, and history. Servatus adds no profile
install/remove/list/show commands. A separate `profile check` command is also excluded until real
use proves that normal load/plan errors are insufficient.

Repository placement provides discoverability and Git provenance, but it does not replace durable
Campaign lineage: a dirty file or another checkout can differ. No extra revision store or profile
digest subsystem is needed because the Campaign already persists and compares the exact resolved
execution values.

### Roster phase

Campaign state has one irreversible roster phase:

- `OPEN`: exact ordered suffixes may be appended; execution is allowed;
- `SEALED`: no Task may be added, removed, reordered, or changed.

`Campaign.seal()` is atomic and idempotent. Sealing increments the Campaign revision and makes every
older view and plan stale. `Campaign.open(path, exact_roster)` may reopen a sealed Campaign only with
the exact roster. Any suffix on a sealed Campaign fails. `Campaign.load(path)` remains the
non-authoring reopen path. `Campaign.tasks` returns the immutable authored Task tuple.

There is no durable `FINALIZED` phase. Readiness is derived from current facts. Scientific
publication remains an application action and cannot be inferred from Slurm or a generic callback.

### Attempts

One durable attempt record replaces parallel intent, receipt, and negative-resolution collections
only if the fixed-range implementation proves that the result is smaller and clearer. The record
must still preserve every accepted safety fact:

- attempt/allocation identity;
- ordered Task keys;
- Campaign revision;
- selected nonbinding profile label;
- target and resource lineage digests;
- plan and script digests;
- exact effective allocation totals;
- nonsecret `sbatch` argument vector;
- reconciliation time window;
- which selected keys were explicit retries;
- unresolved acceptance, accepted receipt, or explicit not-submitted resolution.

Intent is durably synced before possible scheduler acceptance. Acceptance updates the same logical
attempt after `sbatch` returns. Missing or unprovable acceptance remains ambiguous. Explicit
resolution and every retry stay visible; history is never rewritten.

If a single attempt model adds more machinery or duplicates facts, retain the current safe records
and improve their public projection instead. Schema shape is subordinate to one authoritative fact
and a deletion-tested implementation.

Every scheduler-observed Campaign view pairs each accepted attempt with its current observation. It
exposes all attempts and never collapses history into one destructive Task status. For display and
default selection, each Task uses its latest accepted attempt. Earlier attempts remain provenance.
An unresolved acceptance
intent dominates every affected Task until reconciliation or explicit resolution. A later accepted
retry supersedes the earlier attempt only for current operational projection, never for history.

If any accepted attempt for a Task is queued or running, retry is rejected regardless of attempt
age. If any accepted attempt is unknown, retry requires naming the exact Task key explicitly and is
a recorded
operator override acknowledging that accounting cannot prove the older attempt inactive. Servatus
must make this risk explicit in the returned plan and ordinary CLI output. It never performs that
override automatically.

### Application result probe

The only application hook is a plain synchronous callable:

```python
ResultProbe = Callable[[Task], bool]
```

Its contract is:

- return `True` only after validating the canonical result for that exact Task;
- return `False` only when the canonical result is absent or incomplete;
- raise when a present result is invalid, corrupt, mismatched, or cannot be trusted.

The probe must be read-only, side-effect-free, bounded, and deterministic for one underlying
canonical state. A `True` result must be immutable or version-addressed so it cannot later become a
different valid object at the same address. Servatus cannot enforce those application guarantees;
they are the explicit caller contract. Probe calls that occur before a later probe raises have still
executed, so callers must not depend on rollback of probe side effects.

The callable is supplied per inspection. It is never serialized, registered, imported by name,
stored in Campaign state, or invoked by a daemon. Servatus never interprets its schema. Servatus
must not call it while holding a Campaign filesystem lock.

Inspection follows one optimistic Campaign-state protocol:

1. read and validate Campaign state and revision under the existing bounded lock;
2. release the lock;
3. invoke the probe once per Task and, when requested, query accepted Slurm allocations;
4. re-read the Campaign revision;
5. fail as stale if the Campaign changed;
6. return one immutable, revision-bound view.

The revision check proves Campaign consistency, not that external facts stop changing after the
query. Probe and scheduler observations are explicitly time-stamped evidence. `inspect()` has one
explicit scheduler switch: planning enables it; application-only readiness checks may disable it.
Disabled scheduler evidence is `UNOBSERVED`, not `UNKNOWN`, and quiescence is false.
Scheduler-dependent planning or retry rejects such a view. Probe results are not persisted. An
absent probe produces an explicitly unobserved result state, not `False`. A probe exception aborts
the complete inspection; no partial view is returned.

### Scheduler observation

Servatus queries only exact accepted JobReceipt identities already owned by the Campaign. It uses
bounded native `squeue` and `sacct` calls through the existing target and SSH lane. It does not scan
the account, infer jobs by user, or inspect unrelated output.

“Bounded” requires documented internal ceilings: a finite SSH/subprocess timeout, a maximum exact
job-ID batch per command, bounded argv construction, bounded stdout/stderr ingestion, and strict
line/field limits. Large Campaigns are queried in deterministic chunks. Timeout, output overflow,
or malformed output aborts the complete inspection.

Normalized allocation states are:

- `QUEUED`;
- `RUNNING`;
- `SUCCEEDED`;
- `FAILED`;
- `CANCELLED`;
- `UNKNOWN`.

The public observation retains the exact job/allocation identity, normalized state, raw Slurm
state, exit code when available, reason when available, start/end timestamps when available, and
observation time. Packed Task state is derived from its allocation; Servatus does not invent
per-step completion when only allocation evidence exists.

The normalization is exhaustive for supported Slurm base states:

- queued: `PENDING`, `CONFIGURING`, `REQUEUED`, `RESIZING`;
- running: `RUNNING`, `COMPLETING`, `SIGNALING`, `STAGE_OUT`, `SUSPENDED`, `STOPPED`;
- succeeded: `COMPLETED`;
- cancelled: `CANCELLED`, `PREEMPTED`, `REVOKED`;
- failed: `BOOT_FAIL`, `DEADLINE`, `FAILED`, `NODE_FAIL`, `OUT_OF_MEMORY`, `SPECIAL_EXIT`,
  `TIMEOUT`;
- unknown: no exact row or an unrecognized future state retained verbatim.

An exact current `squeue` row is authoritative for an active transition when `sacct` still exposes
an older state. Conflicting identities, multiple incompatible current rows, or incompatible
accounting rows raise. The observation preserves enough raw evidence to audit a legitimate
active-to-terminal transition.

A successful query with no exact matching row yields `UNKNOWN`. Command failure, malformed output,
conflicting identities, or partial untrustworthy results raises and aborts inspection. Observation
is synchronous, read-only, and bounded. There is no poll loop, background worker, cache, database,
or automatic mutation. Current scheduler observations are not durable state.

### Bounded diagnostic logs

Servatus owns the deterministic Slurm log layout, accepted Attempt identity, receipt, packed Task
order, remote host, and `log_root`. Clients must not reconstruct `%j.out`, `%j-<slot>.out`, packed
slots, or SSH transport. Log access therefore deepens the same Campaign authority through one
explicit operation, separate from ordinary inspection:

```python
@dataclass(frozen=True, slots=True)
class LogSnapshot:
    content: bytes = field(repr=False)
    truncated: bool
    observed_at: datetime


snapshot = campaign.read_log(
    allocation_id,
    task_key=task_key,  # omit for the allocation-level wrapper log
    max_bytes=65_536,
)
```

`allocation_id` must name one exact accepted Attempt owned by the Campaign. A supplied `task_key`
must belong to that Attempt; its immutable Attempt-local order determines the zero-based packed
slot. Omitting it selects the allocation wrapper log. Servatus privately derives the stored target,
receipt Job ID, and exact remote path. Future rendered filenames bind both immutable Servatus
allocation identity and the scheduler identity:

- allocation wrapper: `<allocation_id>-%j.out`;
- packed Task: `<allocation_id>-%j-<slot>.out`.

The allocation ID is derived from Campaign identity, exact plan facts, and allocation position, so
reused Slurm Job IDs and same-number jobs in different clusters cannot redirect one Attempt's log
address. Callers cannot supply a host, path, Job ID, shell fragment, or arbitrary remote command.

One call returns only the latest suffix, defaults to 64 KiB, and accepts an exact positive byte
limit no greater than 1 MiB. The implementation may read one extra byte solely to determine
`truncated`; it never returns more than `max_bytes`. Content is arbitrary bytes and may split lines
or encoded characters. stdout and stderr stay combined exactly as authored by the existing Slurm
renderer. Empty content is valid. A missing, unreadable, rejected, or unavailable log; a foreign,
unaccepted, unresolved, or not-submitted Attempt; an unrelated Task; timeout; nonzero remote
command; diagnostic stderr; or output overflow raises one redacted `ObservationError` without
partial content.

The remote operation uses the same private bounded OpenSSH implementation seam as scheduler
observation: fixed command shape, normalized environment, finite timeout, bounded argv and
stdout/stderr, and no public transport adapter. The path comes only from validated immutable
Campaign lineage. Log files may append concurrently; a snapshot claims only the returned bounded
suffix at `observed_at`, not a stable file or immutable provenance. The existing same-account remote
log-namespace trust model remains explicit; Servatus does not add a nonportable remote inode
transaction merely to defend against an out-of-contract hostile replacement. The implementation
reads and validates one immutable accepted Attempt, receipt, target, slot, and derived path under
the existing bounded Campaign-state lock, releases the lock, then performs SSH. It never holds the
Campaign lock across remote I/O. Accepted Attempt and lineage facts are immutable, so no revision
retry or post-read state check is needed. Append, seal, plan, and submit remain available while a
log read is blocked remotely.

Log bytes are sensitive, untrusted diagnostic data and may contain terminal control sequences.
They are never placed in Campaign state, a Campaign view,
plan documents, operational records, exceptions, `repr`, or automatic CLI JSON. Log presence,
absence, silence, text, or failure never affects result state, scheduler state, readiness,
quiescence, planning, retry, reconciliation, or scientific validity. There is no decoder, line
model, parser registry, progress/epoch concept, search, offset/range protocol, follow mode, stream,
poller, cache, persistence, multi-log read, or arbitrary-path escape hatch.

README and SECURITY must document arbitrary binary content, the owner-account remote namespace
trust model, lack of remote inode/authenticity guarantees after Slurm writes the path, and the need
to redirect CLI output to a file or safe viewer. `ObservationError` messages, explicit notes, and
visible chained causes must never contain log bytes, derived remote paths, host secrets, remote
stderr, or commands; unsafe transport and remote exceptions are raised through one sanitized cause
or suppressed chain.

### Campaign view and readiness

One immutable Campaign view binds:

- Campaign identity and revision;
- roster phase and Task order;
- acceptance and ambiguity history;
- current allocation observations;
- current application result observations;
- derived per-Task eligibility;
- `results_ready` and `quiescent`.

Definitions:

- `results_ready`: the roster is sealed and every Task has a valid immutable application result;
- `quiescent`: scheduler evidence was requested and every accepted attempt is proven terminal, with
  no queued, running, unknown, unobserved, or acceptance-ambiguous attempt.

Result readiness and execution quiescence are deliberately independent. A valid canonical result
dominates stale or expired scheduler visibility for scientific publication. The view is evidence,
not a durable phase. Any Campaign mutation invalidates it. It contains no Task stdin or application
result content in ordinary representations.

### Result-aware planning

Planning verifies that its view belongs to the exact current Campaign revision. A Campaign with any
accepted attempt requires scheduler-observed evidence for planning or retry. Task selection is:

| Result | Acceptance/execution | Default action | Explicit retry |
| --- | --- | --- | --- |
| valid | any | exclude | reject |
| missing or unobserved | never accepted | select | unnecessary |
| missing or unobserved | queued/running | withhold | reject |
| missing or unobserved | succeeded/failed/cancelled | withhold | select |
| missing or unobserved | unknown | withhold | select as recorded operator override |
| any | acceptance ambiguous | withhold affected Tasks | reject until resolved |
| invalid | any | inspection raises | no plan |

No Task is retried automatically. No timeout, failure state, or missing result authorizes replay.
The operator names exact Task keys. Retrying unknown work is an explicit duplicate-execution risk,
not a claim that the old attempt stopped. Prior attempts remain in history. Unaffected Tasks may
proceed while a different allocation is ambiguous, but every Task in the ambiguous allocation
remains blocked.

The plan freezes its Campaign revision, roster digest, selected/excluded Task keys, attempt
projection, observation timestamps, view digest, explicit retry keys, selected nonbinding profile
label, resolved target/resources, and exact allocations. A canonical plan document restores that
immutable decision without serializing or rerunning a probe. Restoration validates syntax, Campaign
identity/revision, roster/attempt lineage, and canonical plan bytes.

External evidence is time-specific. Before each allocation's mutating `sbatch` call, `submit`:

1. reads and validates local Campaign state at the expected revision;
2. rechecks that allocation's selected Tasks through the caller probe when result-aware;
3. performs the bounded read-only scheduler query for relevant accepted attempts;
4. rereads Campaign state and revision;
5. compares eligibility with the frozen plan;
6. durably records intent;
7. invokes `sbatch`, then records its receipt or ambiguity.

The sequence repeats for each allocation, carrying forward revisions created by earlier allocations
in the same submit call. Excluded valid results need no second probe because the caller contract
makes `True` immutable; scheduler-based exclusions are covered by the scheduler refresh. A
scheduler-only plan needs no probe. This narrows the ordinary race without gratuitously reloading
completed results. The application contract still forbids an independent non-Campaign producer
from publishing a selected Task's result concurrently with submission.

Packing, resource arithmetic, target ceilings, allocation submission caps, validation, submission,
reconciliation, and explicit acceptance resolution retain their current semantics.

### Redacted provenance

Servatus exposes one canonical, immutable operational record derived from a revision-bound Campaign
view. It contains only generic execution evidence:

- record schema and observation time;
- Campaign identity, revision, sealed state, and roster digest;
- ordered Task keys;
- each attempt's selected nonbinding profile label;
- target/resource lineage digests and allocation shapes;
- attempt, retry, plan, and script digests;
- Slurm job/cluster identities and normalized current scheduler observations;
- ambiguity and explicit resolution history.

It excludes:

- Task stdin and request bodies;
- complete scripts and application arguments;
- credentials and environment variables;
- checkpoints, metrics, models, predictions, and result contents;
- raw scheduler reasons and node names;
- sensitive target paths, account, QoS, and host values.

This record is redacted but not anonymous or secret-safe: Task keys and job identities may identify
work. The owner-only Campaign state also retains Task args/stdin for restart and retry. The API
returns canonical JSON bytes; the caller chooses private retention and whether a separately reviewed
projection is safe to publish. Servatus never publishes the record automatically and does not mutate
Campaign state or application outputs when producing it. Campaign-state deletion is always an
explicit caller retention decision after retries and audit evidence are no longer needed.

## KAIROS clean-break adapter

### Client shape

The final client has four obvious pieces:

```text
typed KAIROS request -> ExperimentTask envelope -> Servatus Task
KAIROS result probe(Task) -> bool or domain-validation exception
Campaign inspect/plan/submit
KAIROS scientific close(Campaign.tasks, result evidence) -> canonical manifest
```

Keep these as small ordinary functions in existing owning modules. Do not introduce an adapters
package, client object, execution service, repository, unit-of-work wrapper, or KAIROS mirror of
`CampaignView`. KAIROS tests capture public calls and scientific bytes; they do not fake Servatus's
state machine.

### Sole pre-publication roster

The Servatus Task roster becomes the sole active execution roster. KAIROS deletes:

- hidden authored experiment bundle directories;
- `cells.tsv`;
- `requests/*.json`;
- `open_bundle()` and `bundle_path()`;
- tune/train/evaluate cell writers and readers;
- active-bundle versus canonical-manifest fallback;
- bundle retirement during experiment publication.

KAIROS converges remote work on one strict KAIROS-owned execution envelope. It contains the typed
candidate or workflow request and an optional experiment cell. Direct one-task commands use no
cell; experiment tasks use a nonempty cell. The envelope is serialized into opaque `Task.stdin`.
KAIROS workers deserialize it. Servatus neither imports nor understands it.

The envelope contains existing KAIROS Study, Artifact, Evaluation, and Dataset associations. It does
not replace them with Servatus IDs. One canonical envelope format replaces the old raw/direct versus
experiment payload split; no compatibility parser remains.

### Campaign address

KAIROS stores durable experiment Campaigns outside the canonical experiment object and outside any
retired authored tree. The default address is:

```text
<storage_root>/experiments/.servatus/<kind>/<experiment_id>/
```

KAIROS owns that address mapping and creates only the exact owner-only parent required by Campaign.
Campaign state survives manifest publication as execution provenance. Canonical experiment objects
remain:

```text
<storage_root>/experiments/<kind>/<experiment_id>/manifest.json
```

No downstream scientific loader reads Campaign private state. After publication, the canonical
manifest remains the only scientific experiment authority.

### Authoring

Fixed experiments construct their complete ordered Tasks, open the Campaign, and seal it during
prepare. Appendable HPO opens its first ordered prefix, launches while open, appends exact suffixes,
and seals only when authoring closes.

KAIROS reconstructs cells and domain record IDs from its own envelopes in `Campaign.tasks`. It
retains scientific rules that Servatus cannot know:

- a Tune cell may map to multiple candidate Tasks but one Study UUID;
- all candidate Tasks for one Study must carry one exact TuneRequest;
- appended HPO cells must be new and scientifically unique;
- Train and Evaluate cells map one-to-one to Artifact or Evaluation UUIDs;
- manifest cell order and record type depend on experiment kind.

### KAIROS result probe

KAIROS supplies one adapter over the existing canonical loaders:

- candidate Task: validate the exact candidate result needed by Study assembly, including the
  KAIROS metric reduction required to verify stored objective equality, or the canonical Study when
  already finalized;
- Train Task: load the exact Artifact and require its embedded TrainRequest association;
- Evaluate Task: load the exact EvaluateRequest, validate canonical observations, and require the
  embedded association.

Missing paths return `False`. Present-but-invalid or mismatched objects raise. The probe may compute
the domain reduction required for validation. It does not produce reports, build manifests, inspect
scheduler state, or publish anything.

### Closing an experiment

KAIROS owns close:

1. seal the Campaign if its scientific authoring phase is complete;
2. inspect it with the KAIROS result probe and scheduler observation explicitly disabled;
3. require `results_ready`; report quiescence only from a separate scheduler-observed view and do not
   make ephemeral Slurm availability an authority over valid immutable scientific objects;
4. assemble any KAIROS-owned Study fan-in still required;
5. validate every cell-to-record association;
6. atomically publish the manifest-only experiment object;
7. leave Campaign state intact as execution history.

Scientific publication failure leaves the Campaign and canonical component objects available for a
later explicit close. Servatus has no `Campaign.close()` and no finalizer callback.

## Explicitly rejected designs

These are non-goals unless a later independently approved ADR supplies a new demonstrated use case.

### Workflow/DAG engine

No generic nodes, edges, dependencies, fan-in, fan-out, conditional branches, map/reduce graph,
dynamic scheduler, or Nextflow-like language. KAIROS already authors finite opaque Tasks; adding a
graph would duplicate its scientific topology and enlarge the public model.

### `run()` or `dispatch()` facade

No one-shot `open -> inspect -> plan -> submit` helper. Those boundaries have different failure and
review meanings. Hiding them would make plans, optional validation, explicit retry, and ambiguous
acceptance harder to reason about while deleting little client code.

### Completion or finalizer plugins

No serialized callback, plugin registry, validator class hierarchy, application enum, or generic
aggregate builder. The in-memory probe answers only whether one opaque Task's result is valid.
KAIROS retains Study assembly and manifest publication.

### `Campaign.close()` or durable `FINALIZED`

A Campaign can derive readiness but cannot know whether an application published the correct
aggregate. Recording finalization would create a second authority and a crash gap between application
publication and Campaign mutation.

### Generic Bundle

No Servatus bundle abstraction. Cells, Study grouping, HPO extension, request schemas, and manifest
meaning are KAIROS concepts. The generic roster already belongs to Campaign; generic atomic output
already belongs to publication.

### Re-home authoring in Workspace

Workspace remains resumable private worker state. It does not become a long-lived experiment-author
lease. That would couple authoring, launching, monitoring, and closure to hidden hashed paths and a
second lifecycle lock.

### Workspace child registry or group

No child registry, expected-child count, readiness callback, or generic fan-in. KAIROS knows which
Study candidates are required. Existing `Workspace.child()` remains sufficient for concurrent
resumable workers.

### Scheduler policy

No automatic retry, cancellation, requeue, timeout policy, queue/QoS throttling, priority, GPU model
selection, or one-free-slot policy. Those are operator or application deployment choices.

### Backend/plugin framework

No public scheduler protocol, executor registry, local backend, Kubernetes adapter, daemon, event
bus, database, or event-sourcing framework. Native Slurm remains the only proven lane. A second real
production backend may justify a later extraction.

### Generic ML semantics

No Study, trial, fold, seed, checkpoint, Artifact, Evaluation, metric, model, dataset, benchmark,
experiment tracker, or result schema in Servatus. Projects encode those meanings in opaque Tasks and
their probes.

### Universal IDs or Task metadata

No generic UUID factory, entity base class, Task annotation mapping, or result registry. Stable
caller keys and opaque stdin already provide the correct seam.

### Compatibility and migration

No schema-3 Campaign reader in Servatus `0.7`, no KAIROS `cells.tsv` reader, no request-file fallback,
no bundle migration, and no dual remote-payload parser. Existing `0.6` Campaigns finish with `0.6`.
New KAIROS Campaigns start cleanly with `0.7`.

## ADR effects

Servatus implementation must add one ADR for the Campaign-engine decision and update the active
context and README. The new ADR:

- narrows ADR 0001: application schemas remain opaque, but an ephemeral boolean result probe is now
  accepted as a generic lifecycle seam;
- supersedes ADR 0003's acceptance-only public status with sealed roster, current execution
  observation, result-aware planning, and derived readiness;
- retains ADR 0003's native Slurm, intent-before-contact, explicit ambiguity, packing, resource, and
  no-plugin decisions;
- leaves ADR 0002 publication and ADR 0004 Workspace-child behavior unchanged.

KAIROS adoption must update its context, main manual, ADR 0006, and ADR 0008. ADR 0006 retains direct
durable object authority but replaces hidden `cells.tsv` authoring with Campaign Tasks before
manifest publication. ADR 0008 retains opaque Servatus mechanics but adds observation/readiness and
removes bundle retirement. ADR 0009 remains unchanged.

## Implementation protocol

Each slice uses the same loop:

1. the orchestrator records the exact accepted baseline and creates one run-owned worktree;
2. a fresh implementer reads repository guidance, relevant context/ADRs, this complete ledger, and
   the implementation skill;
3. the implementer proves required behavior red, makes the smallest coherent change, runs gates,
   commits once, and reports exact range and measured numstat;
4. a distinct reviewer reads the same authority plus the code-review skill and reviews the fixed
   range on Standards and Spec axes;
5. any finding returns to the same implementer as a separate correction commit;
6. the same reviewer rereviews the correction range;
7. the next slice starts only after Standards 0 / Spec 0 and a clean tree.

The orchestrator does not edit product code. It owns ledger/status corrections and coordination.
Workers do not push, tag, publish, contact Slurm, build KAIROS images, mutate outputs, or clean
evidence unless a later external gate explicitly authorizes the exact action.

Tests stay at public observable seams. Do not duplicate private parser, Slurm renderer, POSIX race,
or dependency tests in KAIROS. New tests must replace obsolete tests where the old concept
disappears. Green tests do not excuse dead helpers, duplicate validators, speculative types, or a
larger client surface.

## Slice S1 — Campaign roster and attempt state

Baseline: exact Servatus `281c381548489c1dcf7a6ca8d045908d0b50ba3f` plus this reviewed
ledger-only commit.

Scope:

- introduce Campaign schema 4 as a clean break;
- add durable `OPEN`/`SEALED` roster phase;
- add `Campaign.tasks` and idempotent `Campaign.seal()`;
- preserve exact ordered suffix growth only while open;
- invalidate old plans on append or seal;
- consolidate attempt/receipt/resolution state only when it removes duplicate authority;
- preserve intent-before-SSH, receipt-after-acceptance, reconciliation, explicit resolution, target
  and resource lineage, revision binding, balanced packing, and submission cap;
- update state ingress validation for the new exact schema;
- update `docs/CONTEXT.md`, README, ADR 0003, and add the new Campaign-engine ADR section owned by
  this slice.

Non-goals:

- no scheduler status query;
- no result probe or Campaign view;
- no CLI redesign beyond seal and exact schema fallout;
- no Workspace/publication change;
- no compatibility parser;
- no KAIROS change.

Required public tests:

- exact roster reopen and open-prefix append preserve existing history;
- seal is durable and idempotent;
- sealed append, removal, reorder, or Task byte change fails;
- append and seal stale existing plans;
- execution remains valid while open;
- every intent is synced before SSH;
- accepted, ambiguous, reconciled, explicitly not-submitted, and retry attempts retain exact lineage;
- malformed phase, revision, Task, attempt, receipt, resolution, and numeric JSON values fail at
  bounded ingress;
- schema 3 is rejected without migration;
- public plan/validate/submit/reconcile behavior otherwise remains byte- and semantics-equivalent.

Expected outcome:

- one durable owner for roster phase and attempt history;
- no second public execution abstraction;
- no more source concepts than the behavior needs;
- current safety and resource behavior unchanged;
- exact product and test numstat recorded, without claiming a forced net deletion.

Gate: independent fixed-range review, full Servatus local gates, build/archive inspection, and fresh
installed-wheel API/CLI smoke. No live Slurm gate.

## Slice S2 — Campaign observation engine

Baseline: exact accepted S1 head.

Scope:

- add one bounded `_slurm` query over exact accepted JobReceipt identities;
- enforce finite SSH timeout, deterministic job-ID chunks, and argv/output/line/field limits;
- normalize every supported Slurm state to queued/running/succeeded/failed/cancelled/unknown;
- introduce the only immutable `CampaignView` through `Campaign.inspect()`;
- retain every attempt observation and project current Task execution from its latest accepted
  attempt, with unresolved acceptance dominant;
- add the optional plain synchronous result probe to the same `inspect()` path;
- add one explicit bounded `Campaign.read_log()` diagnostic snapshot over exact accepted Attempt
  and optional packed Task identity, keeping raw bytes outside `CampaignView`;
- change future Slurm allocation and packed-Task log templates to include immutable
  `allocation_id` plus `%j`, so Job-ID reuse cannot redirect an Attempt's diagnostic address;
- run probes, scheduler queries, and remote log I/O outside the Campaign lock; inspection rejects
  revision changes, while log reads snapshot immutable accepted Attempt facts before releasing it;
- derive result readiness and execution quiescence independently;
- keep observations transient, time-stamped, read-only, and redacted;
- update context, README, SECURITY, ADR 0003's rendered log identity, and the Campaign-engine ADR.

Non-goals:

- no account-wide scan, polling, watcher, cache, or persistence;
- no invented task-step state from allocation-only evidence;
- no callback registry, serializer, async API, schema knowledge, finalizer, retry, or queue policy;
- no log parsing, progress inference, offsets, follow/streaming, polling, cache, arbitrary remote
  path, public SSH adapter, or log-derived authority;
- no public scheduler plugin, `observe()` sibling, or second view type.

Required public tests:

- exact receipt identities generate bounded chunked `squeue`/`sacct` requests;
- timeouts and every input/output bound fail closed;
- supported raw states and legitimate active-to-terminal transitions normalize exactly;
- malformed, partial, unrelated, conflicting, or unavailable evidence aborts the whole inspection;
- result-only inspection makes no scheduler call, marks scheduler evidence unobserved, cannot be
  used for scheduler-dependent planning, and still derives sealed result readiness;
- absent exact rows become unknown;
- packed Tasks share allocation evidence;
- multiple attempts remain visible and the latest accepted attempt owns current projection;
- quiescence requires every accepted attempt terminal;
- probe runs once per Task, outside the lock, and prior calls are not claimed rolled back;
- valid/missing/unobserved are distinct; invalid raises without returning a partial view;
- Campaign mutation during inspection makes the view stale;
- sealed result readiness and execution quiescence follow independent definitions;
- valid immutable results remain result-ready despite unknown scheduler visibility;
- view output contains no stdin, args, target secrets, or result content;
- inspection never mutates Campaign bytes or revision.
- accepted allocation and packed-Task log reads derive exact private paths from durable receipt and
  Attempt order; foreign/unaccepted identity and unrelated Task input fail before SSH;
- allocation and Task log filenames contain exact `allocation_id` plus Job ID; retries and equal Job
  IDs across distinct Attempts/clusters cannot alias one another;
- bounded binary tail snapshots prove exact limit/truncation behavior for empty, small, exact-limit,
  and over-limit content without decoding;
- timeout, stderr, nonzero status, output overflow, and unavailable log fail without partial bytes;
- a blocked remote read does not hold the Campaign lock or prevent append/seal/plan/submit;
- log content is absent from view/state/plan/record/exception/`repr`; the complete visible
  `ObservationError` message, notes, cause chain, and context chain also omit derived path, command,
  host, remote stderr, and content;
- the fresh installed-wheel smoke imports public `ObservationError`, `LogSnapshot`, and
  `Campaign.read_log`, and log outcomes never change readiness, quiescence, planning, retry, or
  Campaign bytes/revision.

Genericity gate: include one plain-file and one structured-JSON probe in tests without adding those
semantics to Servatus. The public interface contains no KAIROS or ML domain term. KAIROS K2 later
must prove real client deletion; otherwise integration stops for redesign.

Expected outcome:

- one native, synchronous observation seam joining generic scheduler and opaque result facts;
- no daemon, plugin framework, second status path, or scientific coupling.

Gate: independent review plus full Servatus gates. No live Slurm claim is made here.

## Slice S3 — Evidence-aware planning and CLI

Baseline: exact accepted S2 head.

Scope:

- add the immutable public `Profile` and strict one-pass loading of one repository-local
  `SERVATUS.toml` containing one or more complete named target/resource profiles;
- require the exact root shape, validate every declared profile including unselected ones, and let
  library callers supply the explicit file path while both CLIs read only
  `Path.cwd() / "SERVATUS.toml"`;
- let an explicit name override the optional top-level `default_profile`, and fail directly when no
  selection or selected profile exists;
- make `Campaign.plan()` consume one Profile, bind compatibility to exact resolved execution values,
  and retain its label as nonbinding plan/attempt provenance;
- remove paired target/resource TOML loading paths and direct planning parameters where the Profile
  replaces them without weakening programmatic construction or durable validation;
- make planning consume one exact current Campaign view and implement the approved selection matrix;
- preserve unaffected planning while blocking only ambiguous Tasks;
- allow unknown accepted work only through an explicit recorded duplicate-risk override;
- freeze revision, roster, attempt projection, observations, selection, retry keys, target/resources,
  and allocations into one immutable plan;
- make the public plan codec restore that decision without serializing or rerunning a probe;
- before each allocation's `sbatch`, refresh local state, selected-Task probe evidence, and read-only
  scheduler evidence in the exact documented order;
- abort changed eligibility before each mutating `sbatch` call;
- add CLI `seal` and scheduler-only `inspect`, route plan persistence through the public codec, and
  make CLI planning load exact `SERVATUS.toml` plus optional `--profile`;
- add thin `servatus log CAMPAIGN ALLOCATION_ID [--task TASK_KEY] [--bytes N]`, writing exact raw
  snapshot bytes to stdout and delegating all selection, bounds, and errors to `Campaign.read_log()`;
- remove obsolete acceptance-only status and manual completed-key paths;
- keep validate/submit/reconcile/resolve thin over public Campaign methods;
- update README, context, ADR 0003, Campaign-engine ADR, and CLI help.
- make CLI help and README/SECURITY identify `log` output as sensitive untrusted binary data that
  may contain terminal control sequences and recommend redirecting it to a file or safe viewer;

Non-goals:

- no automatic retry, cancellation, policy, `run()` facade, probe loader, result-aware CLI, generic
  finalizer, provenance record, or Workspace/publication change.
- no CLI log parser, decoder, metadata wrapper, arbitrary path, multi-log command, follow mode, or
  second implementation of Campaign log selection;
- no global profile store, discovery search, environment fallback, inheritance, composition,
  credentials, profile-management CLI, or compatibility for paired target/resource files.

Required public tests:

- exact explicit and declared-default profile selection;
- explicit selection overrides the declared default;
- one or multiple complete profiles load exact typed target/resource values;
- absent document, absent selection, absent selected/default label, malformed profile, and unknown
  raw keys fail once at configuration ingress;
- same-value aliases remain compatible, the selected label remains visible provenance, and changed
  resolved values are rejected on an existing Campaign;
- malformed unselected profiles invalidate the whole document;
- CLI invocation outside the directory containing the exact `SERVATUS.toml` fails without search;
- no filesystem search, environment lookup, management command, or second target/resource planning
  path remains;
- every row of the result/acceptance/execution selection matrix;
- explicit retry and unknown override retain exact attempt history and warnings;
- any older or newer queued/running attempt blocks retry, and any unknown accepted attempt requires
  the recorded duplicate-risk override;
- unaffected Tasks plan while different keys are ambiguous;
- stale/foreign views and plans fail before external contact;
- plan round-trip preserves exact selection/allocation bytes without invoking a probe;
- plan round-trip preserves the selected nonbinding profile label, and each submitted attempt
  retains the label from its exact restored plan;
- submit refreshes scheduler evidence and selected-Task result eligibility before each allocation's
  mutating `sbatch` call;
- changed result or execution eligibility aborts;
- scheduler-only plans need no probe;
- CLI output and owner-only no-clobber plan files remain safe;
- CLI `log` addresses one exact accepted allocation and optional member Task, preserves arbitrary
  bytes without adding a newline, honors the public byte bound, and reports unavailable logs only
  through the existing error path;
- CLI help contains the raw-byte/redirection warning; errors and chained causes contain no log
  content, remote path, command, host secret, or remote stderr; this asserts CLI translation while
  S2 remains the authority for the complete Python exception-chain contract;
- no parallel status, completed-set, or plan implementation remains.

Expected outcome:

- one common Python path from roster through evidence-aware planning;
- one visible project-owned configuration document and one explicit/default named profile selection;
- immutable cross-process plans without false external-snapshot claims;
- a thin execution-only CLI and explicit retry as the sole replay authority.

Gate: independent review plus full Servatus gates.

## Slice S4 — Operational record, consolidation, and `0.7.0` candidate

Baseline: exact accepted S3 head.

Scope:

- add canonical redacted operational-record bytes from one Campaign view;
- include the selected nonbinding profile label while keeping resolved sensitive target values
  redacted;
- exclude payloads, scripts, Task digests, raw reasons, nodes, target secrets, and environment;
- document that Task keys/job IDs remain identifying and Campaign state remains private;
- audit source/tests/docs for duplicate decoding, dead paths, repeated fields, and premature types;
- verify Profile is the only repository-config loading concept and target/resource values have one
  parser and one durable owner;
- split private `_campaign_store.py` only if it materially improves locality and readability;
- keep `_slurm.py`, `_workspace.py`, `_posix.py`, and `cli.py` focused on existing owners;
- keep the public facade minimal, typed, and at zero runtime dependencies;
- set package/lock metadata to `0.7.0`;
- finish README, SECURITY, context, ADR index, and one canonical example;
- record exact public API changes and measured source/test numstat.

Non-goals:

- no automatic record publication or anonymity claim;
- no public domain/workflow/backend/adapter/provenance package;
- no module split solely to shorten files, compatibility alias, or forced net-negative LOC target;
- no removal of raw-ingress, persisted-state, native-response, no-clobber, identity, or demonstrated
  race checks.

Required audit and tests:

- operational record is canonical, revision-bound, and changes only with included facts;
- excluded sensitive values are absent from its fields;
- every public type passes a deletion test;
- Profile removes paired caller configuration and does not create a store/manager abstraction;
- every durable field has one owner and consumer need;
- validations guard only raw ingress, persisted state, external responses, or demonstrated races;
- typed internal paths do not revalidate impossible states;
- tests assert public behavior except bounded corruption ingress;
- Vulture findings receive manual dynamic/framework review;
- fresh build metadata proves version `0.7.0` and zero runtime dependencies.

Expected outcome:

- one coherent `0.7.0` candidate with a deep Campaign and unchanged publication safety;
- measured complexity reported honestly and no KAIROS-shaped public architecture.

Gate: independent review, full package gates, wheel/sdist inspection, fresh no-cache installed-wheel
smoke, and clean worktree. No push, tag, release, PyPI, or live Slurm contact.

## External gate E0 — isolated live Servatus observation

Blocked until S4 and the disposable E2 client proof are GREEN and the user authorizes isolated
Slurm contact. This gate precedes public release because scheduler observation must not first meet a
real controller after publication. It runs against the exact final S4 head after every correction
caused by E2; any later product correction invalidates E0 and requires a rerun.

Use the exact installed wheel built from the final accepted S4 head for one run-owned CPU-only
Campaign and allocation on an administrator-approved CPU partition. It
must not use, inspect, or compete with the protected K-study beyond ordinary shared cluster policy.
Evidence must prove:

- exact accepted receipt identity;
- bounded query commands and response sizes;
- at least queued or running observation when placement permits;
- terminal observation and raw/normalized state agreement;
- one exact Task log containing a known synthetic stdout/stderr sentinel and one allocation-level
  log snapshot read through the public bounded interface;
- exact packed-slot selection where the isolated allocation contains multiple synthetic Tasks;
- redirect the installed `servatus log` CLI stdout to a run-owned file, byte-compare the sentinel
  and no-added-newline behavior, and never render arbitrary log bytes directly to the terminal;
- missing/expired-row handling through a synthetic unit test remains authoritative if the live site
  retains the accounting row;
- Campaign state is unchanged by observation;
- Campaign state is unchanged by diagnostic log reads, and log content changes no readiness,
  quiescence, planning, or retry fact;
- no unrelated job is returned;
- no automatic retry or mutation occurs.

Preserve logs and state as evidence. Failure blocks release and triggers diagnosis; it never changes
production KAIROS state.

## Gate E1 — active K-study closure

KAIROS slices cannot start until the Runner task reports the protected K-study complete and closed.
This program consumes Runner-attested evidence only; it does not independently open the Campaign,
query its jobs, read its outputs, or inspect its heartbeat. The Runner handoff must establish:

- every intended Task has valid canonical KAIROS evidence or an explicitly documented terminal
  scientific disposition;
- no job or heartbeat still needs the Campaign's schema-3 state;
- no further retry, resolution, or authoring is pending;
- the final experiment manifest and downstream scientific handoff are owned by the Runner;
- held-out and inference work remains outside this program unless separately authorized.

Do not migrate or delete the schema-3 Campaign. Preserve it with its exact code/image evidence.

## Gate E2 — disposable KAIROS client proof

Blocked until S4 and E1 are GREEN. Before Servatus `0.7.0` becomes a stable public API, apply the
planned KAIROS K1/K2 shape in a disposable run-owned worktree against the exact local S4 wheel. This
is a deletion/ergonomics proof, not an accepted KAIROS implementation slice and never contacts
Slurm, outputs, remotes, or production Campaigns.

Required evidence:

- one envelope, one probe, public Campaign calls, and direct scientific close are sufficient;
- no adapter class, service, repository, wrapper facade, duplicate view/status, or private Servatus
  import is needed;
- hidden bundles, TSV/request files, active/canonical fallback, and client completed-set joins can be
  deleted without changing canonical scientific paths or bytes;
- representative prepare, append, launch, retry forwarding, result-only close, and manifest tests
  pass against the local wheel;
- measured KAIROS product/test numstat shows material conceptual deletion;
- every awkward or KAIROS-shaped Servatus API pressure is returned to S1-S4 as a reviewed correction
  before release;
- the disposable worktree is retained through S4 correction review, then removed without merging.

Failure blocks stable release. It does not justify a KAIROS shim.

## External gate E3 — Servatus `0.7.0` release

Blocked until E0, E1, and E2 are GREEN and the user separately authorizes external publication.

Required sequence:

1. pin the exact accepted S4 head and clean status;
2. push `main` without rewriting history;
3. run Linux and macOS CI;
4. create annotated tag `v0.7.0` at the exact accepted head;
5. verify tag CI and GitHub Release;
6. publish through the repository's trusted PyPI workflow;
7. record public wheel and sdist SHA-256 values and provenance;
8. fresh no-cache install from the public index;
9. verify metadata version, zero runtime dependencies, exported `ObservationError` and
   `LogSnapshot`, public `Campaign.read_log`, CLI `log` command presence, seal/view/planning
   behavior, and publication smoke without new scheduler or log contact;
10. preserve `0.6.0` and its tag/package plus the protected schema-3 Campaign evidence.

Failure at any step stops before accepted KAIROS implementation. Do not delete local build/review
evidence until the public package is verified.

## Slice K1 — KAIROS execution adoption

Baseline formula: the exact then-current published KAIROS `main` after E1, plus the exact publicly
verified Servatus `0.7.0` head and hashes after E3. Re-pin rather than assuming the planning
baseline remains current.

Scope:

- pin Servatus `0.7.0` in root and mobile manifests/locks;
- replace repository-root `REMOTE.toml` plus `RESOURCES.toml` with one version-controlled
  `SERVATUS.toml` containing exactly the unchanged complete `KAIROS` profile and
  `default_profile = "KAIROS"`;
- load that document once per command at the KAIROS composition root, accept optional
  `--profile NAME`, and pass one public Profile through direct and experiment planning;
- introduce one strict KAIROS-owned execution envelope for direct and experiment Tasks;
- use one hidden remote worker entry point if that deletes the candidate/workflow payload split;
- route existing direct and bundle-authored launch paths through that one envelope;
- implement one KAIROS result probe over canonical domain loaders;
- replace candidate/workflow completed-set scans with `Campaign.inspect()`;
- make direct and experiment planning consume the revision-bound view and exact retry keys;
- pass the same probe to `submit` for immediate eligibility revalidation;
- move experiment Campaign state outside the authored bundle to the final private address;
- retain current bundle authoring and close only for this accepted intermediate slice;
- keep receipt presentation and current tasks-per-allocation option spelling;
- preserve exact Task keys, request semantics, Study grouping, worker calls, and outputs.

Non-goals:

- no sole-roster or scientific-close cutover yet;
- no cell/request/bundle deletion;
- no canonical output/schema/path change;
- no Dataset, model, metric, feature, CUDA, resolved resource, resolved target, image, or queue-policy
  value change;
- no global profile state, config search, environment fallback, repository-name inference, profile
  management command, or paired-file compatibility;
- no schema-3 reader or bundle migration;
- no new adapter package unless multiple real owners require it.

Required tests:

- default KAIROS profile and explicit override load once and forward one exact Profile;
- missing `SERVATUS.toml`, missing default/explicit profile, and malformed selected profile fail at
  the one raw configuration boundary;
- exact current host, image, roots, partitions, ceilings, 24 CPUs, 65536 MiB, one GPU, and three-day
  request survive the two-file-to-one-file structural change;
- `REMOTE.toml`, `RESOURCES.toml`, separate loader calls, and direct target/resource planning args
  are absent from active source/tests/docs;
- exact direct and experiment Task key/argv/stdin bytes through the one envelope;
- candidate grouping and workflow one-to-one mappings;
- public workers hydrate both direct and experiment envelopes into the exact existing KAIROS calls;
- malformed envelopes fail at the KAIROS raw-input boundary;
- missing canonical result returns false; present invalid/mismatched result raises;
- valid Study candidate, Artifact, and Evaluation paths return true through public loaders;
- inspection delegates scheduler state and retry eligibility to Servatus rather than rebuilding a
  KAIROS state machine;
- restored plans cannot bypass a fresh KAIROS probe before mutating `sbatch`;
- experiment publication retires only its current bundle and leaves Campaign history intact;
- no Servatus schema knowledge or private import enters KAIROS;
- current bundle authoring, canonical loaders, and scientific rosters remain exact.

Expected outcome:

- one readable project-local `SERVATUS.toml` with a declared KAIROS default and explicit profile
  override, replacing two execution-config files without hidden state;
- one canonical KAIROS remote payload;
- one thin public Campaign/probe launch path with no caller-built completed sets;
- durable Campaign history survives current bundle closure;
- a fully usable tree and stable envelope for the sole-roster cutover.

Gate: independent KAIROS fixed-range review; root, mobile, and App gates; installed Servatus public
API smoke; exact protected-path diff; no image, scheduler, output, or remote action.

## Slice K2 — Sole Campaign roster and lean scientific close

Baseline: exact accepted K1 head.

Scope:

- add the exact experiment Campaign address mapper to all prepare/extend/launch/close paths;
- author fixed experiments directly as exact ordered Tasks and seal during prepare;
- author HPO as an open exact prefix, append exact suffixes, and seal only when scientific authoring
  closes;
- reconstruct cells and KAIROS record UUIDs from Campaign Task envelopes;
- delete hidden authored bundle paths, `cells.tsv`, request files, writers/readers, active/canonical
  fallback, and bundle retirement;
- require `results_ready`, not execution quiescence, before scientific close;
- retain KAIROS Study assembly, domain reductions, association validation, and manifest-only
  publication;
- retain owner-only Campaign state after publication;
- make downstream experiment consumers read canonical manifests only;
- preserve every canonical experiment path and manifest byte contract;
- audit `src/`, `experiments/`, and `tests/` for dead bundle, completed-set, duplicate roster,
  duplicate request-loading, private Servatus, and transition-only machinery;
- delete obsolete tests rather than preserve the old architecture as fixtures;
- update `docs/CONTEXT.md`, `docs/KAIROS.md`, ADR 0006, ADR 0008, and this ledger;
- preserve ADR 0009 and Blockweaver `0.3.4` authority;
- record exact product/test numstat and explain every retained execution guard.

Non-goals:

- no Servatus ID for KAIROS objects;
- no scheduler policy, automatic retry, generic finalizer, or Campaign close;
- no execution record inside canonical scientific objects;
- no Domain/Dataset/model/metric/CUDA/resource/target/image change;
- no execution-profile value or selection change after K1;
- no legacy bundle or schema-3 reader;
- no held-out or inference campaign launch.

Required tests:

- fixed prepare creates one sealed Campaign with exact order;
- HPO append preserves exact prefix and rejects duplicate scientific cells;
- Campaign Tasks reconstruct exact ordered cells and Study/Artifact/Evaluation associations;
- Tune cell fan-out still maps multiple candidate Tasks to one Study UUID;
- close validates candidate metrics/objective equality and all canonical associations;
- close succeeds with valid immutable results even if scheduler evidence is unknown, while reporting
  quiescence separately;
- publication failure preserves Campaign state and canonical component objects;
- downstream authors reject absent manifests instead of reading Campaign private state;
- no `cells.tsv`, request file, authored bundle, retire call, active-roster fallback, or private
  Servatus import remains;
- canonical paths, manifest bytes, and domain object bytes remain unchanged.

Acceptance standard:

- target KAIROS deletion from the design audit is roughly 80-130 production lines and 140-220 test
  lines, but estimates are not quotas;
- the real gate is deletion of the duplicate roster concept and no replacement mini-framework;
- if the final adapter does not materially simplify KAIROS, stop and redesign before integration;
- raw request/config ingress, scientific association/schema/order checks, canonical publication, and
  one-GPU policy remain.

Required gates:

- root lock check and frozen dry sync;
- root pytest;
- Ruff check and format check;
- repository-configured Pyright;
- Vulture plus manual finding audit;
- mobile lock/frozen sync and pytest, including host XNNPACK coverage;
- App locked install, test, typecheck, strict unused-source check where configured, and dry install;
- installed Servatus/KAIROS CLI and API smokes;
- no legacy roster/private Servatus/config/output/dataset residue;
- no active `REMOTE.toml` or `RESOURCES.toml` reference;
- clean diff and worktree.

Expected outcome:

- Campaign is the sole pre-publication execution roster;
- `SERVATUS.toml` is the sole project execution-profile document;
- scientific close remains direct KAIROS code over public Campaign result evidence;
- durable Campaign history survives without entering scientific authority;
- the KAIROS client is smaller, direct, and coupled only to public Servatus semantics;
- no execution mechanism or redundant defensive validation remains in KAIROS;
- scientific ownership stays explicit and the tree is fully usable.

Gate: one independent review of the coherent cutover, with full KAIROS gates and Standards 0 /
Spec 0. No external/live action.

## Slice I1 — KAIROS main and compact-CUDA integration

Baseline formula: exact then-current KAIROS `main`, exact accepted K2 head, and exact then-current
published compact-CUDA head.

Scope:

- phase A: create one normal non-fast-forward main merge with current `main` first and K2 second;
- resolve only real concurrent changes;
- prove main merge-tree/remerge-diff and first/second-parent deltas;
- preserve Blockweaver `0.3.4`, `datasets/<uuid>`, the accepted single `SERVATUS.toml` profile values
  and image path, canonical schemas, and all science outside K1-K2;
- phase B: merge the accepted main candidate into compact-CUDA with compact first;
- preserve only the already accepted CUDA execution delta;
- prove original compact nonmerge ancestry/order, stable patch parity, conflict resolutions, and
  every contextual hunk;
- remove only integration-created dead helpers;
- run full root/mobile/App/static/lock/API gates plus CUDA-focused and topology gates.

Non-goals:

- no push;
- no image build;
- no config cutover;
- no output, corpus, job, or Campaign action.

Expected outcome: two exact local candidates—main containing current main plus accepted adoption,
and compact containing that main behavior plus only the accepted CUDA delta. One independent review
covers both fixed merge ranges and returns Standards 0 / Spec 0 for each before E4.

## External gate E4 — combined image and isolated acceptance

Requires separate authorization after both local I1 candidates are independently GREEN. No KAIROS
ref is published first. Transfer the exact accepted compact commit into one run-owned remote build
checkout without publishing a branch, assert its full SHA, and follow KAIROS `AGENTS.md` exactly:

- isolated checkout at `/scratch.hpc/edoardo.galli3/build/kairos-cuda-<short-sha>`;
- exact full compact SHA assertion;
- isolated `APPTAINER_CACHEDIR`;
- immutable image `/scratch.hpc/edoardo.galli3/deployments/kairos-cuda-<short-sha>.sif`;
- one `sbuild` node/task, 8 CPUs, 30 GB, one hour;
- `apptainer build` followed by `apptainer test`;
- preserve the preceding accepted image.

Acceptance uses only isolated synthetic data, new Campaign state, and a run-owned
`SERVATUS.toml` whose resolved `KAIROS` values equal the accepted I1 profile except for selecting the
new immutable image. Minimum evidence:

- Campaign seal/reopen;
- scheduler observation for queued/running/terminal state;
- KAIROS result probe missing/valid behavior;
- result-aware planning and one explicit terminal retry;
- exact TRES and log identities;
- KAIROS Study/Artifact or experiment manifest publication/load through the new image;
- redacted Campaign record contains no request bytes or secrets;
- no production output or current scientific Campaign is touched.

Any failure stops before the tracked `profiles.KAIROS.target.image` field changes. No accepted ref
is pushed.

## External gate E5 — reviewed image selection

Requires a separately reviewed KAIROS config-only slice after the exact new image passes E4:

- change only `profiles.KAIROS.target.image` in `SERVATUS.toml` unless another reviewed operational
  fact requires a narrow change;
- rerun focused profile parsing and full proportionate static/test gates;
- integrate the config commit into both local main and compact candidates without push, preserving
  the accepted CUDA-only delta;
- independently review both exact final candidate ranges, including config integration topology and
  compact parity, and require Standards 0 / Spec 0 for each;
- prove the resolved selected profile is byte-for-byte the one used by isolated E4 acceptance;
- do not alter already-submitted jobs or old Campaign state.

## External gate E6 — publish coherent KAIROS refs

Blocked until E4 and E5 are GREEN and the user authorizes the exact pushes. Code, envelope, worker,
profile, and accepted image selection therefore first become public together.

Required sequence:

- fetch and pin `origin` and `research` refs immediately before mutation;
- stop on unexpected remote movement;
- push only the final accepted main and compact refs by normal fast-forward or reviewed ancestry
  merge;
- verify both remotes at exact SHAs and verify their `SERVATUS.toml` selects the accepted image;
- report exact main, compact, Servatus tag/head/hashes, Blockweaver version, profile label, and image
  SHA-256 to the coordinating Runner task.

The preceding image, Servatus `0.6.0`, schema-3 Campaign evidence, and old accepted refs remain until
the coherent refs are verified. Cleanup is a separate final approval.

## Final cleanup

After all accepted refs, packages, image/config, and isolated acceptance evidence are verified:

- remove only worktrees, branches, build/cache directories, and temporary smoke state created by
  this program;
- preserve accepted image and evidence until the user explicitly approves their removal;
- preserve historical Campaign state unless the user explicitly names it for cleanup;
- never recursively delete broad workspace, scratch, output, dataset, or deployment roots;
- verify Servatus and KAIROS return to their intended clean final worktree/branch state.

## Package gates

Every Servatus product slice runs:

```text
uv lock --check
uv sync --locked --dry-run
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run vulture
uv build
git diff --check
```

The implementer also inspects wheel/sdist rosters and metadata, verifies zero runtime dependencies,
and runs a fresh isolated installed-wheel Python API and both CLI-entry-point smoke.

Every KAIROS product/integration slice runs the exact repository-configured root, mobile-export, and
App gates listed in K2. Literal Pyright strict mode is not claimed unless the repository config is
separately changed and reviewed; the current KAIROS gate is the configured mode.

## Current status

| Item | State |
| --- | --- |
| Historical extraction/consolidation/deployment ledger | Complete; archived authority |
| New Campaign-engine architecture | User-approved |
| Repository-local Profile revision | User-approved; independent rereview GREEN |
| New ledger | GREEN, including bounded-log amendment |
| S1 Campaign roster and attempt state | Complete; Standards 0 / Spec 0 at `313b4d32` |
| S2 Campaign observation and bounded logs | Complete; Standards 0 / Spec 0 at `54b05a2e` |
| S3 Profile-based evidence planning and CLI | Complete; Standards 0 / Spec 0 at `08567547` |
| S4 Servatus consolidation | Complete; Standards 0 / Spec 0 at `d04e567c` |
| E1 protected K-study closure | Complete; Runner-attested, schema-3 evidence preserved |
| E2 disposable KAIROS client proof | Next; local and non-production |
| E0 isolated CPU Slurm/log acceptance | Authorized; blocked by E2 |
| Servatus `0.7.0` external release | Authorized; blocked by E0 and E2 |
| K1-K2 KAIROS adoption | Blocked by public `0.7.0` |
| I1 main/compact integration | Blocked by K1-K2 |
| KAIROS image/config/push gates | Separately gated after K1-K2/I1; not authorized here |

## Run record

- 2026-08-12: user approved the deeper Campaign direction and the identity split. Read-only design
  audits converged on one Campaign authority, transient application probes, native Slurm
  observation, explicit retry, derived readiness, and a KAIROS sole-roster adapter. Workflow/DAG,
  finalizer, generic Bundle, WorkspaceGroup, plugin, metadata, and universal-ID proposals were
  rejected.
- 2026-08-12: exact Servatus and KAIROS baselines were clean and equal to their published `main`
  refs. The run-owned Servatus branch/worktree above was created for this ledger only. No product or
  external state was changed.
- 2026-08-13: independent common-path and adversarial reviews rejected the first draft until it
  corrected Workspace identity ownership, offline scientific close, candidate objective validation,
  all-attempt retry/quiescence, bounded Slurm transport, plan restoration and per-allocation
  freshness, pre-release live acceptance, protected Runner evidence, redaction, independently usable
  KAIROS transitions, and the pre-release client deletion test. The plan was consolidated to four
  Servatus, two KAIROS, and one integration slice. Both reviewers then returned GREEN with zero
  actionable findings. No product or external state changed.
- 2026-08-13: user approved S1-S4, K1-K2, and I1, then approved replacing KAIROS's paired
  `REMOTE.toml`/`RESOURCES.toml` with one repository-root `SERVATUS.toml`. The file contains one or
  more complete named Profiles and an optional top-level `default_profile`; explicit `--profile`
  overrides it. No global store, discovery, environment fallback, inheritance, config manager, or
  profile-management CLI is allowed. Git owns configuration provenance while Campaign lineage owns
  exact resolved execution values. The change is folded into S3/K1 without another slice.
- 2026-08-13: implementation was authorized. The shared KAIROS checkout contained one unrelated
  user-owned untracked App-planning document, preserved untouched; a separate task owns App work in
  an isolated worktree. This program begins with Servatus S1-S4 and re-pins KAIROS only after the
  separate task and E1/E3 gates complete. No product or external state changed by this ledger update.
- 2026-08-13: profile-revision adversarial rereview rejected the first revision until it made label
  provenance explicitly nonbinding, required whole-document validation and exact cwd-only CLI
  resolution, and removed a publication window in which new KAIROS worker payloads could target the
  old image. E4 now builds and accepts the exact local candidate, E5 reviews the sole image-field
  selection, and E6 publishes code plus compatible image configuration together. Common-path review
  was already GREEN; the correction returns to the same adversarial reviewer before S1 starts.
- 2026-08-13: the first correction rereview required canonical plans and per-attempt records to
  retain the nonbinding profile label, and required image-selection integration into both final
  KAIROS candidates before reviewing their exact publishable ranges. The ledger was corrected
  without changing the approved architecture or adding a slice.
- 2026-08-13: final fixed-range ledger rereviews returned GREEN with zero actionable findings on the
  common-path and adversarial axes. S1 may start from the exact accepted ledger head; no product or
  external state changed during planning.
- 2026-08-13: S1 implemented clean-break Campaign schema 4 from exact ledger head `abe875c9`.
  Initial commit `a057aa18` added the durable `OPEN`/`SEALED` roster phase, immutable ordered
  `Campaign.tasks`, idempotent seal plus CLI, append/seal plan invalidation, and one tagged Attempt
  owner for submission lineage. Independent review rejected incomplete raw-state reconstruction.
  The same implementer added separate corrections `332168a1`, `388ee676`, and `313b4d32` for exact
  retry provenance, feasible revisions, canonical reconciliation windows, final ambiguity,
  disjoint authored-order plan allocations, and terminal negative outcomes. The same reviewer
  returned final GREEN with Standards 0 / Spec 0 on `388ee676..313b4d32`; all earlier findings
  remained closed. Final gates were 326 passed / 1 environment skip, Ruff check and format,
  configured strict Pyright, Vulture, lock and dry-sync checks, build/archive/metadata inspection,
  zero runtime dependencies, diff check, and fresh installed-wheel API plus both CLI entry points.
  No SSH, Slurm, KAIROS, protected Campaign, remote, release, or S2+ action ran. Exact accepted S1
  head is `313b4d32f905e5d8a7b380e9582ceea3b5240fac`.
- 2026-08-13: the separate App-ownership program integrated first into the clean local KAIROS
  `main`, then removed only temporary/stale Markdown. K1 must re-pin exact then-current KAIROS
  baseline `85209160b57ad146d868090e002cf69ed23a4503`. Its only expected overlap is
  `docs/KAIROS.md`; App product lives under `app/`. This ordering does not affect Servatus S1-S4.
- 2026-08-13: while S2 was focused-green but uncommitted, the user approved generic bounded Campaign
  log access. Three independent interface designs compared reference-only provenance, a ranged
  reader, and a view-centered reader. Reference-only access was too shallow because callers still
  owned SSH and bounds; offsets/following added mutable-file machinery without a demonstrated need;
  embedding content in `inspect()` would make routine views sensitive, large, and failure-prone.
  The accepted design adds one tail-only `Campaign.read_log()` plus `LogSnapshot` to S2, one thin
  raw-byte `servatus log` command to S3, and exact isolated live evidence to E0. It adds no slice,
  KAIROS wrapper, log parser, progress model, completion authority, or persistent log state. S2
  remains paused before commit until this ledger correction receives independent rereview.
- 2026-08-13: the first bounded-log ledger rereview rejected the amendment until future rendered
  filenames bound immutable `allocation_id` as well as reusable Slurm Job ID, remote reads were
  explicitly outside the Campaign lock, S2 owned its SECURITY and ADR 0003 changes, CLI help warned
  about untrusted binary terminal output, and E0/E3 proved the installed wheel and CLI surface.
  Those planning corrections are active; the paused S2 product delta remains uncommitted and must
  not resume until both original ledger reviewers return GREEN.
- 2026-08-13: correction rereview made the adversarial axis GREEN. The common-path axis required S2,
  rather than later CLI work, to prove complete `ObservationError` message/note/cause/context
  redaction and installed-wheel exports. That final evidence-placement correction is under the same
  reviewer; S2 remains paused.
- 2026-08-13: final bounded-log amendment rereview returned GREEN on both original axes. The
  accepted ledger head owns allocation-bound filenames, lock-free bounded remote reads, sensitive
  binary CLI warnings, complete Python exception-chain redaction, installed-wheel exports, and E0
  live CLI evidence without adding a slice or application authority. S2 may resume from this exact
  accepted ledger state; its earlier observation delta remains uncommitted and must incorporate the
  amended contract before fixed-range product review.
- 2026-08-13: S2 implemented `Campaign.inspect()`, immutable Campaign evidence, bounded exact Slurm
  observation, independent result readiness/quiescence, `Campaign.read_log()`, allocation-bound log
  filenames, and complete sensitive-output documentation from exact baseline `99cb093b`. Initial
  commit `a78215e9` passed all local gates but independent review rejected scheduler queries that
  were not exact under native Slurm semantics, missing accounting fallback, Job-ID reuse, incomplete
  SSH/control bounds, duplicate receipt authority, and private-knob tests. Separate corrections
  `296875fa`, `1e68d2a0`, `86195dae`, and `54b05a2e` made each query Attempt-specific, anchored
  duplicate/requeue accounting history, forced remote `TZ=UTC`, separated active/accounting
  protocols, restored real subprocess deadline/pipe-cap tests, and ordered same-incarnation state
  transitions without hiding conflicting terminal or future states. The same reviewer returned
  final GREEN with Standards 0 / Spec 0 on `86195dae..54b05a2e`; all earlier findings remained
  closed. Final gates were 408 passed / 1 environment skip, locked dry sync, Ruff check/format,
  strict Pyright, Vulture, build/archive/metadata inspection, zero runtime dependencies, diff check,
  and fresh installed-wheel public API plus both CLI entry points. No live SSH/Slurm, KAIROS,
  protected Campaign, remote, release, or S3+ action ran. Exact accepted S2 head is
  `54b05a2eb8371efd8cda741ca1f7f1161147df11`.
- 2026-08-13: S3 implemented strict repository Profile loading, one revision-bound evidence-aware
  planning path, canonical plan schema 4, per-allocation submit freshness, scheduler-only CLI
  `inspect`, and raw bounded CLI `log` from exact baseline `eaa08ab4`. Initial commit `8f8c5957`
  passed all local gates but independent review rejected global ambiguity blocking unaffected work,
  over-restricted Profile labels, quadratic scans, duplicate decoders, and serialized derived
  warnings. Correction `5eb08f83` closed those findings but exposed delayed acceptance-outcome
  chronology and one remaining refresh scan. Correction `1d166087` added the minimal
  `acceptance.outcome_revision` fact and direct refresh pairs; rereview then rejected raw-revision
  memory growth and repeated append scans. Final correction `08567547` bounded revision validation
  arithmetically by roster/Attempt facts and used one monotone chronology traversal. The same
  reviewer returned final GREEN with Standards 0 / Spec 0 on `1d166087..08567547`; all earlier S3
  findings remained closed. Final gates were 454 passed / 1 environment skip, lock and dry-sync,
  Ruff check/format, strict Pyright, Vulture, build/archive/metadata inspection, zero runtime
  dependencies, diff check, and fresh installed-wheel Profile/plan plus both CLI entry points. No
  live SSH/Slurm, KAIROS, protected Campaign, remote, release, or S4 action ran. Exact accepted S3
  head is `08567547568f945db4ea250dedde818b870cab6a`.
- 2026-08-13: S4 added canonical redacted `Campaign.record()` provenance, completed the package
  audit, and prepared version `0.7.0` from exact baseline `c8591386`. Initial commit `0238ac0a`
  passed all local gates but independent review rejected forgeable included view evidence, repeated
  roster decoding, and missing explicit attempt/retry digests. Separate correction `d04e567c`
  bound the complete authentic view semantics without a new public token, reused one validated
  roster projection, and added documented canonical redacted digests. The same reviewer returned
  GREEN with Standards 0 / Spec 0 on `0238ac0a..d04e567c`; all earlier S4 findings remained closed.
  Final gates were 456 passed / 1 environment skip, lock and dry-sync, Ruff check/format, strict
  Pyright, Vulture, build/archive/metadata inspection, zero runtime dependencies, diff check, and
  fresh installed-wheel record/API plus both CLI entry points. Exact accepted S4 head is
  `d04e567c82f9f3b566b664ea24d132aba9f3b81b`. No external action ran.
- 2026-08-13: the Runner attested E1 complete for protected K-study
  `b0e6d421-86e9-4ef6-8d81-d02f355b2da0` and schema-3 Campaign
  `9afb18b38088a3dfa88147011945ba7f`. Its canonical manifest contains exactly 27 unique selected
  LSTM references: 24 newly trained artifacts plus the three frozen `K=5` artifacts. Every intended
  Task has valid canonical evidence; no K-study job, heartbeat, retry, ambiguity resolution,
  authoring, copying, or closure work remains. The schema-3 Campaign stays preserved and untouched;
  active held-out and future inference work remain excluded. The user separately authorized E0
  isolated CPU Slurm/log acceptance and E3 Servatus `0.7.0` publication, both still subject to
  their prerequisite GREEN gates.
