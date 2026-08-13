# ADR 0003: Own one native Slurm campaign

Status: accepted

Servatus V1 supports one concrete lane: an unprivileged workstation invokes OpenSSH, absolute Slurm
executables, and one immutable Apptainer image. A Campaign freezes each registered opaque task,
permits only an exact append-only suffix while its roster is open, and seals authoring
irreversibly. Append, seal, and submission mutations preserve prior order, bytes, resource lineage,
and every attempt. Campaign owns deterministic balanced single-node allocations, complete scripts,
and durable submission records, and fails closed when scheduler acceptance is ambiguous.

One immutable Profile supplies a nonbinding label plus complete target guardrails and requested
resources. Durable compatibility binds only to exact resolved target/resource values. Each Attempt
adds the selected label, ordered Task, explicit-retry, and duplicate-risk keys, its Campaign
revision, lineage digests, plan and script digests, exact effective allocation totals, the
nonsecret `sbatch` argument vector, its reconciliation window, and one unresolved, accepted, or
not-submitted outcome. Job names derive from allocation identity; receipts remain public
projections of accepted Attempt outcomes.

Public plan files redact task arguments and payloads but retain private operational evidence. A
loaded plan parses its frozen revision-bound Campaign view, selection, retry/override decisions,
Profile, and allocations, regenerates the immutable plan without external observation, and requires
identical canonical bytes. Complete script display remains an explicit sensitive diagnostic.
Authored wall time remains provenance while planned and submitted time reflects Slurm's one-time
upward minute rounding.

Campaign schema 4 and plan schema 4 are a clean break. Schema-3 Campaign state is rejected without
migration. Every bounded owner-only state read validates the complete atomic snapshot; one tagged
Attempt outcome removes the former parallel receipt and negative-resolution authorities.
Internally typed mutations are encoded directly rather than decoded again immediately before the
atomic write.

`Campaign.open` registers an authored roster; `Campaign.load` reopens an existing one.
`Campaign.tasks` exposes the immutable authored tuple, and `Campaign.seal()` atomically and
idempotently ends suffix authoring. Execution is valid in both roster phases.
`Campaign.plan` requires one exact current Campaign view. Valid results are excluded; never-accepted
missing/unobserved Tasks are selected; ambiguity and active accepted work are withheld; terminal
accepted work requires explicit retry; and unknown accepted work also requires a recorded
duplicate-risk acknowledgement. Every accepted Attempt matters, so older active or unknown work
cannot be hidden by a newer Attempt. `Campaign.validate` first proves that a plan belongs to the
current Campaign revision, then issues bounded time-specific `sbatch --test-only` calls.
Scheduler-only `inspect` replaces the old acceptance-only status. Reconciliation reconstructs its
target from validated immutable lineage, so callers cannot supply a second route.

Before every allocation's mutating `sbatch`, submission rereads local state, reprobes only selected
Tasks for result-aware plans, refreshes relevant accepted scheduler evidence, rereads state, and
compares current eligibility with the frozen decision. It then durably records intent and contacts
Slurm. Revisions created by earlier allocations are carried through the same submission call.

The package invokes stable command-line seams directly. Submitit is prior art, not a dependency:
its cluster-local Python callable and post-acceptance pickle transport do not fit the workstation
SSH boundary or the requirement that every accepted job already own its complete payload. There is
no scheduler adapter or plugin interface until a second proven production lane requires one.

Allocation resources equal the sum of concurrent exact child steps. Servatus never infers node
capacity, escalates an explicit request, emits job-level exclusivity, or accepts raw Slurm options.
Target limits prevent user mistakes but do not replace cluster policy. Application completion and
the meaning of every task remain with the caller.

Allocation stdout/stderr share `log_root/<allocation_id>-%j.out`; each task's stdout/stderr share
`log_root/<allocation_id>-%j-<zero-based-slot>.out`. Slurm expands `%j` after acceptance, while the
immutable allocation identity prevents reused job numbers from aliasing distinct Attempts without
post-acceptance plan mutation.

## Production acceptance

On 2026-08-10, candidate `0c454bd38da4f3d5b0ba4f0777b708f8a2eb011c` passed the live gate as
an unprivileged user-side client on Slurm 23.11.4. The site used `select/cons_tres` with
`CR_CPU_MEMORY`, task cgroup and affinity plugins, `/usr/bin/ssh`, Slurm commands under `/usr/bin`,
and `/usr/bin/apptainer`. Bounded validation and jobs 44592–44595 proved CPU-only, one-GPU,
one-process/two-GPU, byte-exact argv/stdin, exact requested and allocated TRES, receipt durability,
and sibling failure aggregation. Job 44598 proved four packed one-GPU steps with four distinct GPU
UUIDs beginning within four milliseconds.

The first four-pack also established the topology boundary: on this SMT2 site, one requested Slurm
CPU represented one logical thread while each exclusive step occupied a physical core, so only two
one-CPU steps placed simultaneously. The accepted four-pack used `cpus_per_task=2` and requested
exactly eight CPUs. Callers must describe that topology truthfully. Servatus continues to preserve
resource arithmetic and binding; it does not auto-inflate CPUs, disable affinity, or add raw Slurm
options.
