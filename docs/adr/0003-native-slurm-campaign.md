# ADR 0003: Own one native Slurm campaign

Status: accepted

Servatus supports one concrete lane: an unprivileged workstation invokes OpenSSH, absolute Slurm
executables, and one immutable Apptainer image. Campaign owns balanced single-node allocations,
complete scripts, and durable submission records. Application meaning and result validity remain
with the caller. ADR 0005 defines the current Campaign state and public workflow.

The package invokes stable command-line seams directly. A cluster-local Python callable and
post-acceptance pickle transport do not fit the workstation SSH boundary or the requirement that
accepted work already owns its complete payload. There is no scheduler adapter or plugin interface
without a second proven production lane.

Allocation resources equal the sum of concurrent exact child steps. Servatus never infers node
capacity, escalates an explicit request, emits job-level exclusivity, or accepts raw Slurm options.
Authored wall time remains provenance; planned and submitted time reflects Slurm's one-time upward
minute rounding. Target limits prevent user mistakes but do not replace cluster policy.

Each allocation starts concurrent `srun --exclusive --exact --nodes=1 --ntasks=1` steps. GPU steps
forward step-local `CUDA_VISIBLE_DEVICES` into Apptainer and set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
Missing visibility fails the step; site configuration remains responsible for device isolation.
Before launching siblings, the batch checks decoding of all binary payloads into owner-only scratch
files. It waits for every started sibling and cleans scratch after completion or handled
interruption. No remote Python runtime is required.

Allocation stdout/stderr share `log_root/<allocation_id>-%j.out`; each task's stdout/stderr share
`log_root/<allocation_id>-%j-<zero-based-slot>.out`. Slurm expands `%j` after acceptance, while the
immutable allocation identity prevents reused job numbers from aliasing distinct Attempts.

Every SSH operation has a bounded command, output, and deadline. Stream draining is concurrent;
local failures kill and reap the child and close its streams. Scheduler commands use fixed C locale
and UTC timezone. Submission checks deterministic command bounds before durable intent, while
failures after launch preserve acceptance ambiguity.

Inspection batches by original Attempt route and cluster, matching immutable allocation identities.
Positive queue evidence remains authoritative evidence of retained work even without an accounting
anchor. Terminal evidence requires anchored accounting. Held or requeued work, including
`SPECIAL_EXIT`, cannot establish quiescence or permit retry.

## Recorded production acceptance

The following evidence concerns the tested candidate and site, not a live acceptance of every later
release. On 2026-08-10, candidate `0c454bd38da4f3d5b0ba4f0777b708f8a2eb011c` passed the live gate as
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
