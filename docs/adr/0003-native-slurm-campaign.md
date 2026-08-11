# ADR 0003: Own one native Slurm campaign

Status: accepted

Servatus V1 supports one concrete lane: an unprivileged workstation invokes OpenSSH, absolute Slurm
executables, and one immutable Apptainer image. A Campaign freezes each registered opaque task and
permits only an exact append-only suffix, preserving prior order, bytes, resource lineage, intent,
and receipts. It owns deterministic balanced single-node allocations, complete scripts, and durable
submission records, and fails closed when scheduler acceptance is ambiguous.

The durable lineage retains normalized target guardrails and requested resources once. Each intent
adds ordered Task keys, plan and script digests, exact effective allocation totals, the nonsecret
`sbatch` argument vector, and its reconciliation window. Job names derive from allocation identity;
receipts retain only allocation and Slurm identities; negative resolutions are allocation IDs.
Public plan files redact task arguments and payloads. A loaded plan parses only its typed planning
inputs, regenerates the immutable plan once, and requires identical canonical bytes, so no second
derived-field validator is needed. Complete script display remains an explicit sensitive
diagnostic. Authored wall time remains provenance while planned and submitted time reflects Slurm's
one-time upward minute rounding.

Campaign and plan schema 3 are a clean break. Every bounded owner-only state read validates the
complete atomic snapshot, including that an allocation cannot have both an acceptance receipt and
an explicit not-submitted resolution. Internally typed mutations are encoded directly rather than
decoded again immediately before the atomic write.

The package invokes stable command-line seams directly. Submitit is prior art, not a dependency:
its cluster-local Python callable and post-acceptance pickle transport do not fit the workstation
SSH boundary or the requirement that every accepted job already own its complete payload. There is
no scheduler adapter or plugin interface until a second proven production lane requires one.

Allocation resources equal the sum of concurrent exact child steps. Servatus never infers node
capacity, escalates an explicit request, emits job-level exclusivity, or accepts raw Slurm options.
Target limits prevent user mistakes but do not replace cluster policy. Application completion and
the meaning of every task remain with the caller.

Allocation stdout/stderr share `log_root/%j.out`; each task's stdout/stderr share
`log_root/%j-<zero-based-slot>.out`. Slurm expands `%j` after acceptance, so conventional job-ID
logs do not require post-acceptance plan mutation.

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
