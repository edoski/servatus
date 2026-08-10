# ADR 0003: Own one native Slurm campaign

Status: accepted

Servatus V1 supports one concrete lane: an unprivileged workstation invokes OpenSSH, absolute Slurm
executables, and one immutable Apptainer image. A Campaign freezes opaque tasks, exact homogeneous
resources, deterministic balanced single-node allocations, complete scripts, and durable intent
and receipt records. It fails closed when scheduler acceptance is ambiguous.

The package invokes stable command-line seams directly. Submitit is prior art, not a dependency:
its cluster-local Python callable and post-acceptance pickle transport do not fit the workstation
SSH boundary or the requirement that every accepted job already own its complete payload. There is
no scheduler adapter or plugin interface until a second proven production lane requires one.

Allocation resources equal the sum of concurrent exact child steps. Servatus never infers node
capacity, escalates an explicit request, emits job-level exclusivity, or accepts raw Slurm options.
Target limits prevent user mistakes but do not replace cluster policy. Application completion and
the meaning of every task remain with the caller.
