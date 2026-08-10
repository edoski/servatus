# ADR 0004: Coordinate concurrent child workspaces

Status: accepted

Some applications run independent resumable workers concurrently, then assemble their validated
results into one immutable destination. An ordinary parent `Workspace` cannot wrap those workers:
its exclusive lifecycle lock would reject or serialize them.

`Workspace.child(name, identity=...)` creates one ordinary resumable child beneath the parent's
private work. A child holds a shared parent lifecycle lease and an exclusive child lease. Different
children may overlap; duplicate children and parent finalization fail immediately with
`WorkspaceBusy`. The application decides which children are required and when they are ready.

Open and cleanup take a short exclusive `flock` on the pinned canonical destination-parent
directory. While coordinated, Servatus checks that the canonical destination is absent, opens or
creates the private hierarchy, and acquires lifecycle leases without blocking. This prevents
compliant Servatus open and cleanup paths from splitting onto different lock inodes and avoids
waiting on a lifecycle lease while holding coordination. Within the authentic owner-only container,
the durable identity record binds exact container, lifecycle-lock, and work inode identities. Its
device values remain in the V1 format but are client-local information. Active handles still require
matching local device and inode values and enforce entry type and same-filesystem checks. A
distributed filesystem must expose stable inode identities and one coherent `flock` domain across
every participating client. These entries are verified before application access, publication, and
cleanup.

The hidden Workspace container is the lifecycle trust root. Arbitrary same-Unix-account code can
rename and recreate that whole root, which an unprivileged library cannot distinguish from first
initialization without a separate registry or broader lock. That behavior is outside the threat
model; Servatus does not add external state or serialize unrelated destinations to claim otherwise.

Child publication atomically retains one immutable result under parent work and removes only that
child's private workspace. Child or parent failure preserves resumable work; parent success
publishes the canonical destination and then removes the complete private hierarchy.

Servatus does not add child registries, readiness, polling, dependencies, workflow topology,
recursive children, or application-specific assembly rules.
