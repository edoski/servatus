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
creates the private hierarchy, and acquires lifecycle leases without blocking. This prevents a
removed lock pathname from being recreated as an independently locked inode and avoids waiting on
a lifecycle lease while holding coordination. Container, lifecycle-lock, and work entries remain
pinned and are verified before publication and cleanup. The durable identity record binds their
device and inode identities, so a later opener fails closed if any lifecycle pathname was replaced.

Child publication atomically retains one immutable result under parent work and removes only that
child's private workspace. Child or parent failure preserves resumable work; parent success
publishes the canonical destination and then removes the complete private hierarchy.

Servatus does not add child registries, readiness, polling, dependencies, workflow topology,
recursive children, or application-specific assembly rules.
