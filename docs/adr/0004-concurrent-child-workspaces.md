# ADR 0004: Coordinate concurrent child workspaces

Status: accepted

Some applications run independent resumable workers concurrently, then assemble their validated
results into one immutable destination. An ordinary parent `Workspace` cannot wrap those workers:
its exclusive lifecycle lock would reject or serialize them.

`Workspace.child(name, identity=...)` creates one ordinary resumable child beneath the parent's
private work. A child holds a shared parent lifecycle lease and an exclusive child lease. Different
children may overlap; duplicate children and parent finalization fail immediately with
`WorkspaceBusy`. The application decides which children are required and when they are ready.

Opening each level takes a short exclusive `flock` on its pinned parent directory while creating
or opening the private hierarchy and acquiring its lifecycle lease without blocking. Destination
absence is checked after the lease is acquired and again before application access. A publication
that finishes before the lease cannot expose redundant private work. Lifecycle leases exclude
compliant cleanup while a workspace is in use; moved or unlinked entries are rejected by live
verification. Parent coordination ends before identity initialization and every durability sync,
so slow synchronization cannot convoy independent children or destinations. Within the authentic owner-only container,
the durable identity record stores exact container, lifecycle-lock, and work inode identities only.
Active handles still require matching local device and inode values and enforce entry type and
same-filesystem checks. A distributed filesystem must expose stable inode identities and one coherent `flock` domain across
every participating client. These entries are verified before application access, publication, and
cleanup.

The hidden Workspace container is the lifecycle trust root. The root and child containers, work
directories, locks, and identity files must be owned by the effective user with no group or world
permissions; live verification checks those properties with the pinned entries. One immutable
location describes root or child identity and paths; a separate session owns only live descriptors,
leases, and publication state. First identity
commit follows durable initialization of the lock, work directory, container, and its parent.
Descriptor-rooted cleanup removes only the pinned hierarchy and preserves a moved or substituted
name. Arbitrary same-Unix-account code can rename and recreate a whole root, which an unprivileged
library cannot distinguish from first initialization without a separate registry or broader lock.
That behavior is outside the threat model; Servatus does not add external state or serialize
unrelated destinations to claim otherwise.

Child publication atomically retains one immutable result under parent work and removes only that
child's private workspace. Child or parent failure preserves resumable work; parent success
publishes the canonical destination and then removes the complete private hierarchy.

Servatus does not add child registries, readiness, polling, dependencies, workflow topology,
recursive children, or application-specific assembly rules.
