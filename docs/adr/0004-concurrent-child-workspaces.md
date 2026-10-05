# ADR 0004: Coordinate concurrent child workspaces

Status: accepted (revised for 0.12)

Some applications run independent resumable workers concurrently, then assemble their validated
results into one immutable destination. An ordinary parent `Workspace` cannot wrap those workers:
its exclusive lifecycle lock would reject or serialize them.

`Workspace.child(name, identity=...)` returns one ordinary resumable child beneath the parent's
private work. Children are built through a private constructor from an immutable, fully initialized
location; nothing is mutated after construction. A child holds a shared parent lifecycle lease and
an exclusive child lease. Different children may overlap; a duplicate child or parent finalization
fails immediately with `Busy`. Children cannot have children. The application decides which
children are required and when they are ready.

Opening each level takes a short exclusive `flock` on its pinned parent directory while creating or
opening the private hierarchy and acquiring its lifecycle lease without blocking. Destination
absence is checked after the lease is acquired and again before application access. If a level's
destination already exists, its leftover private work (bound to the same identity, or never
initialized) is reclaimed under an exclusive lease and entry raises `DestinationExists`; a parent
level is reclaimed only when no sibling child holds its shared lease. Parent coordination ends
before identity initialization and every durability sync, so slow synchronization cannot convoy
independent children or destinations. Within the owner-only container, the durable identity record
stores exact container, lifecycle-lock, and work inode identities. A distributed filesystem must
expose stable inode identities and one coherent `flock` domain across every participating client.

The hidden Workspace container is the lifecycle trust root. Root and child containers, work
directories, locks, and identity files must be owned by the effective user with no group or world
permissions. Descriptor-rooted cleanup and `discard()` remove only the pinned hierarchy and preserve
a moved or substituted name as residue. Arbitrary same-account code can rename and recreate a whole
root, which an unprivileged library cannot distinguish from first initialization without a separate
registry; that is outside the threat model.

Child publication atomically retains one immutable result under parent work (with the same `mode=`
and `Draft` rules as ADR 0002) and removes only that child's private workspace. Child or parent
failure preserves resumable work; parent success publishes the canonical destination and then
removes the complete private hierarchy. The parent builder typically assembles child results with
`Draft.link` or `Draft.link_tree`.

Servatus does not add child registries, readiness, polling, dependencies, workflow topology,
recursive children, or application-specific assembly rules.
