# Architecture decisions

- [0001: Keep application meaning opaque](0001-opaque-application-seam.md)
- [0002: Publish through a durable POSIX transaction](0002-posix-workspace-publication.md)
- [0003: Own one native Slurm campaign](0003-native-slurm-campaign.md)
- [0004: Coordinate concurrent child workspaces](0004-concurrent-child-workspaces.md)
- [0005: Keep one Campaign execution authority](0005-campaign-engine.md)
- [0006: Make a clean break in 0.12](0006-clean-break-0.12.md)

ADR 0002 defines the publication transaction and Workspace lifecycle; ADR 0004 extends it to
concurrent children. ADR 0003 defines the transport, launchers, batch script, and scheduler
evidence, and records the scope of prior production acceptance. ADR 0005 defines Campaign state,
planning, submission, and status. ADR 0006 records what the 0.12 clean break changed and why.
