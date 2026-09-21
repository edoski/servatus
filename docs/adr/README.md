# Architecture decisions

- [0001: Keep application meaning opaque](0001-opaque-application-seam.md)
- [0002: Publish through a durable POSIX transaction](0002-posix-workspace-publication.md)
- [0003: Own one native Slurm campaign](0003-native-slurm-campaign.md)
- [0004: Coordinate concurrent child workspaces](0004-concurrent-child-workspaces.md)
- [0005: Keep one Campaign execution authority](0005-campaign-engine.md)

ADR 0005 defines Campaign authoring, state transactions, planning, and diagnostics. ADR 0003 defines
the native Slurm lane and records the scope of prior production acceptance.
