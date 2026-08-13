# Architecture decisions

- [0001: Keep application meaning opaque](0001-opaque-application-seam.md)
- [0002: Publish through a durable POSIX transaction](0002-posix-workspace-publication.md)
- [0003: Own one native Slurm campaign](0003-native-slurm-campaign.md)
- [0004: Coordinate concurrent child workspaces](0004-concurrent-child-workspaces.md)
- [0005: Deepen one Campaign execution authority](0005-campaign-engine.md)

ADR 0005 is the current authority for Campaign views, planning, bounded logs, and the redacted
operational record. ADR 0003 retains the native Slurm lane and production-acceptance evidence.
