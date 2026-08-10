# Security policy

Please report vulnerabilities privately through GitHub's security-advisory form for this
repository. Do not include sensitive paths, credentials, or research data in a public issue.

Only the latest released version receives security fixes.

Servatus is an unprivileged user library, not an authorization, sandboxing, tenant-isolation, or
cluster-policy layer. Its publication transaction protects against accidental partial visibility,
overwrites, and ordinary lifecycle races. Callers remain responsible for directory permissions,
trusted builders, application validation, filesystem guarantees, and scheduler policy.

The owner-only hidden Workspace container is its lifecycle trust root. Servatus detects compliant
open/cleanup races and lock or work substitution inside that authentic container, and an active
handle rejects replacement of its container path. Arbitrary code running as the same Unix account
can rename and recreate the complete trust root and is outside this unprivileged library's threat
model. Keep destination parents private and treat workers and builders as trusted code.

Campaign task arguments and stdin are embedded in the submitted batch script. Redaction from
ordinary local summaries does not make them secret; do not submit credentials or other secrets.
Target TOML is an editable user-side guardrail, not an enforcement boundary. Slurm remains
authoritative for identity, admission, isolation, allocation, accounting, and billing.
