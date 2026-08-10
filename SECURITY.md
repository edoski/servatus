# Security policy

Please report vulnerabilities privately through GitHub's security-advisory form for this
repository. Do not include sensitive paths, credentials, or research data in a public issue.

Only the latest released version receives security fixes.

Servatus is an unprivileged user library, not an authorization, sandboxing, tenant-isolation, or
cluster-policy layer. Its publication transaction protects against accidental partial visibility,
overwrites, and ordinary lifecycle races. Callers remain responsible for directory permissions,
trusted builders, application validation, filesystem guarantees, and scheduler policy.

Campaign task arguments and stdin are embedded in the submitted batch script. Redaction from
ordinary local summaries does not make them secret; do not submit credentials or other secrets.
Target TOML is an editable user-side guardrail, not an enforcement boundary. Slurm remains
authoritative for identity, admission, isolation, allocation, accounting, and billing.
