# ADR 0001: Keep application meaning opaque

Status: accepted

Servatus accepts opaque identity bytes and one application-owned builder callback. The callback
writes the exact destination tree and performs domain validation before returning.

Servatus does not receive artifact schemas, validators, checkpoint formats, completion enums, or
workflow topology. This keeps the lifecycle reusable without becoming an ML framework or moving
scientific authority away from the calling project.
