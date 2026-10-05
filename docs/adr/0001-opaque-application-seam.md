# ADR 0001: Keep application meaning opaque

Status: accepted

Servatus accepts opaque values from the application: Task keys, argument vectors, environment, and
stdin bytes; Workspace identity bytes; builder and writer callbacks that write the exact
destination and validate it before returning; and an optional result probe that answers which Task
keys have valid results.

Servatus does not receive artifact schemas, validators, checkpoint formats, completion enums, or
workflow topology. It never stores probe answers or callbacks. This keeps the lifecycle reusable
without becoming an ML framework or moving scientific authority away from the calling project.
