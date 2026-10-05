"""The only module that spawns processes: a bounded command runner behind one ``Transport`` seam.

``Ssh`` runs scheduler commands on a login host through OpenSSH; ``Local`` runs them directly when
Servatus already runs on a login node. Both execute the exact argument vector under a scrubbed
environment (fixed C locale, UTC, minimal ``PATH``). Every spawn, deadline, byte-bound, or protocol
failure surfaces as ``Unavailable``: the remote outcome is unknown. Only deterministic local bound
violations raise ``ConfigurationError``, and ``check_command`` checks them without spawning.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import selectors
import shlex
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..errors import ConfigurationError, Unavailable
from ._config import Target

MAX_ARGS = 32
MAX_COMMAND_BYTES = 16 * 1024
MAX_FIELD_BYTES = 4096
MAX_STREAM_BYTES = 1024 * 1024
DEADLINE_SECONDS = 30.0
# Login banners and shell start-up noise precede the output marker; bound them separately.
NOISE_ALLOWANCE = 64 * 1024

_PATH = "/usr/bin:/bin"
_REMOTE_ENV = ("PATH=" + _PATH, "LANG=C", "LC_ALL=C", "TZ=UTC")
_LOCAL_ENV = {"PATH": _PATH, "LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
_SSH_PASSTHROUGH = ("HOME", "LOGNAME", "USER", "SSH_AUTH_SOCK")


@dataclass(frozen=True, slots=True)
class Completed:
    """One finished command: its exit status and both output streams."""

    returncode: int
    stdout: bytes
    stderr: bytes


class Transport(Protocol):
    """Runs one absolute argument vector with byte-exact stdin and bounded output."""

    def run(
        self, argv: Sequence[str], *, stdin: bytes = b"", max_stdout: int = MAX_STREAM_BYTES
    ) -> Completed: ...


def check_command(argv: Sequence[str]) -> tuple[str, ...]:
    """Check the deterministic bounds of one command without running it.

    At most 32 arguments (the first a nonempty program), 4 KiB per argument, and 16 KiB in total,
    all UTF-8 text without NUL.
    Raises ``ConfigurationError``, so callers can reject a command before recording any intent.
    """
    if isinstance(argv, (str, bytes)):
        raise ConfigurationError("command must be a sequence of arguments")
    fields = tuple(argv)
    if not fields or any(not isinstance(field, str) for field in fields):
        raise ConfigurationError("command must be a nonempty sequence of strings")
    if not fields[0]:
        raise ConfigurationError("command program must be nonempty")
    if len(fields) > MAX_ARGS:
        raise ConfigurationError(f"command exceeds its bound of {MAX_ARGS} arguments")
    total = 0
    for field in fields:
        if "\0" in field:
            raise ConfigurationError("command arguments cannot contain NUL")
        try:
            size = len(field.encode("utf-8"))
        except UnicodeEncodeError:
            raise ConfigurationError("command arguments must be valid UTF-8 text") from None
        if size > MAX_FIELD_BYTES:
            raise ConfigurationError(
                f"command argument exceeds its bound of {MAX_FIELD_BYTES} bytes"
            )
        total += size + 1
    if total > MAX_COMMAND_BYTES:
        raise ConfigurationError(f"command exceeds its bound of {MAX_COMMAND_BYTES} bytes")
    return fields


def _check_stdout_bound(max_stdout: int) -> None:
    if type(max_stdout) is not int or max_stdout < 0:
        raise ConfigurationError("max_stdout must be a non-negative integer")


@dataclass(frozen=True, slots=True, init=False)
class Ssh:
    """Run commands on ``host`` through ``ssh -T -o BatchMode=yes -o LogLevel=ERROR``.

    The remote command prints a fresh random marker on stdout and stderr, then ``exec``s
    ``/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TZ=UTC <argv>``. Anything before the
    markers (login banners, shell start-up noise) is discarded; a missing marker means the command
    never ran as requested and raises ``Unavailable``. ``command`` replaces the local ``ssh``
    invocation, for example with a fake executable in tests.
    """

    host: str
    command: tuple[str, ...]

    def __init__(self, host: str, *, command: Sequence[str] = ("ssh",)) -> None:
        if (
            not isinstance(host, str)
            or not host
            or host.startswith("-")
            or any(character.isspace() or not character.isprintable() for character in host)
            or len(host.encode("utf-8", "replace")) > MAX_FIELD_BYTES
        ):
            raise ConfigurationError("host must be one SSH destination")
        prefix = check_command(command)
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "command", prefix)

    def run(
        self, argv: Sequence[str], *, stdin: bytes = b"", max_stdout: int = MAX_STREAM_BYTES
    ) -> Completed:
        fields = check_command(argv)
        _check_stdout_bound(max_stdout)
        marker = f"servatus-{secrets.token_hex(16)}"
        script = (
            f"printf '%s\\n' {marker}; printf '%s\\n' {marker} >&2; "
            f"exec /usr/bin/env -i {' '.join(_REMOTE_ENV)} {shlex.join(fields)}"
        )
        remote = shlex.join(("/bin/sh", "-c", script))
        command = (
            *self.command,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "LogLevel=ERROR",
            "--",
            self.host,
            remote,
        )
        allowance = NOISE_ALLOWANCE + len(marker) + 1
        raw = run_bounded(
            command,
            stdin=stdin,
            max_stdout=max_stdout + allowance,
            max_stderr=MAX_STREAM_BYTES + allowance,
            env=ssh_environment(),
        )
        stdout = after_marker(raw.stdout, marker)
        stderr = after_marker(raw.stderr, marker)
        if len(stdout) > max_stdout or len(stderr) > MAX_STREAM_BYTES:
            raise Unavailable("remote command output exceeds its byte bound")
        return Completed(raw.returncode, stdout, stderr)


@dataclass(frozen=True, slots=True)
class Local:
    """Run commands directly on this host (a login node): no shell, scrubbed environment."""

    def run(
        self, argv: Sequence[str], *, stdin: bytes = b"", max_stdout: int = MAX_STREAM_BYTES
    ) -> Completed:
        fields = check_command(argv)
        _check_stdout_bound(max_stdout)
        return run_bounded(
            fields,
            stdin=stdin,
            max_stdout=max_stdout,
            max_stderr=MAX_STREAM_BYTES,
            env=dict(_LOCAL_ENV),
        )


def connect(target: Target) -> Transport:
    """The default transport for ``target``: SSH to ``target.host``, or local when it is None."""
    return Local() if target.host is None else Ssh(target.host)


def ssh_environment() -> dict[str, str]:
    """The local environment for the ``ssh`` client: no scheduler overrides leak through."""
    environment = {"PATH": _PATH, "LANG": "C", "LC_ALL": "C"}
    for name in _SSH_PASSTHROUGH:
        if value := os.environ.get(name):
            environment[name] = value
    return environment


def after_marker(output: bytes, marker: str) -> bytes:
    """The bytes after the first ``marker`` line; ``Unavailable`` when the marker is absent."""
    token = marker.encode("ascii") + b"\n"
    index = output.find(token)
    if index < 0:
        raise Unavailable("remote shell did not run the command (output marker missing)")
    return output[index + len(token) :]


def run_bounded(
    command: Sequence[str],
    *,
    stdin: bytes,
    max_stdout: int,
    max_stderr: int,
    env: Mapping[str, str],
) -> Completed:
    """Run ``command`` with a deadline, concurrent bounded draining, and guaranteed reaping.

    Stdin comes from an anonymous temporary file, so a child that never reads cannot deadlock the
    parent. Any failure kills the child's process group, reaps it, and raises ``Unavailable``.
    """
    if not isinstance(stdin, bytes):
        raise ConfigurationError("stdin must be bytes")
    try:
        with tempfile.TemporaryFile() as source:
            source.write(stdin)
            source.flush()
            source.seek(0)
            process = subprocess.Popen(
                tuple(command),
                stdin=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),
                start_new_session=True,
            )
            try:
                return _drain(process, max_stdout, max_stderr)
            except BaseException:
                _kill(process)
                raise
            finally:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
    except Unavailable:
        raise
    except subprocess.TimeoutExpired:
        raise Unavailable(f"command exceeded its {DEADLINE_SECONDS:g} s deadline") from None
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        detail = error.strerror if isinstance(error, OSError) and error.strerror else ""
        raise Unavailable(f"command could not run: {detail or type(error).__name__}") from None


def _drain(process: subprocess.Popen[bytes], max_stdout: int, max_stderr: int) -> Completed:
    stdout, stderr = process.stdout, process.stderr
    if stdout is None or stderr is None:  # pragma: no cover - both pipes are always requested
        raise Unavailable("command streams are unavailable")
    out, err = stdout.fileno(), stderr.fileno()
    limits = {out: max_stdout, err: max_stderr}
    buffers = {out: bytearray(), err: bytearray()}
    deadline = time.monotonic() + DEADLINE_SECONDS
    with selectors.DefaultSelector() as selector:
        for descriptor in limits:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("command", DEADLINE_SECONDS)
            for key, _ in selector.select(remaining):
                descriptor = key.fd
                buffer = buffers[descriptor]
                try:
                    chunk = os.read(descriptor, min(65_536, limits[descriptor] + 1 - len(buffer)))
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                buffer.extend(chunk)
                if len(buffer) > limits[descriptor]:
                    raise Unavailable("command output exceeds its byte bound")
    returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    return Completed(returncode, bytes(buffers[out]), bytes(buffers[err]))


def _kill(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        process.kill()
    process.wait()
