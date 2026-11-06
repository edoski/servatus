"""The only module that spawns processes: a bounded command runner behind one ``Transport`` seam.

``Ssh`` runs scheduler commands on a login host through OpenSSH; ``Local`` runs them directly when
Servatus already runs on a login node. Both execute the exact argument vector under a scrubbed
environment (fixed C locale, UTC, minimal ``PATH``). Every spawn, deadline, byte-bound, or protocol
failure surfaces as ``Unavailable``: the remote outcome is unknown. Only deterministic local bound
violations raise ``ConfigurationError`` (``_script.check_command`` checks them without spawning).
"""

from __future__ import annotations

import contextlib
import os
import posixpath
import secrets
import selectors
import shlex
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Protocol, cast

from ..errors import Unavailable
from ._config import CONTROL, Target
from ._script import check_command

MAX_STREAM_BYTES = 1024 * 1024
DEADLINE_SECONDS = 30.0
# Login banners and shell start-up noise precede the output marker; bound them separately.
NOISE_ALLOWANCE = 64 * 1024

# Each connection attempt is a new TCP flow, made before any command runs, so a flow a NAT has lost
# recovers and nothing can run twice. Three attempts take at most 26 s, inside the 30 s deadline;
# the operating system alone would wait 75 s on the first flow.
_SSH_OPTIONS = (
    "-T",
    "-o",
    "BatchMode=yes",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ConnectionAttempts=3",
    "--",
)
_ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
_REMOTE_ENV = " ".join(f"{name}={value}" for name, value in _ENVIRONMENT.items())
# What the ssh client needs to find its configuration, agent, and Kerberos (GSSAPI) credentials.
_SSH_PASSTHROUGH = ("HOME", "LOGNAME", "USER", "SSH_AUTH_SOCK", "KRB5CCNAME", "KRB5_CONFIG")
# How often to check whether the child exited while its pipes stay open.
_POLL_SECONDS = 0.05


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


@dataclass(frozen=True, slots=True)
class Ssh:
    """Run commands on ``host`` through ``ssh -T`` in batch mode, retrying the connection only.

    The remote command prints a fresh random marker on stdout and stderr, then ``exec``s
    ``/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TZ=UTC <argv>``. Anything before the
    markers (login banners, shell start-up noise) is discarded; a missing marker means the command
    never ran as requested and raises ``Unavailable``. ``command`` replaces the local ``ssh``
    invocation, for example with a fake executable in tests. ``Target`` validates ``host``, and
    ``--`` keeps it from parsing as an option.
    """

    host: str
    command: tuple[str, ...] = ("ssh",)

    def run(
        self, argv: Sequence[str], *, stdin: bytes = b"", max_stdout: int = MAX_STREAM_BYTES
    ) -> Completed:
        fields = check_command(argv)
        marker = f"servatus-{secrets.token_hex(16)}"
        script = (
            f"printf '%s\\n' {marker}; printf '%s\\n' {marker} >&2; "
            f"exec /usr/bin/env -i {_REMOTE_ENV} {shlex.join(fields)}"
        )
        remote = shlex.join(("/bin/sh", "-c", script))
        command = (*self.command, *_SSH_OPTIONS, self.host, remote)
        allowance = NOISE_ALLOWANCE + len(marker) + 1
        raw = run_bounded(
            command,
            what=posixpath.basename(fields[0]),
            stdin=stdin,
            max_stdout=max_stdout + allowance,
            max_stderr=MAX_STREAM_BYTES + allowance,
            env=ssh_environment(),
        )
        try:
            stdout = after_marker(raw.stdout, marker)
            stderr = after_marker(raw.stderr, marker)
        except Unavailable as error:
            raise Unavailable(f"{error}: {diagnostic(raw)}") from None
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
        return run_bounded(
            fields,
            what=posixpath.basename(fields[0]),
            stdin=stdin,
            max_stdout=max_stdout,
            max_stderr=MAX_STREAM_BYTES,
            env=_ENVIRONMENT,
        )


def connect(target: Target) -> Transport:
    """The default transport for ``target``: SSH to ``target.host``, or local when it is None."""
    return Local() if target.host is None else Ssh(target.host)


def ssh_environment() -> dict[str, str]:
    """The local environment for the ``ssh`` client: no scheduler overrides leak through."""
    return _ENVIRONMENT | {
        name: os.environ[name] for name in _SSH_PASSTHROUGH if os.environ.get(name)
    }


def diagnostic(completed: Completed) -> str:
    """The exit status and first stderr line, bounded and printable, for operator messages."""
    first = completed.stderr.decode("utf-8", "replace").strip().partition("\n")[0]
    text = CONTROL.sub("?", first)[:200]
    return f"exit status {completed.returncode}" + (f": {text}" if text else "")


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
    what: str,
    stdin: bytes,
    max_stdout: int,
    max_stderr: int,
    env: Mapping[str, str],
) -> Completed:
    """Run ``command`` with a deadline, concurrent bounded draining, and guaranteed reaping.

    ``what`` names the command in errors. Stdin comes from an anonymous temporary file, so a child
    that never reads cannot deadlock the parent. Any failure kills the child's process group, reaps
    it, and raises ``Unavailable``.
    """
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
        raise Unavailable(f"{what} exceeded its {DEADLINE_SECONDS:g} s deadline") from None
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        detail = getattr(error, "strerror", None) or type(error).__name__
        raise Unavailable(f"{what} could not run: {detail}") from None


def _drain(process: subprocess.Popen[bytes], max_stdout: int, max_stderr: int) -> Completed:
    """Read both pipes until EOF, or until the child has exited and its buffered output is read.

    A descendant may outlive the child and keep the pipes open (an OpenSSH ControlPersist master
    holding stderr); waiting for EOF would then stall every command until the deadline.
    """
    out, err = (cast(IO[bytes], stream).fileno() for stream in (process.stdout, process.stderr))
    limits = {out: max_stdout, err: max_stderr}
    buffers = {out: bytearray(), err: bytearray()}

    def take(descriptor: int) -> bytes | None:
        """The next chunk (empty at EOF), or None when nothing is buffered right now."""
        buffer = buffers[descriptor]
        try:
            chunk = os.read(descriptor, min(65_536, limits[descriptor] + 1 - len(buffer)))
        except BlockingIOError:
            return None
        buffer.extend(chunk)
        if len(buffer) > limits[descriptor]:
            raise Unavailable("command output exceeds its byte bound")
        return chunk

    deadline = time.monotonic() + DEADLINE_SECONDS
    with selectors.DefaultSelector() as selector:
        for descriptor in limits:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("command", DEADLINE_SECONDS)
            for key, _ in selector.select(min(remaining, _POLL_SECONDS)):
                if take(key.fd) == b"":
                    selector.unregister(key.fd)
            if process.poll() is not None:
                for key in tuple(selector.get_map().values()):
                    while take(key.fd):
                        pass
                break
    returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    return Completed(returncode, bytes(buffers[out]), bytes(buffers[err]))


def _kill(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        process.kill()
    process.wait()
