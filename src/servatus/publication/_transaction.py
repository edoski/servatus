"""Durable no-replace publication: one private stage directory, two commit shapes.

Every stage is a fresh owner-only (0700) directory beside the destination. A directory
publication commits the stage itself; a file publication commits one member of the stage that
carries the real destination name, so writers may use any strategy (suffix-appending savers,
temp-file-then-rename) and their leftovers are discarded with the stage.
"""

from __future__ import annotations

import errno
import os
import stat
import warnings
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .. import _fs
from .._fs import Pin, StrPath
from ..errors import ConfigurationError, CrossDeviceError, DestinationExists, UnsafeFilesystem

STAGE_PREFIX = ".servatus-stage-"


@dataclass(frozen=True, slots=True)
class Publication:
    """A committed destination.

    `cleanup_pending` is true when the destination is committed and durable but some private
    residue (stage, Workspace tree, or retired sibling) could not be removed or its removal could
    not be synced. It never means the publication failed.
    """

    destination: Path
    cleanup_pending: bool


@dataclass(slots=True)
class DraftState:
    """Mutable lifecycle of one Draft, owned by the transaction that created it."""

    stage: Pin
    path: Path
    live: bool = True
    failure: UnsafeFilesystem | None = None


class Draft:
    """Builder handle for one private stage directory.

    A Draft is valid only while its builder runs: any use after the builder returns raises
    `RuntimeError`. Files hard-linked into a Draft alias their source inode, so rewriting the
    source in place (rather than replacing it) also changes the published file.
    """

    __slots__ = ("_state",)

    def __init__(self, state: DraftState) -> None:
        self._state = state

    @property
    def path(self) -> Path:
        """The stage directory. Write the result here; it becomes the destination on commit."""
        return self._live().path

    def link(self, source: StrPath, destination: str | os.PathLike[str]) -> None:
        """Hard-link one same-filesystem regular file to a safe relative draft path.

        The hard-link operation selects the source inode, so a safe replacement of the source
        path just before the call may be selected. Missing intermediate directories are created.
        """
        state = self._live()
        source_path = _fs.fspath(source)
        *parents, name = _components(destination, allow_root=False)
        with _fs.Boundary(destination), _directory(state, parents) as directory:
            try:
                os.link(source_path, name, dst_dir_fd=directory.fd, follow_symlinks=False)
            except OSError as error:
                raise _link_error(error, source_path, destination) from error
            try:
                linked = os.stat(name, dir_fd=directory.fd, follow_symlinks=False)
            except OSError as error:
                _withdraw(state, directory, name, f"{destination} could not be inspected", error)
                raise UnsafeFilesystem(
                    f"hard-link destination is unavailable: {destination}"
                ) from error
            if not stat.S_ISREG(linked.st_mode):
                _withdraw(state, directory, name, f"{destination} is not a regular file", None)
                raise UnsafeFilesystem(f"hard-link source is not a regular file: {source_path}")

    def link_tree(self, source: StrPath, destination: str | os.PathLike[str] = ".") -> None:
        """Hard-link every regular file under directory `source` into `destination`.

        Subdirectories are recreated; symlinks and special files are rejected. Every linked
        entry must be the exact inode inspected in the source tree. A source tree that contains
        the draft itself (or the destination inside it) is a `ConfigurationError`.
        """
        state = self._live()
        parts = _components(destination, allow_root=True)
        with (
            _fs.Boundary(source),
            _fs.open_path(source) as tree,
            _directory(state, parts) as target,
        ):
            _link_contents(state, tree, target, {_key(state.stage.entry), _key(target.entry)})

    def _live(self) -> DraftState:
        if not self._state.live:
            raise RuntimeError("draft is no longer valid after its builder returned")
        return self._state


@contextmanager
def _directory(state: DraftState, parts: list[str] | tuple[str, ...]) -> Generator[Pin]:
    with ExitStack() as stack:
        current = state.stage
        for part in parts:
            current = stack.enter_context(current.mkdir(part, exist_ok=True))
        yield current


def _key(entry: os.stat_result) -> tuple[int, int]:
    return entry.st_dev, entry.st_ino


def _link_contents(
    state: DraftState, source: Pin, target: Pin, inside: set[tuple[int, int]]
) -> None:
    """Mirror `source` into `target`; `inside` holds the draft directories it must not contain."""
    if _key(source.entry) in inside:
        raise ConfigurationError("link_tree source contains the draft it is linked into")

    def visit(directory: Pin, name: str, entry: os.stat_result) -> None:
        if stat.S_ISDIR(entry.st_mode):
            with (
                directory.open(name, expected=entry) as child,
                target.mkdir(name, exist_ok=True) as into,
            ):
                inside.add(_key(into.entry))
                _link_contents(state, child, into, inside)
            return
        try:
            os.link(
                name, name, src_dir_fd=directory.fd, dst_dir_fd=target.fd, follow_symlinks=False
            )
        except OSError as error:
            raise _link_error(error, name, name) from error
        try:
            target.expect(name, entry)  # the linked inode is exactly the one inspected
        except UnsafeFilesystem as error:
            _withdraw(state, target, name, f"{name} was substituted during linking", error)
            raise

    _fs.walk(source, visit)


def _withdraw(
    state: DraftState, directory: Pin, name: str, problem: str, cause: BaseException | None
) -> None:
    """Remove a rejected hard link; if that fails, poison the Draft so it cannot publish."""
    try:
        os.unlink(name, dir_fd=directory.fd)
    except OSError as error:
        if cause is not None:
            error.add_note(f"Rejected because {problem}: {cause}")
        failure = UnsafeFilesystem(f"rejected hard link could not be removed: {problem}")
        state.failure = failure
        raise failure from error


def _components(destination: str | os.PathLike[str], *, allow_root: bool) -> tuple[str, ...]:
    text = _fs.fspath(destination)
    path = PurePosixPath(text)
    if allow_root and text == ".":
        return ()
    if path.is_absolute() or not path.parts or any(p in {"", ".", ".."} for p in text.split("/")):
        raise ConfigurationError(f"draft path must be safe and relative: {text!r}")
    return path.parts


def _link_error(error: OSError, source: object, destination: object) -> Exception:
    if error.errno == errno.EEXIST:
        return DestinationExists(f"draft path already exists: {destination}")
    if error.errno == errno.ENOENT:
        return ConfigurationError(f"hard-link source does not exist: {source}")
    if error.errno in {errno.EPERM, errno.EACCES, errno.ELOOP, errno.EISDIR}:
        return UnsafeFilesystem(f"unsafe hard-link source: {source}")
    if error.errno == errno.EXDEV:
        return CrossDeviceError(f"hard-link source is on another filesystem: {source}")
    return _fs.os_error(error, f"hard link of {source}")


# -- transaction -----------------------------------------------------------------------------
Build = Callable[[Path, Pin], "str | None"]
"""Fill the stage; return the member name to publish, or None to publish the stage itself."""


@dataclass(frozen=True, slots=True)
class Retired:
    name: str
    pin: Pin


def transact(
    parent: Path,
    directory: Pin,
    name: str,
    build: Build,
    *,
    mode: int | None,
    retire: Retired | None = None,
) -> bool:
    """Run one publication; return True when every private cleanup step was proven."""
    directory.verify_path(parent)
    directory.absent(name, parent / name)
    stage_name, stage = directory.unique(STAGE_PREFIX)
    with stage, ExitStack() as pins:
        try:
            umask = _fs.probe_umask(stage) if mode is None else 0
            member = build(parent / stage_name, stage)
            if member is None:
                _fs.sync_tree(stage, 0o777 & ~umask if mode is None else mode)
                source, source_name, entry = directory, stage_name, stage.entry
            else:
                payload = pins.enter_context(_payload(stage, member, name))
                os.fchmod(payload.fd, 0o666 & ~umask if mode is None else mode)
                _fs.sync(payload.fd)
                source, source_name, entry = stage, member, payload.entry
            directory.expect(stage_name, stage.entry)
            directory.verify_path(parent)
            _fs.commit(source, source_name, directory, name, entry)
        except BaseException as error:
            if not _fs.discard(directory, stage_name, stage):
                error.add_note(f"Servatus could not remove the failed stage {parent / stage_name}")
            raise
        clean = member is None or _fs.discard(directory, stage_name, stage)
    _fs.sync(directory.fd)  # the durability point; failure here leaves an unproven commit
    if retire is not None:
        retired = _fs.succeeded(
            lambda: _fs.require_owner_only(os.fstat(retire.pin.fd), "retirement source")
        ) and _fs.discard(directory, retire.name, retire.pin)
        clean = retired and _fs.succeeded(_fs.sync, directory.fd) and clean
    return clean


def _payload(stage: Pin, member: str, name: str) -> Pin:
    """Pin the writer's single-link regular file."""
    if stage.lstat(member) is None:
        raise ConfigurationError(f"the writer did not create the file to publish: {name}")
    payload = stage.open(member, directory=False)
    if payload.entry.st_nlink != 1:
        payload.close()
        raise UnsafeFilesystem(f"published file must have one link: {name}")
    return payload


@contextmanager
def _retirement(
    parent: Path, directory: Pin, name: str, retire: StrPath | None
) -> Generator[Retired | None]:
    if retire is None:
        yield None
        return
    retire_parent, retire_name = _fs.split(retire)
    if retire_parent != parent:
        raise ConfigurationError(f"retirement source must be a destination sibling: {retire}")
    if retire_name == name:
        raise ConfigurationError(f"retirement source must differ from the destination: {retire}")
    if directory.lstat(retire_name) is None:
        raise ConfigurationError(f"retirement source does not exist: {retire}")
    with directory.open(retire_name) as pin:
        _fs.require_owner_only(pin.entry, f"retirement source {retire}")
        directory.expect(retire_name, pin.entry)
        yield Retired(retire_name, pin)


def warn_pending(message: str, *, stacklevel: int) -> None:
    """Emit one best-effort warning; no filter or hook may turn committed work into failure."""
    with suppress(BaseException):
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel + 1)


def result(destination: Path, clean: bool) -> Publication:
    if not clean:
        warn_pending("publication committed, but private cleanup remains pending", stacklevel=3)
    return Publication(destination, cleanup_pending=not clean)


def draft_builder(build: Callable[[Draft], object]) -> Build:
    def run(path: Path, stage: Pin) -> None:
        state = DraftState(stage, path)
        try:
            build(Draft(state))
        finally:
            state.live = False
        if state.failure is not None:
            raise state.failure

    return run


def checked_mode(mode: int | None) -> int | None:
    return None if mode is None else _fs.check_mode(mode)


# -- public entry points ---------------------------------------------------------------------
def publish(
    destination: StrPath,
    build: Callable[[Draft], object],
    *,
    retire: StrPath | None = None,
    mode: int | None = None,
) -> Publication:
    """Build a directory in a private stage and commit it as `destination`, never replacing.

    `build` receives a `Draft` and must finish all content mutations before returning. Every
    directory in the published tree is set to `mode` just before commit (default: `0o777` minus
    the umask); file modes are left as written. With `retire`, one existing owner-only sibling
    directory is pinned before the builder and removed only after the commit is durable.
    """
    _fs.require_supported_platform()
    checked = checked_mode(mode)
    parent, name = _fs.split(destination)
    with (
        _fs.Boundary(parent / name) as boundary,
        _fs.open_path(parent) as directory,
        _retirement(parent, directory, name, retire) as retired,
    ):
        build = boundary.passthrough(build)
        clean = transact(
            parent, directory, name, draft_builder(build), mode=checked, retire=retired
        )
    return result(parent / name, clean)


def publish_file(
    destination: StrPath, write: Callable[[Path], object], *, mode: int | None = None
) -> Publication:
    """Commit one regular file as `destination`, never replacing.

    `write` receives `<private stage>/<destination name>` and may create it by any means
    (in place, or temp file then rename); other files it leaves in the stage are discarded.
    The file must be a single-link regular file when `write` returns. Its mode is set to `mode`
    just before commit (default: `0o666` minus the umask), overriding any mode the writer chose.
    """
    _fs.require_supported_platform()
    checked = checked_mode(mode)
    parent, name = _fs.split(destination)

    with _fs.Boundary(parent / name) as boundary, _fs.open_path(parent) as directory:
        written = boundary.passthrough(write)

        def build(path: Path, stage: Pin) -> str:
            del stage
            written(path / name)
            return name

        clean = transact(parent, directory, name, build, mode=checked)
    return result(parent / name, clean)
