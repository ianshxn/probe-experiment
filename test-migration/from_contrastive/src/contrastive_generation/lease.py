"""Best-effort single-machine lease for one generation run pair."""

from __future__ import annotations

import json
import os
import platform
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .utils import (
    CONTRASTIVE_ROOT,
    ContrastiveError,
    atomic_write_json,
    load_json,
    resolve_local_path,
    utc_now,
)


LEASE_SCHEMA_PATH = (
    CONTRASTIVE_ROOT / "schemas" / "contrastive_run_lease.schema.json"
)


class LeaseHeld(ContrastiveError):
    """Another live process currently owns the run pair."""


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no UTC offset")
    return parsed.astimezone(timezone.utc)


class RunLease:
    """Exclusive-create lease with heartbeat and best-effort stale takeover."""

    def __init__(
        self,
        run_dir_a: str | Path,
        run_dir_b: str | Path,
        *,
        timeout_seconds: float = 300,
    ) -> None:
        self.run_dir_a = resolve_local_path(run_dir_a)
        self.run_dir_b = resolve_local_path(run_dir_b)
        self.path = self.run_dir_a / "lease.json"
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ContrastiveError("Lease timeout must be positive")
        self.owner_id = f"{platform.node()}:{os.getpid()}:{uuid.uuid4()}"
        self._held = False
        self._stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _payload(self, acquired_at: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "owner_id": self.owner_id,
            "run_dir_a": str(self.run_dir_a),
            "run_dir_b": str(self.run_dir_b),
            "acquired_at": acquired_at,
            "heartbeat_at": utc_now(),
            "timeout_seconds": self.timeout_seconds,
        }

    @staticmethod
    def current(path_or_run_dir: str | Path) -> dict[str, Any] | None:
        path = resolve_local_path(path_or_run_dir)
        if path.is_dir():
            path = path / "lease.json"
        if not path.is_file():
            return None
        try:
            value = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ContrastiveError(f"Could not read lease {path}: {exc}") from exc
        errors = list(
            Draft202012Validator(load_json(LEASE_SCHEMA_PATH)).iter_errors(value)
        )
        if errors:
            raise ContrastiveError(
                f"Invalid lease {path}: {errors[0].message}"
            )
        return value

    def _stale(self, value: dict[str, Any]) -> bool:
        try:
            heartbeat = _parse_timestamp(str(value["heartbeat_at"]))
        except (KeyError, TypeError, ValueError):
            return True
        age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
        return age > self.timeout_seconds

    def _try_stale_takeover(self) -> bool:
        try:
            before = self.path.read_bytes()
            value = json.loads(before.decode("utf-8"))
        except FileNotFoundError:
            return True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(value, dict) or not self._stale(value):
            return False
        try:
            if self.path.read_bytes() != before:
                return False
            self.path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return True

    def acquire(self) -> "RunLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        acquired_at = utc_now()
        payload = self._payload(acquired_at)
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                if self._try_stale_takeover():
                    continue
                holder = self.current(self.path)
                label = holder.get("owner_id") if holder else "<unreadable>"
                raise LeaseHeld(
                    f"Generation run pair is leased by {label}: {self.path}"
                )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._held = True
            interval = max(1.0, min(30.0, self.timeout_seconds / 3))
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(interval, acquired_at),
                name="contrastive-run-lease-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()
            return self
        raise LeaseHeld(f"Could not acquire generation lease: {self.path}")

    def _heartbeat_loop(self, interval: float, acquired_at: str) -> None:
        while not self._stop.wait(interval):
            try:
                self.heartbeat(acquired_at=acquired_at)
            except ContrastiveError:
                return

    def heartbeat(self, *, acquired_at: str | None = None) -> None:
        with self._lock:
            if not self._held:
                raise ContrastiveError("Cannot heartbeat a lease that is not held")
            current = self.current(self.path)
            if current is None or current.get("owner_id") != self.owner_id:
                self._held = False
                raise LeaseHeld("Generation lease ownership was lost")
            atomic_write_json(
                self.path,
                self._payload(acquired_at or str(current["acquired_at"])),
            )

    def release(self) -> None:
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
        with self._lock:
            if not self._held:
                return
            current = self.current(self.path)
            if current is None:
                self._held = False
                return
            if current.get("owner_id") != self.owner_id:
                self._held = False
                raise LeaseHeld("Refusing to release a lease owned by another process")
            self.path.unlink()
            self._held = False

    @classmethod
    def force_release(cls, run_dir_a: str | Path) -> dict[str, Any] | None:
        """Manual break-glass release; the caller must display the returned holder."""
        path = resolve_local_path(run_dir_a) / "lease.json"
        holder = cls.current(path)
        if holder is None:
            return None
        path.unlink()
        return holder

    def __enter__(self) -> "RunLease":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()
