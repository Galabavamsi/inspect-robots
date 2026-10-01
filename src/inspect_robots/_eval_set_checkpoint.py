"""Atomic, single-writer manifests for explicitly resumable evaluation sets."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_robots.embodiment import Embodiment
from inspect_robots.errors import ConfigError
from inspect_robots.log import read_eval_log
from inspect_robots.policy import Policy
from inspect_robots.task import Task

_CHECKPOINT_VERSION = 1


def _identity(
    tasks: Sequence[Task],
    policy: Policy | str,
    embodiment: Embodiment | str,
    *,
    seed: int,
    log_dir: str,
    options: Mapping[str, object],
) -> dict[str, object]:
    """Describe run inputs whose change would make a saved scene unsafe to reuse."""
    try:
        task_specs = [
            {
                "name": task.name,
                "scenes": [asdict(scene) for scene in task.scenes],
                "epochs": asdict(task.epoch_spec),
                "scorers": [scorer.name for scorer in task.scorers],
                "max_steps": task.max_steps,
                "max_seconds": task.max_seconds,
                "metadata": task.metadata,
            }
            for task in tasks
        ]
        policy_spec: object = (
            {
                "name": policy.info.name,
                "checkpoint": policy.info.checkpoint,
                "config": asdict(policy.config),
            }
            if not isinstance(policy, str)
            else {"name": policy}
        )
        embodiment_spec: object = (
            {
                "name": embodiment.info.name,
                "environment_id": embodiment.info.environment_id,
                "environment_revision": embodiment.info.environment_revision,
                "control_hz": embodiment.info.control_hz,
                "is_simulated": embodiment.info.is_simulated,
            }
            if not isinstance(embodiment, str)
            else {"name": embodiment}
        )
        raw = {
            "tasks": task_specs,
            "policy": policy_spec,
            "embodiment": embodiment_spec,
            "seed": seed,
            "log_dir": str(Path(log_dir).resolve()),
            "options": dict(options),
        }
        # Normalize mappings/tuples to JSON values and reject objects or NaN
        # that cannot be compared reliably on a later invocation.
        return json.loads(json.dumps(raw, sort_keys=True, allow_nan=False))  # type: ignore[no-any-return]
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise ConfigError(f"checkpoint identity must be JSON serializable: {exc}") from exc


@dataclass
class _Manifest:
    """A validated checkpoint held under its sibling writer lock."""

    path: Path
    identity: dict[str, object]
    attempts: list[dict[str, object]] = field(default_factory=list)
    aggregates: dict[str, str] = field(default_factory=dict)

    def attempt_log_path(self, entry: Mapping[str, object]) -> Path:
        """Resolve and validate a referenced immutable attempt log."""
        try:
            index = entry["task_index"]
            scene_ids = entry["scene_ids"]
            relative = entry["log"]
            tasks = self.identity["tasks"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not isinstance(tasks, list)
                or index < 0
                or index >= len(tasks)
                or not isinstance(scene_ids, list)
                or not all(isinstance(value, str) for value in scene_ids)
                or not isinstance(relative, str)
                or Path(relative).is_absolute()
            ):
                raise ValueError("invalid attempt entry")
            path = (self.path.parent / relative).resolve()
            log_dir = Path(str(self.identity["log_dir"]))
            if not path.is_relative_to(log_dir) or not path.is_file():
                raise ValueError("attempt log is outside log_dir or missing")
            log = read_eval_log(str(path))
            task_spec = tasks[index]
            if not isinstance(task_spec, dict) or log.eval.task != task_spec["name"]:
                raise ValueError("attempt task does not match checkpoint")
            if log.eval.seed != self.identity["seed"]:
                raise ValueError("attempt seed does not match checkpoint")
            if any(sample.scene_id not in scene_ids for sample in log.samples):
                raise ValueError("attempt scenes do not match checkpoint")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ConfigError(f"invalid checkpoint attempt: {exc}") from exc
        return path

    def add_attempt(self, task_index: int, scene_ids: list[str], log_path: Path) -> None:
        """Publish an attempt only after its referenced log is readable."""
        entry: dict[str, object] = {
            "task_index": task_index,
            "scene_ids": scene_ids,
            "log": os.path.relpath(log_path, self.path.parent),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.attempt_log_path(entry)
        self.attempts.append(entry)
        try:
            self.write_atomic()
        except Exception:
            self.attempts.pop()
            raise

    def set_aggregate(self, task_index: int, log_path: Path) -> None:
        """Point to the latest aggregate without altering earlier attempt logs."""
        prior = self.aggregates.get(str(task_index))
        self.aggregates[str(task_index)] = os.path.relpath(log_path, self.path.parent)
        try:
            self.write_atomic()
        except Exception:
            if prior is None:
                del self.aggregates[str(task_index)]
            else:
                self.aggregates[str(task_index)] = prior
            raise

    def write_atomic(self) -> None:
        """Replace the manifest only after a synced temporary file is complete."""
        tmp = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        payload = {
            "version": _CHECKPOINT_VERSION,
            "identity": self.identity,
            "attempts": self.attempts,
            "aggregates": self.aggregates,
        }
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)


def _checkpoint_seed(path: Path) -> int | None:
    """Read an existing seed before identity construction; open revalidates it."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = data["identity"]["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed is not an integer")
        return seed
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ConfigError(f"invalid checkpoint seed: {exc}") from exc


@contextmanager
def _open_checkpoint(path: Path, identity: dict[str, object]) -> Iterator[_Manifest]:
    """Open a matching checkpoint with exclusive ownership until exit."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f"{path.name}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ConfigError(f"checkpoint lock exists: {lock}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        if path.exists():
            try:
                data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
                if data["version"] != _CHECKPOINT_VERSION:
                    raise ValueError("unsupported checkpoint schema version")
                if data["identity"] != identity:
                    raise ConfigError("checkpoint identity differs from this evaluation set")
                attempts = data["attempts"]
                aggregates = data.get("aggregates", {})
                if not isinstance(attempts, list) or not isinstance(aggregates, dict):
                    raise ValueError("invalid checkpoint entries")
                manifest = _Manifest(path, identity, attempts, aggregates)
                for entry in manifest.attempts:
                    manifest.attempt_log_path(entry)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise ConfigError(f"invalid checkpoint: {exc}") from exc
        else:
            manifest = _Manifest(path, identity)
            manifest.write_atomic()
        yield manifest
    finally:
        lock.unlink()
