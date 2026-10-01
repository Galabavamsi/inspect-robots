"""Resumable evaluation set behavior with the deterministic CubePick world."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from inspect_robots import eval
from inspect_robots.errors import PolicyError
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Epochs, Task
from inspect_robots.types import ActionChunk, Observation


class _OneTransientFailure(ScriptedPolicy):
    """Fail one inference transiently, then follow the scripted trajectory."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def act(self, observation: Observation) -> ActionChunk:
        if not self.failed:
            self.failed = True
            raise PolicyError("server restarted", retryable=True)
        return super().act(observation)


def test_real_attempt_records_scene_local_retryability_and_error_count(tmp_path: Path) -> None:
    """A failed epoch must retain its retry marker even if the next epoch scores."""
    task = Task(
        name="two-epochs",
        scenes=[Scene(id="s0", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
        epochs=Epochs(count=2),
    )

    (log,) = eval(task, _OneTransientFailure(), CubePickEmbodiment(), log_dir=str(tmp_path))

    assert log.samples[0].status == "error"
    assert log.samples[0].errored_trials == 1
    assert log.samples[0].retryable_error is True
    assert log.results.errored_trials == 1
    assert len(log.samples[0].epochs) == 2


def test_real_attempt_marks_frame_source_on_scene(tmp_path: Path) -> None:
    """A scene retains the attempt frame root needed after aggregate merging."""
    task = Task(
        name="one-scene",
        scenes=[Scene(id="s0", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )

    (log,) = eval(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        store_frames=True,
    )

    assert log.samples[0].frames_dir == log.stats.frames_dir
    assert log.samples[0].frames_dir is not None


def _task() -> Task:
    """Use a stable two-scene task for checkpoint identity tests."""
    return Task(
        name="resume-demo",
        scenes=[Scene(id="s0", instruction="reach"), Scene(id="s1", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )


def test_checkpoint_reopens_with_attempt_and_rejects_changed_identity(tmp_path: Path) -> None:
    """A resumed call may reuse only a matching scene declaration and attempt log."""
    from inspect_robots._eval_set_checkpoint import _identity, _open_checkpoint
    from inspect_robots.errors import ConfigError

    task = _task()
    policy = ScriptedPolicy()
    embodiment = CubePickEmbodiment()
    log_dir = tmp_path / "logs"
    checkpoint = tmp_path / "run.checkpoint.json"
    identity = _identity([task], policy, embodiment, seed=17, log_dir=str(log_dir), options={})
    (attempt,) = eval(task, policy, embodiment, log_dir=str(log_dir), seed=17)
    attempt_path = next(log_dir.glob("*.json"))
    assert attempt.samples[0].status == "success"

    with _open_checkpoint(checkpoint, identity) as manifest:
        manifest.add_attempt(0, ["s0", "s1"], attempt_path)
    with _open_checkpoint(checkpoint, identity) as manifest:
        assert manifest.attempts[0]["scene_ids"] == ["s0", "s1"]
        assert manifest.attempt_log_path(manifest.attempts[0]) == attempt_path

    changed = Task(
        name="resume-demo",
        scenes=[Scene(id="s0", instruction="different"), Scene(id="s1", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )
    changed_identity = _identity(
        [changed], policy, embodiment, seed=17, log_dir=str(log_dir), options={}
    )
    with (
        pytest.raises(ConfigError, match="checkpoint identity"),
        _open_checkpoint(checkpoint, changed_identity),
    ):
        pass


def test_checkpoint_rejects_non_json_scene_before_creation(tmp_path: Path) -> None:
    """Opaque scene metadata cannot produce a reliable resume identity."""
    from inspect_robots._eval_set_checkpoint import _identity
    from inspect_robots.errors import ConfigError

    task = Task(
        name="opaque",
        scenes=[Scene(id="s", instruction="reach", metadata={"opaque": object()})],
        scorer=success_at_end(),
        max_steps=30,
    )
    with pytest.raises(ConfigError, match="JSON"):
        _identity(
            [task],
            ScriptedPolicy(),
            CubePickEmbodiment(),
            seed=17,
            log_dir=str(tmp_path),
            options={},
        )


def test_checkpoint_lock_blocks_second_writer(tmp_path: Path) -> None:
    """A live writer must prevent another process from mutating the manifest."""
    from inspect_robots._eval_set_checkpoint import _identity, _open_checkpoint
    from inspect_robots.errors import ConfigError

    identity = _identity(
        [_task()],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        seed=17,
        log_dir=str(tmp_path / "logs"),
        options={},
    )
    checkpoint = tmp_path / "run.json"
    with (
        _open_checkpoint(checkpoint, identity),
        pytest.raises(ConfigError, match="lock"),
        _open_checkpoint(checkpoint, identity),
    ):
        pass


def _saved_attempt(tmp_path: Path, *, seed: int = 17) -> Path:
    """Write a real two-scene EvalLog for manifest validation."""
    log_dir = tmp_path / "logs"
    eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(log_dir), seed=seed)
    return next(log_dir.glob("*.json"))


def _checkpoint_identity(tmp_path: Path) -> dict[str, object]:
    """Build the matching identity for a real saved attempt."""
    from inspect_robots._eval_set_checkpoint import _identity

    return _identity(
        [_task()],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        seed=17,
        log_dir=str(tmp_path / "logs"),
        options={},
    )


def test_checkpoint_seed_reads_existing_and_rejects_invalid(tmp_path: Path) -> None:
    """The recorded seed survives restart, while corrupt seed data fails closed."""
    from inspect_robots._eval_set_checkpoint import _checkpoint_seed, _open_checkpoint
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    assert _checkpoint_seed(checkpoint) is None
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)):
        pass
    assert _checkpoint_seed(checkpoint) == 17
    payload = json.loads(checkpoint.read_text())
    payload["identity"]["seed"] = True
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="seed"):
        _checkpoint_seed(checkpoint)
    checkpoint.write_text("not JSON")
    with pytest.raises(ConfigError, match="seed"):
        _checkpoint_seed(checkpoint)


def test_checkpoint_rejects_missing_or_untrusted_attempt_log(tmp_path: Path) -> None:
    """A manifest cannot point outside the run log directory or to a missing log."""
    from inspect_robots._eval_set_checkpoint import _open_checkpoint
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    log_path = _saved_attempt(tmp_path)
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)) as manifest:
        with pytest.raises(ConfigError, match="attempt"):
            manifest.add_attempt(0, ["s0", "s1"], tmp_path / "missing.json")
        with pytest.raises(ConfigError, match="attempt"):
            manifest.add_attempt(0, ["s0", "s1"], tmp_path / "outside.json")
        manifest.add_attempt(0, ["s0", "s1"], log_path)
    log_path.unlink()
    with (
        pytest.raises(ConfigError, match="attempt"),
        _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)),
    ):
        pass


def test_checkpoint_rejects_tampered_attempt_metadata(tmp_path: Path) -> None:
    """A forged scene, seed, task, or path cannot be reused as prior work."""
    from inspect_robots._eval_set_checkpoint import _open_checkpoint
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    log_path = _saved_attempt(tmp_path)
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)) as manifest:
        manifest.add_attempt(0, ["s0", "s1"], log_path)
        entry = manifest.attempts[0]
        bad_entries = [
            {**entry, "task_index": True},
            {**entry, "task_index": -1},
            {**entry, "task_index": 1},
            {**entry, "scene_ids": "s0"},
            {**entry, "scene_ids": [0]},
            {**entry, "scene_ids": ["other"]},
            {**entry, "log": 0},
            {**entry, "log": str(log_path)},
            {"scene_ids": ["s0", "s1"], "log": entry["log"]},
        ]
        for bad in bad_entries:
            with pytest.raises(ConfigError, match="attempt"):
                manifest.attempt_log_path(bad)

        data = json.loads(log_path.read_text())
        for field, value in [("task", "other"), ("seed", 999)]:
            changed = json.loads(json.dumps(data))
            changed["eval"][field] = value
            log_path.write_text(json.dumps(changed))
            with pytest.raises(ConfigError, match="attempt"):
                manifest.attempt_log_path(entry)
        log_path.write_text(json.dumps(data))


def test_checkpoint_atomic_failure_retains_previous_manifest(tmp_path: Path) -> None:
    """Failed publication rolls back memory and leaves the old JSON readable."""
    from inspect_robots._eval_set_checkpoint import _open_checkpoint

    checkpoint = tmp_path / "run.json"
    log_path = _saved_attempt(tmp_path)
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)) as manifest:
        before = checkpoint.read_text()
        with (
            patch("inspect_robots._eval_set_checkpoint.os.replace", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            manifest.add_attempt(0, ["s0", "s1"], log_path)
        assert manifest.attempts == []
        assert checkpoint.read_text() == before
        assert not list(tmp_path.glob("*.tmp"))
        manifest.add_attempt(0, ["s0", "s1"], log_path)
        manifest.set_aggregate(0, log_path)
        previous_aggregate = manifest.aggregates["0"]
        with (
            patch("inspect_robots._eval_set_checkpoint.os.replace", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            manifest.set_aggregate(0, tmp_path / "other.json")
        assert manifest.aggregates["0"] == previous_aggregate
        with (
            patch("inspect_robots._eval_set_checkpoint.os.replace", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            manifest.set_aggregate(1, tmp_path / "other.json")
        assert "1" not in manifest.aggregates
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)) as manifest:
        assert len(manifest.attempts) == 1
        assert manifest.aggregates["0"] == previous_aggregate


def test_checkpoint_invalid_schema_and_entries_fail_closed(tmp_path: Path) -> None:
    """Corrupt or unknown manifests fail before they can schedule work."""
    from inspect_robots._eval_set_checkpoint import _open_checkpoint
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    identity = _checkpoint_identity(tmp_path)
    with _open_checkpoint(checkpoint, identity):
        pass
    original = json.loads(checkpoint.read_text())
    variants = [
        {**original, "version": 999},
        {**original, "attempts": "bad"},
        {**original, "aggregates": []},
        {**original, "attempts": [{}]},
        {"version": 1, "identity": identity},
    ]
    for variant in variants:
        checkpoint.write_text(json.dumps(variant))
        with pytest.raises(ConfigError, match="checkpoint"), _open_checkpoint(checkpoint, identity):
            pass
        assert not checkpoint.with_name("run.json.lock").exists()
    checkpoint.write_text(json.dumps(original))


def test_merge_keeps_first_complete_zero_score_and_task_order(tmp_path: Path) -> None:
    """A completed scene is retained even if its score is zero and later logs differ."""
    from dataclasses import replace

    from inspect_robots._eval_set_merge import _merge_task, _pending_scenes

    task = _task()
    (original,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), seed=17)
    zero = replace(
        original.samples[0],
        reduced={"success_at_end": 0.0},
        epochs=({"success_at_end": 0.0},),
    )
    first = replace(
        original,
        samples=(original.samples[1], zero),
        stats=replace(original.stats, duration_s=1.25, total_steps=10),
    )
    second = replace(original, stats=replace(original.stats, duration_s=2.5, total_steps=20))

    merged = _merge_task(task, [(first, "attempt-1.json"), (second, "attempt-2.json")])

    assert [sample.scene_id for sample in merged.samples] == ["s0", "s1"]
    assert merged.samples[0].reduced == {"success_at_end": 0.0}
    assert merged.results.metrics == {"success_at_end": 0.5}
    assert merged.results.total_trials == 2
    assert merged.stats.duration_s == 3.75
    assert merged.stats.total_steps == 30
    assert merged.stats.mean_inference_latency_s is None
    assert merged.stats.frames_dir is None
    assert merged.source_logs == ("attempt-1.json", "attempt-2.json")
    assert merged.status == "success"
    assert _pending_scenes(task, merged) == []


def test_merge_replaces_partial_scene_and_keeps_unreached_pending(tmp_path: Path) -> None:
    """Incomplete scenes use the latest record and stay pending until fully rerun."""
    from dataclasses import replace

    from inspect_robots._eval_set_merge import _merge_task, _pending_scenes

    task = _task()
    (original,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), seed=17)
    failed = replace(
        original.samples[0],
        status="error",
        epochs=({},),
        reduced={},
        error="PolicyError: offline",
        errored_trials=1,
        retryable_error=True,
    )
    first = replace(original, status="error", samples=(failed,), error="offline")
    incomplete = _merge_task(task, [(first, "first.json")])
    assert incomplete.status == "error"
    assert incomplete.results.errored_trials == 1
    assert [scene.id for scene in _pending_scenes(task, incomplete)] == ["s0", "s1"]

    second = replace(original, samples=(original.samples[0],), error=None)
    merged = _merge_task(task, [(first, "first.json"), (second, "second.json")])
    assert merged.samples[0].status == "success"
    assert merged.results.errored_trials == 0
    assert [scene.id for scene in _pending_scenes(task, merged)] == ["s1"]


def test_merge_uses_attempt_frame_root_for_selected_scene(tmp_path: Path) -> None:
    """A selected scene remembers the source even when only the log has the root."""
    from dataclasses import replace

    from inspect_robots._eval_set_merge import _merge_task

    task = _task()
    (original,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), seed=17)
    first = replace(
        original,
        samples=(original.samples[0],),
        stats=replace(original.stats, frames_dir="first-frames"),
    )
    second = replace(
        original,
        samples=(original.samples[1],),
        stats=replace(original.stats, frames_dir="second-frames"),
    )
    merged = _merge_task(task, [(first, "first.json"), (second, "second.json")])
    assert [sample.frames_dir for sample in merged.samples] == [
        "first-frames",
        "second-frames",
    ]


def test_merge_requires_real_attempts_and_rejects_foreign_scenes(tmp_path: Path) -> None:
    """A merge cannot fabricate provenance or accept a scene outside the task."""
    from dataclasses import replace

    from inspect_robots._eval_set_merge import _merge_task
    from inspect_robots.errors import ConfigError

    task = _task()
    with pytest.raises(ConfigError, match="without attempt"):
        _merge_task(task, [])
    (original,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), seed=17)
    foreign = replace(original.samples[0], scene_id="foreign")
    with pytest.raises(ConfigError, match="unknown scene"):
        _merge_task(task, [(replace(original, samples=(foreign,)), "foreign.json")])
