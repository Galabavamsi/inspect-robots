"""Resumable evaluation set behavior with the deterministic CubePick world."""

from __future__ import annotations

from pathlib import Path

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
