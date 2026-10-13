"""HDF5 dataset round-trip and the quality validator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simreach.cli import main
from simreach.constants import Quality
from simreach.dataset import generate_episode, load_episodes, write_dataset
from simreach.validate import audit_episode, validate_episodes, validate_file


def test_hdf5_roundtrip_and_sidecar(tmp_path: Path) -> None:
    episodes = [
        generate_episode(11, Quality.GOLD, image_size=32),
        generate_episode(12, Quality.NOISY, image_size=32),
        generate_episode(13, Quality.MISLABELED, image_size=32),
    ]
    path = tmp_path / "reach_demos.h5"
    write_dataset(path, episodes, image_size=32)
    info = path.parent / "meta" / "info.json"
    assert info.is_file()
    loaded = load_episodes(path, load_images=True)
    assert len(loaded) == 3
    assert loaded[0].quality is Quality.GOLD
    assert np.allclose(loaded[0].q, episodes[0].q)
    assert np.array_equal(loaded[0].image, episodes[0].image)
    assert loaded[2].instruction == episodes[2].instruction
    report = validate_file(path)
    assert report.structural_failures == 0
    assert report.passed


def test_validator_rejects_broken_kinematics() -> None:
    episode = generate_episode(21, Quality.GOLD, image_size=32)
    clean = audit_episode(episode, 0, image_size=32)
    assert clean.structural_ok
    assert not clean.quality_issues
    episode.q = episode.q.copy()
    episode.q[1, 0] += 0.5
    broken = audit_episode(episode, 0, image_size=32)
    assert not broken.structural_ok
    assert any("kinematic" in issue for issue in broken.structural)


def test_cli_generate_and_validate(tmp_path: Path) -> None:
    path = tmp_path / "demos.h5"
    main(["generate", "--out", str(path), "--gold", "2", "--noisy", "1", "--mislabeled", "1", "--image-size", "32"])
    with pytest.raises(SystemExit) as caught:
        main(["validate", str(path), "--report", str(tmp_path / "validation.json")])
    assert caught.value.code == 0
    assert (tmp_path / "validation.json").is_file()
    report = validate_episodes(load_episodes(path), image_size=32)
    assert report.by_quality["gold"]["success_rate"] == 1.0
