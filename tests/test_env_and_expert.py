"""Simulator, kinematics, and scripted expert."""

from __future__ import annotations

import numpy as np

from simreach.constants import OBS_DIM, Quality
from simreach.dataset import generate_episode
from simreach.env import PlanarReachEnv
from simreach.evaluate import run_episode
from simreach.expert import expert_action
from simreach.kinematics import forward_kinematics, integrate, jacobian
from simreach.task import other_color


def _expert(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
    del obs
    return expert_action(env.q, env.goal)


def test_jacobian_matches_finite_difference() -> None:
    q = np.array([0.4, 1.1])
    analytic = jacobian(q)
    eps = 1e-6
    numeric = np.zeros((2, 2))
    base = forward_kinematics(q)
    for index in range(2):
        delta = np.zeros(2)
        delta[index] = eps
        numeric[:, index] = (forward_kinematics(q + delta) - base) / eps
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_step_matches_integrator_and_is_deterministic() -> None:
    env = PlanarReachEnv()
    obs_a, info_a = env.reset(seed=3)
    q_before = env.q.copy()
    action = np.array([0.2, -0.4], dtype=np.float32)
    expected_q, expected_qd = integrate(q_before, action)
    obs_b, _reward, _term, _trunc, info_b = env.step(action)
    assert obs_a.shape == (OBS_DIM,)
    assert obs_b.shape == (OBS_DIM,)
    assert np.allclose(env.q, expected_q, atol=1e-8)
    assert np.allclose(env.q_dot, expected_qd, atol=1e-8)
    assert info_a["instruction"] in ("reach the red target", "reach the blue target")
    assert env.observation_space.contains(obs_b)
    assert env.action_space.contains(action)

    again = PlanarReachEnv()
    obs_c, info_c = again.reset(seed=3)
    assert np.allclose(obs_a, obs_c)
    assert info_a["instruction_id"] == info_c["instruction_id"]
    frame = env.render()
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8
    assert float(frame.std()) > 2.0


def test_expert_reaches_instructed_target() -> None:
    env = PlanarReachEnv()
    successes = [run_episode(env, _expert, seed).success for seed in range(10)]
    assert sum(successes) >= 9


def test_mislabeled_expert_tracks_the_other_color() -> None:
    closer = 0
    for seed in range(6):
        episode = generate_episode(seed + 50, Quality.MISLABELED, image_size=32)
        assert episode.tracked_color == other_color(episode.instruction_id)
        ee = forward_kinematics(episode.q[-1])
        red = episode.target_red.astype(np.float64)
        blue = episode.target_blue.astype(np.float64)
        dist_tracked = np.linalg.norm(ee - (red if episode.tracked_color == 0 else blue))
        dist_instructed = np.linalg.norm(ee - (blue if episode.tracked_color == 0 else red))
        closer += int(dist_tracked < dist_instructed)
    assert closer >= 5
