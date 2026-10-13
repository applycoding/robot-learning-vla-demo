# Demonstration dataset specification

SimReach stores scripted planar-arm demonstrations in HDF5, with a sidecar
`meta/info.json` that uses LeRobot-style feature names. The validator recomputes
kinematics, rewards, and success from the recorded joints. Stored labels are
accepted only when they match that ground truth.

## Layout

```
dataset/
  reach_demos.h5
  meta/info.json
```

`info.json` records `codebase_version`, robot type, fps, episode and frame
counts, quality counts, and a `features` map. The episode payloads themselves
are HDF5 groups rather than parquet and mp4. A later LeRobot export would write
the same fields (`observation.state`, `observation.image`, `observation.language`,
`action`) without changing their meaning.

## File-level attributes

| Attribute | Value |
| --- | --- |
| `schema_version` | `1.0.0` |
| `dt` | `0.1` seconds |
| `max_joint_vel` | `2.0` rad/s |
| `l1`, `l2` | `1.0` m (link lengths) |
| `success_threshold` | `0.12` m |
| `success_bonus` | `5.0` |
| `image_size` | pixels on a side, default `64` |
| `fps` | `10` |

`instruction_vocab` is a string dataset:

- `reach the red target` (id 0)
- `reach the blue target` (id 1)

## Episode group `/episodes/{index:06d}`

State arrays include the terminal state. Action and reward do not.

| Dataset | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `q` | `(T+1, 2)` | float32 | Joint angles in radians, before each action, plus the final state |
| `q_dot` | `(T+1, 2)` | float32 | Joint velocity (rad/s). The first row is zero |
| `action` | `(T, 2)` | float32 | Normalized joint-velocity command in `[-1, 1]` |
| `reward` | `(T,)` | float32 | Reward of the transition into `q[t+1]` |
| `image` | `(T+1, H, W, 3)` | uint8 | Top-down RGB frame of `q[t]`. No instruction text |
| `target_red` | `(2,)` | float32 | Red marker position in the plane |
| `target_blue` | `(2,)` | float32 | Blue marker position in the plane |

Attributes:

| Attribute | Meaning |
| --- | --- |
| `instruction` | Canonical string from the vocabulary |
| `instruction_id` | `0` red or `1` blue. Must match `instruction` |
| `quality` | `gold`, `noisy`, or `mislabeled` |
| `tracked_color` | Color the scripted controller was aiming at |
| `success` | Whether any post-action state entered the success ball of the **instructed** target |
| `seed` | Environment seed |
| `episode_return` | Sum of `reward` |

### Integrator

```
q_dot = clip(action, -1, 1) * max_joint_vel
q_next = q + q_dot * dt
```

The validator recomputes every `q[t+1]` and `q_dot[t+1]` from `q[t]` and `action[t]`
and rejects the episode when the absolute error exceeds `1e-4`.

### Forward kinematics

```
ee_x = l1 cos q1 + l2 cos(q1 + q2)
ee_y = l1 sin q1 + l2 sin(q1 + q2)
```

End-effector position is not stored. It is always recomputed.

### Reward and success

Distance is the Euclidean distance from the end effector to the **instructed**
target. A state is a success when `distance < 0.12`.

```
reward = exp(-3 * distance) + (5.0 if success else 0)
```

Episode `success` is true if any state after an action is a success. The
environment also ends the episode on that step. The validator recomputes the
reward and the episode flag and rejects disagreements above `1e-4` (reward) or
any boolean mismatch (success).

### State vector used by behavior cloning and PPO

Built at train time from `q`, `q_dot`, the targets, and `instruction_id`. It is
not a separate dataset because it is a pure function of those fields.

| Index | Name |
| --- | --- |
| 0–3 | `sin q1`, `cos q1`, `sin q2`, `cos q2` |
| 4–5 | joint velocity |
| 6–7 | end-effector position |
| 8–9 | instructed target minus end effector |
| 10–11 | instruction one-hot |

The vision-language policy does not receive target coordinates. It receives the
image, a 6-D proprioceptive prefix (the joint features above), and the
instruction string or id.

## Quality tiers

All tiers share the same scenes, language vocabulary, and integrator. They differ
in the controller that produced `action`.

| Tier | Controller | Label rule |
| --- | --- | --- |
| `gold` | Damped least-squares expert aimed at the instructed color | Must succeed, language must match the tracked color, mean absolute action delta ≤ 0.45, final end effector closer to the instructed target than to the other one |
| `noisy` | Same expert, then corrupted: with probability 0.45 the action is uniform in `[-1, 1]`; otherwise it is `0.35 * expert + N(0, 0.85)` and clipped | Language must still match the tracked color. Success is not required |
| `mislabeled` | Expert aimed at the **other** color, while `instruction` names the original color | `tracked_color` must differ from `instruction_id`. This is an intentional negative control for annotation error |

`gold` is the only tier that has to pass the gold standard for `validate` to
exit 0. Structural failures (bad shapes, non-finite values, actions outside
`[-1, 1]`, kinematic or reward mismatch, blank images, unknown language) fail
the file regardless of tier.

## Images

Frames are schematic, not natural photos: a dark background, a gray arm, and
solid red and blue markers. The instructed target is not highlighted in stored
frames, so a vision policy cannot read the command off the image. GIFs in
`results/` draw a ring and a caption for people; those pixels are not in the dataset.

Images must be `uint8` RGB with the file's `image_size` and a standard deviation
above 2, which rejects blank frames.

## Validator report

`python -m simreach validate DATA.h5 --report validation.json` prints episode
counts, success, length, and smoothness by tier, plus the maximum kinematic
error. The JSON report repeats those stats and lists up to 30 flagged episodes.
