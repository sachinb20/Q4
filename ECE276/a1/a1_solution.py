"""ECE276C Assignment 1 -- Robot Kinematics and Simple Control.

Usage (Python with numpy, matplotlib, pybullet, imageio and Pillow):
    python a1_solution.py            # run Q3, Q5, Q6 headless, write results/
    python a1_solution.py --part 5   # one part only
    python a1_solution.py --gui      # also open the PyBullet GUI while simulating
"""
import argparse
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont

from arm import fk, ik, jacobian, pinv_and_null, wrap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
DT = 1. / 240.

CONFIGS = np.array([[0, 0, np.pi / 2],
                    [0, np.pi, 0],
                    [np.pi / 2, np.pi / 2, np.pi / 4],
                    [np.pi / 3, np.pi / 2, 0]])
LOCATIONS = np.array([[1.2, 0.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.5, -0.5, 0.0],
                      [0.4, 0.0, 0.0],
                      [1.2, 0.0, 0.0]])    # 1 -> 2 -> 3 -> 4 -> 1
TOTAL_TIME = 10.0
GIF_FPS = 20
FONT = ImageFont.load_default(size=16)


# ----------------------------------------------------------------------------- simulation
def make_sim(gui=False):
    """Same setup as the starter code; returns robot id, movable joints, ee link."""
    if p.isConnected():
        p.disconnect()
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0., 0., -9.81)
    p.setTimeStep(DT)
    # lowered slightly so the arm (centred on z = 0) is not half buried in the floor
    p.loadURDF('plane.urdf', [0, 0, -0.06])
    robot = p.loadURDF(os.path.join(HERE, 'three-link-robot.urdf'), useFixedBase=True)

    movable_joints, ee_link = [], None
    for idx in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, idx)
        if info[2] != p.JOINT_FIXED:
            movable_joints.append(idx)
        if info[12].decode('utf-8') == 'end_effector':
            ee_link = idx
    # distinct colours per link so overlapping links can be told apart
    for joint, rgba in zip(movable_joints, [[0.85, 0.1, 0.1, 1], [0.95, 0.55, 0.0, 1],
                                            [0.55, 0.1, 0.75, 1]]):
        p.changeVisualShape(robot, joint, rgbaColor=rgba)
    # disable the default velocity motors so we fully own the joints
    p.setJointMotorControlArray(robot, movable_joints, p.VELOCITY_CONTROL, forces=[0.] * 3)
    return robot, movable_joints, ee_link


def set_q(robot, joints, q):
    for j, qi in zip(joints, q):
        p.resetJointState(robot, j, qi, 0.)


def get_q(robot, joints):
    states = p.getJointStates(robot, joints)
    return np.array([s[0] for s in states]), np.array([s[1] for s in states])


def ee_pose(robot, ee_link):
    """Measured end-effector [x, y, theta] from PyBullet."""
    st = p.getLinkState(robot, ee_link, computeForwardKinematics=True)
    yaw = p.getEulerFromQuaternion(st[5])[2]
    return np.array([st[4][0], st[4][1], yaw])


class TopCamera:
    """Top-down camera. The z = 0 plane maps affinely to pixels, so world xy
    points can be drawn directly onto the rendered image."""

    def __init__(self, center, half_height, size=(560, 560), fov=50.):
        self.W, self.H = size
        self.c = np.asarray(center, float)
        height = half_height / np.tan(np.radians(fov / 2))
        self.hh = half_height
        self.hw = half_height * self.W / self.H
        self.view = p.computeViewMatrix([self.c[0], self.c[1], height],
                                        [self.c[0], self.c[1], 0.], [0, 1, 0])
        self.proj = p.computeProjectionMatrixFOV(fov, self.W / self.H, 0.05, height + 2)

    def render(self):
        img = p.getCameraImage(self.W, self.H, self.view, self.proj,
                               renderer=p.ER_TINY_RENDERER, shadow=0)[2]
        return np.reshape(img, (self.H, self.W, 4))[:, :, :3].astype(np.uint8)

    def to_px(self, xy):
        xy = np.atleast_2d(xy)
        u = (xy[:, 0] - self.c[0]) / self.hw * self.W / 2 + self.W / 2
        v = self.H / 2 - (xy[:, 1] - self.c[1]) / self.hh * self.H / 2
        return list(zip(u.tolist(), v.tolist()))

    def extent(self):
        return [self.c[0] - self.hw, self.c[0] + self.hw, self.c[1] - self.hh, self.c[1] + self.hh]


# ----------------------------------------------------------------------------- Q3
def part3(gui=False):
    robot, joints, ee = make_sim(gui)
    cam = TopCamera(center=[0.25, 0.32], half_height=0.68, size=(700, 450))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))
    print('\nQ3: FK check (analytic vs PyBullet)')
    for k, (q, ax) in enumerate(zip(CONFIGS, axes.ravel()), start=1):
        set_q(robot, joints, q)
        img = cam.render()
        imageio.imwrite(os.path.join(OUT, f'q3_config{k}.png'), img)

        pose_fk, pose_pb = fk(q), ee_pose(robot, ee)
        print(f'  config {k}: q = {np.round(q, 3)}  fk = {np.round(pose_fk, 4)}  '
              f'pybullet = {np.round(pose_pb, 4)}  '
              f'|dpos| = {np.linalg.norm(pose_fk[:2] - pose_pb[:2]):.1e}  '
              f'dtheta = {abs(wrap(pose_fk[2] - pose_pb[2])):.1e}')

        ax.imshow(img, extent=cam.extent())
        ax.set_title(f'Config {k}: q = ({_pi_str(q[0])}, {_pi_str(q[1])}, {_pi_str(q[2])})')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        if k == 2:
            # Links 1 and 2 coincide exactly in this configuration. Keep the
            # unchanged full render and add a labelled cutaway of the same pose.
            p.changeVisualShape(robot, joints[1], rgbaColor=[0.95, 0.55, 0., 0.])
            cutaway_cam = TopCamera([0.15, 0.], 0.17, size=(600, 170))
            cutaway = cutaway_cam.render()
            p.changeVisualShape(robot, joints[1], rgbaColor=[0.95, 0.55, 0., 1.])
            imageio.imwrite(os.path.join(OUT, 'q3_config2_cutaway.png'), cutaway)
            inset = ax.inset_axes([0.08, 0.57, 0.84, 0.23])
            inset.imshow(cutaway)
            inset.set_title('Same pose: link 2 hidden to expose link 1', fontsize=8)
            inset.set_xticks([])
            inset.set_yticks([])
    axes[0, 0].plot([], [], color=(0.85, 0.1, 0.1), lw=4, label='link 1')
    axes[0, 0].plot([], [], color=(0.95, 0.55, 0.0), lw=4, label='link 2')
    axes[0, 0].plot([], [], color=(0.55, 0.1, 0.75), lw=4, label='link 3')
    axes[0, 0].legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'q3_configs.png'), dpi=120)
    plt.close(fig)
    p.disconnect()


def _pi_str(a):
    names = {0: '0', np.pi: 'π', np.pi / 2: 'π/2', np.pi / 3: 'π/3', np.pi / 4: 'π/4'}
    return next((s for v, s in names.items() if np.isclose(a, v)), f'{a:.2f}')


# ----------------------------------------------------------------------------- trajectory
def segment_durations():
    """Four equal moves, with each waypoint on a simulation time step."""
    return np.full(len(LOCATIONS) - 1, TOTAL_TIME / (len(LOCATIONS) - 1))


def profile(tau):
    """C2 move-and-stop profile; position, velocity and acceleration are continuous.

    Lecture 4's polynomial boundary-value construction, adding zero endpoint
    acceleration: s = 10*tau^3 - 15*tau^4 + 6*tau^5. Returns s, s', s''.
    """
    tau = np.clip(tau, 0., 1.)
    s = tau**3 * (10. + tau * (-15. + 6. * tau))
    ds = 30. * tau**2 * (1. - tau)**2
    dds = 60. * tau * (1. - tau) * (1. - 2. * tau)
    return s, ds, dds


def reference(t, durations=None):
    """Straight-line reference pose and velocity at time t."""
    durations = segment_durations() if durations is None else durations
    starts = np.concatenate([[0.], np.cumsum(durations)])
    i = int(np.clip(np.searchsorted(starts, t, side='right') - 1, 0, len(durations) - 1))
    T = durations[i]
    s, ds, _ = profile(np.clip((t - starts[i]) / T, 0., 1.))
    a, b = LOCATIONS[i], LOCATIONS[i + 1]
    return a + s * (b - a), ds / T * (b - a)


# ----------------------------------------------------------------------------- GIF helpers
def draw_frame(img, cam, desired_xy, actual_xy, t, title, extra=''):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.line(cam.to_px(desired_xy), fill=(120, 120, 120), width=1)
    if len(actual_xy) > 1:
        d.line(cam.to_px(actual_xy), fill=(0, 170, 60), width=3)
    for k, (u, v) in enumerate(cam.to_px(LOCATIONS[:4, :2]), start=1):
        d.ellipse([u - 5, v - 5, u + 5, v + 5], outline=(0, 0, 0), width=2)
        d.text((u + 8, v - 16), str(k), fill=(0, 0, 0))
    d.rectangle([0, 0, cam.W, 24], fill=(255, 255, 255))
    d.text((6, 3), f'{title}   t = {t:5.2f} s   {extra}', fill=(0, 0, 0), font=FONT)
    return np.asarray(im)


def desired_path(n=400):
    ts = np.linspace(0, TOTAL_TIME, n)
    return np.array([reference(t)[0][:2] for t in ts])


def run_and_record(step_fn, robot, joints, ee, title, gif_name):
    """Shared loop: call step_fn(t) each physics step, log data, write GIF."""
    cam = TopCamera(center=[0.55, 0.0], half_height=0.85)
    n_steps = int(round(TOTAL_TIME / DT))
    every = int(round(1. / (GIF_FPS * DT)))
    log = {k: [] for k in ['t', 'x_des', 'x_meas', 'q', 'qdot']}
    frames, des_xy = [], desired_path()
    for k in range(n_steps + 1):
        t = k * DT
        x_des, _ = reference(t)
        x_meas = ee_pose(robot, ee)
        q, qdot = get_q(robot, joints)
        for key, val in zip(log, [t, x_des, x_meas, q, qdot]):
            log[key].append(val)
        if k % every == 0:
            act = np.array(log['x_meas'])[:, :2]
            frames.append(draw_frame(cam.render(), cam, des_xy, act, t, title,
                                     f'ee theta = {np.degrees(wrap(x_meas[2])):6.1f} deg'))
        if k < n_steps:
            step_fn(t, q, x_meas)
            p.stepSimulation()
    imageio.mimsave(os.path.join(OUT, gif_name), frames, duration=1000. / GIF_FPS, loop=0)
    log = {k: np.array(v) for k, v in log.items()}
    np.savez_compressed(os.path.join(OUT, gif_name.replace('.gif', '.npz')), **log)
    return log


def plot_results(log, fname, title):
    err = np.linalg.norm(log['x_meas'][:, :2] - log['x_des'][:, :2], axis=1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(log['x_meas'][:, 0], log['x_meas'][:, 1], color='tab:green', lw=3, label='measured')
    ax.plot(log['x_des'][:, 0], log['x_des'][:, 1], 'k--', lw=1, label='reference')
    for k, (x, y, _) in enumerate(LOCATIONS[:4], start=1):
        ax.plot(x, y, 'ko', ms=5)
        ax.annotate(str(k), (x, y), xytext=(7, 6), textcoords='offset points')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title + '\n' + r'$1\to2\to3\to4\to1$')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=120)
    plt.close(fig)
    return err


# ----------------------------------------------------------------------------- Q5
def part5(gui=False):
    """Open-loop setpoint control: the reference pose is converted to joint
    setpoints with analytic IK; the end-effector pose is never fed back."""
    robot, joints, ee = make_sim(gui)
    q_prev = ik(LOCATIONS[0], elbow=1.)
    set_q(robot, joints, q_prev)

    def step(t, q_meas, x_meas):
        nonlocal q_prev
        x_des, _ = reference(t)
        q_des = ik(x_des, elbow=1., q_near=q_prev)
        q_next = ik(reference(t + DT)[0], elbow=1., q_near=q_des)
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL,
                                    targetPositions=q_des.tolist(),
                                    targetVelocities=((q_next - q_des) / DT).tolist(),
                                    positionGains=[0.5] * 3, velocityGains=[1.] * 3,
                                    forces=[200.] * 3)
        q_prev = q_des

    log = run_and_record(step, robot, joints, ee, 'Q5 open-loop', 'q5_open_loop.gif')
    err = plot_results(log, 'q5_open_loop.png', 'Q5: open-loop setpoint control')
    th_err = np.abs(wrap(log['x_meas'][:, 2] - log['x_des'][:, 2]))
    print(f'\nQ5 open loop: max |pos err| = {err.max():.2e} m, '
          f'RMS = {np.sqrt(np.mean(err**2)):.2e} m, max |θ err| = {np.degrees(th_err.max()):.2e} deg')
    _print_setpoints(log)
    p.disconnect()
    return log


def _print_setpoints(log):
    starts = np.concatenate([[0.], np.cumsum(segment_durations())])
    for k, ts in enumerate(starts):
        i = int(round(ts / DT))
        speed = np.linalg.norm(jacobian(log['q'][i])[:2] @ log['qdot'][i])
        print(f'  t = {ts:5.2f} s  location {[1, 2, 3, 4, 1][k]}: '
              f'measured ee = {np.round(log["x_meas"][i], 4)}, speed = {speed:.2e} m/s')


# ----------------------------------------------------------------------------- Q6
K_NULL = 10.     # nullspace gradient gain


def null_gradient(q):
    """Gradient of U = 1/2 (wrap(q2)^2 + wrap(q3)^2), away from wrap cuts.

    The interior angle at a shared joint is alpha_i = pi - |wrap(q_{i+1})|.
    U penalises squared departures from the largest angle, pi. This is a local
    posture objective, not a claim of a global maximum of both angles.
    """
    qw = wrap(q)
    return np.array([0., qw[1], qw[2]])


def closed_loop_target(q, x_meas, t):
    """Lecture 4's measured-state, next-sample resolved-rate step plus nullspace.

    Correct the O(dt^2) error of the linearised step by repeated Cartesian
    corrections. No simulated joint state is reset during motion.
    """
    x_next, _ = reference(t + DT)
    if t == 0.:
        # At full extension J has rank one. IK selects the departure branch at
        # the first nonzero sample; the actual initial state is exactly q = 0.
        return ik(x_next, elbow=1., q_near=q)
    if t + DT >= TOTAL_TIME - 1e-12:
        # The final full-reach point has the unique straight-arm posture.
        return q + wrap(-q)
    J_dagger, N = pinv_and_null(jacobian(q)[:2])
    delta_x = x_next[:2] - x_meas[:2]
    q_next = q + J_dagger @ delta_x - DT * K_NULL * N @ null_gradient(q)
    # Retain measured Cartesian feedback while correcting finite-step error.
    goal = fk(q)[:2] + delta_x
    for _ in range(12):
        error = goal - fk(q_next)[:2]
        if np.linalg.norm(error) < 1e-10:
            break
        J_dagger, _ = pinv_and_null(jacobian(q_next)[:2])
        correction = J_dagger @ error
        # Limit Newton updates near full extension without changing the goal.
        q_next += correction * min(1., 0.2 / max(np.linalg.norm(correction), 1e-15))
    return q_next


def part6(gui=False):
    """Closed-loop position tracking with projected gradient descent on U(q)."""
    robot, joints, ee = make_sim(gui)
    set_q(robot, joints, np.zeros(3))

    def step(t, q_meas, x_meas):
        q_next = closed_loop_target(q_meas, x_meas, t)
        qdot = (q_next - q_meas) / DT
        p.setJointMotorControlArray(robot, joints, p.VELOCITY_CONTROL,
                                    targetVelocities=qdot.tolist(), forces=[500.] * 3)

    log = run_and_record(step, robot, joints, ee, 'Q6 closed-loop + nullspace', 'q6_closed_loop.gif')
    err = plot_results(log, 'q6_closed_loop.png', 'Q6: closed-loop control with nullspace motion')
    print(f'Q6 closed loop: max |pos err| = {err.max() * 1e3:.3f} mm, '
          f'RMS = {np.sqrt(np.mean(err**2)) * 1e3:.3f} mm')
    _print_setpoints(log)
    p.disconnect()
    return log


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, choices=[3, 5, 6])
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print('segment durations [s]:', np.round(segment_durations(), 3))
    for part, fn in [(3, part3), (5, part5), (6, part6)]:
        if args.part in (None, part):
            fn(args.gui)
