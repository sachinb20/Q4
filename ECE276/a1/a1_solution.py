"""ECE276C Assignment 1 -- Robot Kinematics and Simple Control.

Usage (conda env q4):
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

from arm import L, dls_pinv, fk, ik, jacobian, joint_points, wrap

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
    cam = TopCamera(center=[0.25, 0.25], half_height=1.05)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
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
        pts = joint_points(q)
        ax.plot(pts[:, 0], pts[:, 1], 'w--', lw=0.8, alpha=0.8)
        ax.annotate(f'ee ({pose_fk[0]:.3f}, {pose_fk[1]:.3f})\nθ = {np.degrees(wrap(pose_fk[2])):.0f}°',
                    pts[-1], xytext=(10, -28), textcoords='offset points', color='k', fontsize=9,
                    bbox=dict(fc='w', ec='none', alpha=0.85))
        ax.set_title(f'Config {k}: θ = ({_pi_str(q[0])}, {_pi_str(q[1])}, {_pi_str(q[2])})')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
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
    """Equal peak acceleration on every segment: T_i ~ sqrt(distance_i), sum = 10 s."""
    d = np.linalg.norm(np.diff(LOCATIONS[:, :2], axis=0), axis=1)
    return TOTAL_TIME * np.sqrt(d) / np.sum(np.sqrt(d))


def profile(tau):
    """2nd-order (constant accel / constant decel) profile on tau = t/T in [0, 1].
    Returns s, ds/dtau, d2s/dtau2; s(0)=0, s(1)=1, s'(0)=s'(1)=0."""
    if tau < 0.5:
        return 2 * tau**2, 4 * tau, 4.
    return 1 - 2 * (1 - tau)**2, 4 * (1 - tau), -4.


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
    log = {k: [] for k in ['t', 'x_des', 'x_meas', 'q']}
    frames, des_xy = [], desired_path()
    for k in range(n_steps + 1):
        t = k * DT
        x_des, _ = reference(t)
        x_meas = ee_pose(robot, ee)
        q, _ = get_q(robot, joints)
        for key, val in zip(log, [t, x_des, x_meas, q]):
            log[key].append(val)
        if k % every == 0:
            act = np.array(log['x_meas'])[:, :2]
            frames.append(draw_frame(cam.render(), cam, des_xy, act, t, title,
                                     f'ee theta = {np.degrees(wrap(x_meas[2])):6.1f} deg'))
        if k < n_steps:
            step_fn(t, q, x_meas)
            p.stepSimulation()
    imageio.mimsave(os.path.join(OUT, gif_name), frames, duration=1000. / GIF_FPS, loop=0)
    return {k: np.array(v) for k, v in log.items()}


def plot_results(log, fname, title, show_theta_des=True):
    err = np.linalg.norm(log['x_meas'][:, :2] - log['x_des'][:, :2], axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].plot(log['x_des'][:, 0], log['x_des'][:, 1], 'k--', lw=1, label='desired')
    ax[0].plot(log['x_meas'][:, 0], log['x_meas'][:, 1], 'g', lw=2, alpha=0.8, label='end effector')
    for k, (x, y, _) in enumerate(LOCATIONS[:4], start=1):
        ax[0].plot(x, y, 'ko')
        ax[0].annotate(str(k), (x, y), xytext=(6, 6), textcoords='offset points')
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('x [m]')
    ax[0].set_ylabel('y [m]')
    ax[0].set_title('End-effector path')
    ax[0].legend()

    t = log['t']
    ax[1].plot(t, log['x_meas'][:, 0], label='x')
    ax[1].plot(t, log['x_meas'][:, 1], label='y')
    ax[1].plot(t, wrap(log['x_meas'][:, 2]), label='θ_ee')
    if show_theta_des:
        ax[1].plot(t, log['x_des'][:, 2], 'k:', label='θ_des')
    ax[1].plot(t, log['x_des'][:, 0], 'k--', lw=0.8)
    ax[1].plot(t, log['x_des'][:, 1], 'k--', lw=0.8)
    ax[1].set_xlabel('t [s]')
    ax[1].set_title('Pose vs time (dashed: desired)')
    ax[1].legend()

    ax[2].semilogy(t, np.maximum(err, 1e-9))
    ax[2].set_xlabel('t [s]')
    ax[2].set_ylabel('|p_des - p| [m]')
    ax[2].set_title(f'Position error (max {err.max() * 1e3:.2f} mm)')
    fig.suptitle(title)
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
        print(f'  t = {ts:5.2f} s  location {[1, 2, 3, 4, 1][k]}: measured ee = {np.round(log["x_meas"][i], 4)}')


# ----------------------------------------------------------------------------- Q6
K_P = 20.        # task-space feedback gain [1/s]
K_NULL = 10.     # nullspace gradient gain
LAMBDA = 1e-3    # DLS damping (only matters next to the stretched-out singularity)


def null_gradient(q):
    """Gradient ascent direction for H(q) = -1/2 (q2^2 + q3^2).

    The angle between link i and link i+1 is pi - |q_{i+1}|, so maximising the
    angles between sequential links means driving q2, q3 towards 0 (a smooth
    surrogate of maximising sum_i (pi - |q_{i+1}|))."""
    qw = wrap(q)
    return -np.array([0., qw[1], qw[2]])


def part6(gui=False):
    """Closed-loop resolved-rate control on (x, y) with a nullspace term:
        qdot = J+ (pdot_des + K (p_des - p)) + (I - J+ J) K_null grad H(q)
    J is the 2x3 position Jacobian, p is the *measured* end-effector position."""
    robot, joints, ee = make_sim(gui)
    # Location 1 is at full reach (only the straight arm reaches it), which is a
    # kinematic singularity: no joint velocity moves the tip radially. Start with
    # a tiny symmetric bend (rotated so the tip stays on the x axis) to pick the
    # elbow direction; the initial tip error is below 0.1 mm.
    eps = 0.01
    q0 = np.array([0., eps, eps])
    q0[0] = -np.arctan2(fk(q0)[1], fk(q0)[0])
    set_q(robot, joints, q0)
    print(f'\nQ6 initial q = {np.round(q0, 4)}, initial tip error = '
          f'{np.linalg.norm(fk(q0)[:2] - LOCATIONS[0, :2]) * 1e3:.3f} mm')

    def step(t, q_meas, x_meas):
        p_des, v_des = reference(t)
        J = jacobian(q_meas)[:2]
        J_pinv = dls_pinv(J, LAMBDA)
        task = J_pinv @ (v_des[:2] + K_P * (p_des[:2] - x_meas[:2]))
        null = (np.eye(3) - J_pinv @ J) @ (K_NULL * null_gradient(q_meas))
        qdot = task + null
        p.setJointMotorControlArray(robot, joints, p.VELOCITY_CONTROL,
                                    targetVelocities=qdot.tolist(), forces=[500.] * 3)

    log = run_and_record(step, robot, joints, ee, 'Q6 closed-loop + nullspace', 'q6_closed_loop.gif')
    err = plot_results(log, 'q6_closed_loop.png', 'Q6: closed-loop control with nullspace link-angle maximisation',
                       show_theta_des=False)
    print(f'Q6 closed loop: max |pos err| = {err.max() * 1e3:.3f} mm, '
          f'RMS = {np.sqrt(np.mean(err**2)) * 1e3:.3f} mm')
    _print_setpoints(log)
    _plot_link_angles(log)
    p.disconnect()
    return log


def optimal_link_angles(p_xy, n=2001):
    """Brute-force reference: over all ee orientations reaching p_xy, the one
    minimising q2^2 + q3^2. Returns the two inter-link angles [rad]."""
    ths = np.linspace(-np.pi, np.pi, n)
    best, best_q = np.inf, None
    for elbow in (1., -1.):
        q = np.array([ik([p_xy[0], p_xy[1], th], elbow) for th in ths])
        cost = wrap(q[:, 1])**2 + wrap(q[:, 2])**2
        reach = np.hypot(p_xy[0] - L[2] * np.cos(ths), p_xy[1] - L[2] * np.sin(ths)) <= L[0] + L[1]
        cost[~reach] = np.inf
        i = np.argmin(cost)
        if cost[i] < best:
            best, best_q = cost[i], q[i]
    return np.pi - np.abs(wrap(best_q[1:]))


def _plot_link_angles(log):
    """Inter-link angles of Q6 vs the theta = 0 IK solution of Q5 and the static optimum."""
    q5 = [ik(x, elbow=1.) for x in log['x_des']]
    q5 = np.unwrap(np.array(q5), axis=0)
    sub = slice(None, None, 24)
    opt = np.array([optimal_link_angles(x[:2]) for x in log['x_des'][sub]])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, c in zip([1, 2], ['C0', 'C1']):
        ax.plot(log['t'][sub], np.degrees(opt[:, i - 1]), c + 'o', ms=3, mfc='none',
                label=f'optimum: angle between link {i} and {i + 1}')
        ax.plot(log['t'], np.degrees(np.pi - np.abs(wrap(log['q'][:, i]))), c,
                label=f'Q6: angle between link {i} and {i + 1}')
        ax.plot(log['t'], np.degrees(np.pi - np.abs(wrap(q5[:, i]))), c + '--',
                label=f'Q5 (θ = 0): angle between link {i} and {i + 1}')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('angle [deg]  (180 = straight)')
    ax.set_title('Angles between sequential links')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'q6_link_angles.png'), dpi=120)
    plt.close(fig)


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
