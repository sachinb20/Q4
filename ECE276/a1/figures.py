"""Report figures and representative frames extracted from the Q5/Q6 GIFs."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyBboxPatch
from PIL import Image

from arm import L, joint_points, wrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# ----------------------------------------------------------------------------- Q1
def frame(ax, origin, angle, name, length=0.16, z_mark=True, x_label_shift=(0., 0.)):
    """Draw x (red) and y (green) axes of a planar frame; z points out of the page."""
    o = np.asarray(origin)
    for vec_angle, color, lbl, shift in [(angle, 'tab:red', 'x', x_label_shift),
                                         (angle + np.pi / 2, 'tab:green', 'y', (0., 0.))]:
        d = length * np.array([np.cos(vec_angle), np.sin(vec_angle)])
        ax.annotate('', o + d, o, arrowprops=dict(arrowstyle='-|>', color=color, lw=1.8))
        ax.text(*(o + 1.22 * d + np.asarray(shift)), f'${lbl}_{{{name}}}$', color=color, fontsize=12,
                ha='center', va='center')
    if z_mark:
        ax.plot(*o, 'o', ms=9, mfc='w', mec='tab:blue', mew=1.5, zorder=5)
        ax.plot(*o, '.', ms=4, color='tab:blue', zorder=6)


def angle_arc(ax, center, a0, a1, r, label):
    lo, hi = sorted([a0, a1])
    ax.add_patch(Arc(center, 2 * r, 2 * r, theta1=np.degrees(lo), theta2=np.degrees(hi), lw=1.2))
    mid = (a0 + a1) / 2
    ax.text(center[0] + 1.35 * r * np.cos(mid), center[1] + 1.35 * r * np.sin(mid), label,
            fontsize=13, ha='center', va='center')


def q1_figure():
    q = np.radians([30., 50., -45.])
    pts = joint_points(q)
    c = np.cumsum(q)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pts[:, 0], pts[:, 1], '-', color='0.35', lw=7, solid_capstyle='round', alpha=0.35, zorder=1)
    ax.plot(pts[:-1, 0], pts[:-1, 1], 'ko', ms=7, zorder=2)
    ax.plot(*pts[-1], 's', color='tab:green', ms=9, zorder=2)

    # base and world frame {0}
    ax.fill([-0.08, 0.08, 0.05, -0.05], [-0.04, -0.04, 0.0, 0.0], color='0.6', zorder=0)
    frame(ax, [0, 0], 0., '0', length=0.22)
    frame(ax, pts[0], c[0], '1')
    frame(ax, pts[1], c[1], '2')
    # x3 label below link 3, clear of the theta3 arc
    frame(ax, pts[2], c[2], '3', length=0.11, x_label_shift=0.045 * np.array([np.sin(c[2]), -np.cos(c[2])]))
    frame(ax, pts[3], c[2], 'ee', length=0.12)

    # joint angles: measured from the previous x axis (dashed extension) to the new one
    angle_arc(ax, pts[0], 0., c[0], 0.3, r'$\theta_1$')
    for i, (a_prev, a_new) in enumerate([(c[0], c[1]), (c[1], c[2])], start=2):
        o = pts[i - 1]
        ext = o + 0.22 * np.array([np.cos(a_prev), np.sin(a_prev)])
        ax.plot([o[0], ext[0]], [o[1], ext[1]], 'k:', lw=1)
        angle_arc(ax, o, a_prev, a_new, 0.12, fr'$\theta_{i}$')

    for i in range(3):
        m = (pts[i] + pts[i + 1]) / 2
        nrm = np.array([-np.sin(c[i]), np.cos(c[i])])
        ax.text(*(m - (0.09 if i == 2 else 0.06) * nrm), f'$l_{i + 1}$', fontsize=14, ha='center', va='center', color='0.2')

    ax.text(0.98, 0.02, r'$\odot$: all $z_i$ out of the page; gravity along $-z_0$.',
            transform=ax.transAxes, ha='right', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(-0.23, 1.02)
    ax.set_ylim(-0.13, 1.02)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Three-link planar arm and body-fixed frames')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'q1_frames.png'), dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- Q4
def box(ax, xy, text, w=1.7, h=0.8, fc='#e8f0fb'):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.03', fc=fc, ec='k'))
    ax.text(x, y, text, ha='center', va='center', fontsize=10)
    return dict(l=(x - w / 2, y), r=(x + w / 2, y), t=(x, y + h / 2), b=(x, y - h / 2))


def summer(ax, xy, r=0.17):
    ax.add_patch(plt.Circle(xy, r, fc='w', ec='k'))
    ax.text(*xy, r'$\Sigma$', ha='center', va='center', fontsize=10)
    x, y = xy
    return dict(l=(x - r, y), r=(x + r, y), t=(x, y + r), b=(x, y - r))


def arrow(ax, a, b, label=None, off=(0, 0.14), sign=None, sign_off=(-0.12, 0.1)):
    ax.annotate('', b, a, arrowprops=dict(arrowstyle='-|>', lw=1.3))
    if label:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], label, ha='center', fontsize=10)
    if sign:
        ax.text(b[0] + sign_off[0], b[1] + sign_off[1], sign, fontsize=11, ha='center')


def poly(ax, pts, end_arrow=True):
    pts = np.array(pts)
    ax.plot(pts[:-1, 0], pts[:-1, 1], 'k', lw=1.3)
    if end_arrow:
        ax.annotate('', pts[-1], pts[-2], arrowprops=dict(arrowstyle='-|>', lw=1.3))


def q4_figure():
    """Full-pose setpoint control, following Lecture 4 slides 10 and 14."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 4.7))
    for ax in (a1, a2):
        ax.set_xlim(-0.2, 13.5)
        ax.set_aspect('equal')
        ax.axis('off')

    a1.set_ylim(-0.8, 1.)
    a1.set_title('(a) Open-loop setpoint control', fontsize=12, loc='left')
    inv = box(a1, (3., 0.), 'Inverse\nkinematics', w=2.)
    act = box(a1, (6.8, 0.), 'Actuator\ncontrol loop', w=2.)
    rob = box(a1, (10.5, 0.), 'Robot\n' r'$\mathbf{x}=f(\mathbf{q})$', w=2.)
    arrow(a1, (0., 0.), inv['l'], r'$\mathbf{x}_{ref}$')
    arrow(a1, inv['r'], act['l'], r'$\mathbf{q}_{ref}$')
    arrow(a1, act['r'], rob['l'], r'$\mathbf{q}$')
    arrow(a1, rob['r'], (13., 0.), r'$\mathbf{x}$')

    a2.set_ylim(-1.65, 1.)
    a2.set_title('(b) Closed-loop setpoint control', fontsize=12, loc='left')
    summ = summer(a2, (1.65, 0.))
    gain = box(a2, (3.0, 0.), r'$K$', w=1.0)
    pinv = box(a2, (4.9, 0.), r'$J^\dagger(\mathbf{q})$', w=1.4)
    summ_q = summer(a2, (6.6, 0.))
    act = box(a2, (8.45, 0.), 'Actuator\ncontrol loop', w=1.7)
    rob = box(a2, (11.2, 0.), 'Robot\n' r'$\mathbf{x}=f(\mathbf{q})$', w=1.7)
    arrow(a2, (0., 0.), summ['l'], r'$\mathbf{x}_{ref}$', sign='+')
    arrow(a2, summ['r'], gain['l'], r'$\mathbf{e}$')
    arrow(a2, gain['r'], pinv['l'], r'$\delta\mathbf{x}$')
    arrow(a2, pinv['r'], summ_q['l'], r'$\delta\mathbf{q}$', sign='+')
    arrow(a2, summ_q['r'], act['l'], r'$\mathbf{q}_{ref}$')
    arrow(a2, act['r'], rob['l'], r'$\mathbf{q}$')
    arrow(a2, rob['r'], (13., 0.), r'$\mathbf{x}$')
    # measured joint angles: q_ref = q_meas + delta q
    poly(a2, [(9.8, 0.), (9.8, -0.6), (6.6, -0.6), summ_q['b']])
    a2.text(6.75, -0.45, '+', fontsize=11)
    a2.text(8.2, -0.47, r'$\mathbf{q}_{meas}$', fontsize=10, ha='center')
    sensor = box(a2, (7.7, -1.2), 'Sensor', w=1.5, h=0.5, fc='#f2f2f2')
    poly(a2, [(12.5, 0.), (12.5, -1.2), sensor['r']])
    poly(a2, [sensor['l'], (1.65, -1.2), summ['b']])
    a2.text(1.38, -0.4, '−', fontsize=13)
    a2.text(3.5, -1.05, r'$\mathbf{x}_{meas}$', fontsize=10)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(os.path.join(OUT, 'q4_block_diagrams.png'), dpi=160)
    plt.close(fig)


def animation_snapshots():
    """Extract the actual GIF frames at the three intermediate waypoints."""
    times_ms = [2500, 5000, 7500]
    for stem in ['q5_open_loop', 'q6_closed_loop']:
        with Image.open(os.path.join(OUT, stem + '.gif')) as gif:
            elapsed_ms = 0
            target = 0
            for index in range(gif.n_frames):
                gif.seek(index)
                duration_ms = gif.info['duration']
                if elapsed_ms <= times_ms[target] < elapsed_ms + duration_ms:
                    gif.convert('RGB').save(os.path.join(OUT, f'{stem}_frame{target + 1}.png'))
                    target += 1
                    if target == len(times_ms):
                        break
                elapsed_ms += duration_ms


# ----------------------------------------------------------------------------- Q6
def optimal_link_angles(p_xy, n=3600):
    """Brute-force reference for Q6: over every ee orientation that reaches p_xy
    (both elbow branches), the posture minimising wrap(q2)^2 + wrap(q3)^2.
    Returns the two interior angles [rad]."""
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    xw, yw = p_xy[0] - L[2] * np.cos(th), p_xy[1] - L[2] * np.sin(th)
    c2 = (xw**2 + yw**2 - L[0]**2 - L[1]**2) / (2 * L[0] * L[1])
    best, best_q = np.inf, None
    for elbow in (1., -1.):
        q2 = elbow * np.arccos(np.clip(c2, -1., 1.))
        q1 = np.arctan2(yw, xw) - np.arctan2(L[1] * np.sin(q2), L[0] + L[1] * np.cos(q2))
        q3 = th - q1 - q2
        cost = wrap(q2)**2 + wrap(q3)**2
        cost[np.abs(c2) > 1. + 1e-9] = np.inf
        i = np.argmin(cost)
        if cost[i] < best:
            best, best_q = cost[i], (q2[i], q3[i])
    return np.pi - np.abs(wrap(np.array(best_q)))


def q6_link_angles():
    """Interior angles of Q6 vs the brute-force optimum and the theta = 0 posture of Q5."""
    q6 = np.load(os.path.join(OUT, 'q6_closed_loop.npz'))
    q5 = np.load(os.path.join(OUT, 'q5_open_loop.npz'))
    t = q6['t']
    a6 = np.pi - np.abs(wrap(q6['q'][:, 1:]))
    a5 = np.pi - np.abs(wrap(q5['q'][:, 1:]))
    sub = np.arange(0, len(t), 24)
    opt = np.array([optimal_link_angles(x[:2]) for x in q6['x_des'][sub]])

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, color, name in [(0, 'tab:blue', r'$\alpha_{12}$'), (1, 'tab:orange', r'$\alpha_{23}$')]:
        ax.plot(t, np.degrees(a6[:, j]), color=color, lw=2, label=f'{name}  Q6 nullspace')
        ax.plot(t[sub], np.degrees(opt[:, j]), 'o', color=color, ms=3.5, mfc='none',
                label=f'{name}  constrained optimum')
        ax.plot(t, np.degrees(a5[:, j]), '--', color=color, lw=1, label=f'{name}  Q5 ($\\theta=0$)')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('interior angle [deg]')
    ax.set_ylim(0, 185)
    ax.set_title('Angles between sequential links (180° = straight)')
    ax.legend(fontsize=7.5, ncol=2, loc='lower left')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'q6_link_angles.png'), dpi=150)
    plt.close(fig)

    gap = np.degrees(np.abs(a6[sub] - opt)).max(axis=0)
    print(f'Q6 min interior angles: {np.degrees(a6.min(axis=0)).round(1)} deg; '
          f'Q5 (theta = 0): {np.degrees(a5.min(axis=0)).round(1)} deg; '
          f'max deviation from optimum: {gap.round(2)} deg')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    q1_figure()
    q4_figure()
    animation_snapshots()
    q6_link_angles()
