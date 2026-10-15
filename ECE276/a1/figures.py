"""Q1 (arm with body-fixed frames) and Q4 (control block diagrams) figures."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyBboxPatch

from arm import L, joint_points

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# ----------------------------------------------------------------------------- Q1
def frame(ax, origin, angle, name, length=0.16, z_mark=True):
    """Draw x (red) and y (green) axes of a planar frame; z points out of the page."""
    o = np.asarray(origin)
    for vec_angle, color, lbl in [(angle, 'tab:red', 'x'), (angle + np.pi / 2, 'tab:green', 'y')]:
        d = length * np.array([np.cos(vec_angle), np.sin(vec_angle)])
        ax.annotate('', o + d, o, arrowprops=dict(arrowstyle='-|>', color=color, lw=1.8))
        ax.text(*(o + 1.22 * d), f'${lbl}_{{{name}}}$', color=color, fontsize=12, ha='center', va='center')
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
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(pts[:, 0], pts[:, 1], '-', color='0.35', lw=7, solid_capstyle='round', alpha=0.35, zorder=1)
    ax.plot(pts[:-1, 0], pts[:-1, 1], 'ko', ms=7, zorder=2)
    ax.plot(*pts[-1], 's', color='tab:green', ms=9, zorder=2)

    # base and world frame {0}
    ax.fill([-0.08, 0.08, 0.05, -0.05], [-0.04, -0.04, 0.0, 0.0], color='0.6', zorder=0)
    frame(ax, [0, 0], 0., '0', length=0.22)
    frame(ax, pts[0], c[0], '1')
    frame(ax, pts[1], c[1], '2')
    frame(ax, pts[2], c[2], '3')
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
        ax.text(*(m - 0.06 * nrm), f'$l_{i + 1}$', fontsize=14, ha='center', va='center', color='0.2')

    ax.text(0.40, 0.03, r'All $z_i$ axes point out of the page ($\odot$), parallel to the joint axes;'
            '\n' r'gravity acts along $-z$. Arm moves in the $x_0y_0$ plane.'
            '\n' r'Frame $\{i\}$ is attached to link $i$ at joint $i$ ($x_i$ along link $i$);'
            '\n' r'$\{ee\}$ is at the tip, $a_3=l_3$ along $x_3$ (Table 1, modified D-H).',
            transform=ax.transAxes, va='bottom', fontsize=10,
            bbox=dict(fc='w', ec='0.7'))
    ax.set_aspect('equal')
    ax.set_xlim(-0.25, 1.15)
    ax.set_ylim(-0.15, 1.08)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(r'Q1: three-link planar arm with body-fixed frames ($\theta$ = 30°, 50°, −45°)')
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
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 9.5), gridspec_kw=dict(height_ratios=[1, 1.6]))
    for ax in (a1, a2):
        ax.set_aspect('equal')
        ax.axis('off')

    # --- open loop
    a1.set_xlim(-0.3, 15.2)
    a1.set_ylim(-0.9, 1.0)
    a1.set_title('(a) Open-loop control (no end-effector feedback)', fontsize=12, loc='left')
    b1 = box(a1, (1.9, 0), 'Trajectory generator\n2nd-order profile s(t)')
    b2 = box(a1, (5.0, 0), 'Inverse kinematics\n' r'$q_d=\mathrm{IK}(x_d,y_d,\theta_d)$', w=2.0)
    b3 = box(a1, (8.3, 0), 'Joint servos\n(PyBullet motors)')
    b4 = box(a1, (11.1, 0), 'Robot arm\n(3-link)', w=1.4)
    b5 = box(a1, (13.9, 0), 'Forward\nkinematics', w=1.4, fc='#f2f2f2')
    arrow(a1, (-0.3, 0), b1['l'], r'$\{x,y,\theta\}_{ref}$', off=(0.0, 0.18))
    a1.text(-0.3, -0.35, r'(setpoint list)', fontsize=9)
    arrow(a1, b1['r'], b2['l'], r'$\{x,y,\theta\}_d(t)$', off=(0, 0.18))
    arrow(a1, b2['r'], b3['l'], r'$q_d(t),\ \dot q_d(t)$', off=(0, 0.18))
    arrow(a1, b3['r'], b4['l'], r'$\tau$', off=(0, 0.12))
    arrow(a1, b4['r'], b5['l'], r'$q$', off=(0, 0.12))
    a1.text(13.9, -0.65, r'$\{x,y,\theta\}$ (output, not fed back)', ha='center', fontsize=10)

    # --- closed loop
    a2.set_xlim(-0.3, 15.2)
    a2.set_ylim(-3.3, 1.9)
    a2.set_title('(b) Closed-loop resolved-rate control with nullspace term', fontsize=12, loc='left')
    y0 = 0.6
    g = box(a2, (1.9, y0), 'Trajectory generator\n2nd-order profile s(t)')
    arrow(a2, (-0.3, y0), g['l'], r'$\{x,y\}_{ref}$', off=(0, 0.18))
    s1 = summer(a2, (3.9, y0))
    arrow(a2, g['r'], s1['l'], r'$p_d$', off=(0, 0.14), sign='+')
    k = box(a2, (5.3, y0), r'$K_p$', w=0.9)
    arrow(a2, s1['r'], k['l'], r'$e$', off=(0, 0.12))
    s2 = summer(a2, (6.6, y0))
    arrow(a2, k['r'], s2['l'], sign='+')
    # feed-forward velocity
    poly(a2, [(1.9, y0 + 0.4), (1.9, 1.45), (6.6, 1.45), s2['t']])
    a2.text(4.2, 1.55, r'$\dot p_d$ (feed-forward)', ha='center', fontsize=10)
    a2.text(6.38, y0 + 0.28, '+', fontsize=11)
    jp = box(a2, (8.0, y0), r'$J^{+}(q)$' '\n' r'(damped pinv)', w=1.5)
    arrow(a2, s2['r'], jp['l'], r'$\dot p_{cmd}$', off=(0.05, -0.33))
    s3 = summer(a2, (9.5, y0))
    arrow(a2, jp['r'], s3['l'], sign='+')
    rob = box(a2, (11.4, y0), 'Joint velocity\nservos + robot', w=1.8)
    arrow(a2, s3['r'], rob['l'], r'$\dot q$', off=(0, 0.14))
    fk_b = box(a2, (13.9, y0), 'Forward\nkinematics', w=1.4)
    arrow(a2, rob['r'], fk_b['l'], r'$q$', off=(0, 0.12))
    poly(a2, [fk_b['r'], (15.0, y0)], end_arrow=True)
    a2.text(15.0, y0 + 0.15, r'$p$', fontsize=11)
    # position feedback
    poly(a2, [(14.9, y0), (14.9, -2.85), (3.9, -2.85), s1['b']])
    a2.text(9.0, -3.1, r'measured end-effector position $p=[x,y]$', ha='center', fontsize=10)
    a2.text(3.65, y0 - 0.45, '−', fontsize=13)
    # nullspace branch
    h = box(a2, (8.0, -1.0), r'$\nabla H(q)$, $H=-\frac{1}{2}(\theta_2^2+\theta_3^2)$' '\n'
            '(maximise inter-link angles)', w=3.4, fc='#fdf1e3')
    n = box(a2, (8.0, -2.05), r'$(I-J^{+}J)\,k_0$', w=1.9, fc='#fdf1e3')
    poly(a2, [(12.4, y0 - 0.4), (12.4, -1.0), (h['r'][0], -1.0)])
    a2.text(12.55, -0.6, r'$q$', fontsize=11)
    arrow(a2, h['b'], n['t'])
    poly(a2, [n['r'], (9.5, -2.05), s3['b']])
    a2.text(9.6, y0 - 0.45, '+', fontsize=11)
    a2.text(9.65, -1.6, r'$\dot q_0$', fontsize=10)
    a2.text(0.0, -1.6, r'$\dot q = J^{+}(\dot p_d + K_p(p_d-p)) + (I-J^{+}J)\,k_0\nabla H(q)$',
            fontsize=12)
    a2.text(0.0, -2.1, r'$J=\partial[x,y]/\partial q\ \in\mathbb{R}^{2\times3}$', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'q4_block_diagrams.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    q1_figure()
    q4_figure()
