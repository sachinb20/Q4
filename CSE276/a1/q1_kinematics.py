"""CSE276C HW1 - Q1: Forward and Inverse Kinematics of a planar 3-DoF arm.

Conventions
-----------
* World frame {W}: origin at the base, +x to the right, +y up (the arm's
  zero-pose direction), +z out of the page.
* The first joint q0 sits at (0, L0) with L0 = 5 cm. Links are L = 10 cm each.
* Joint angles are measured counter-clockwise (right-hand rule about +z),
  relative to the previous link; q = 0 means the link continues straight.
  A link whose absolute angle is theta (measured from +y) points along
  (-sin(theta), cos(theta)).
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

L0, L = 5.0, 10.0
LIMIT = np.pi / 2  # +/- 90 deg joint limits


# --------------------------------------------------------------------------
# (a) Forward kinematics, derived symbolically
# --------------------------------------------------------------------------
def derive_symbolic():
    q0, q1, q2 = sp.symbols("q0 q1 q2", real=True)
    l0, l = sp.symbols("L0 L", positive=True)

    def rot(t):
        return sp.Matrix([[sp.cos(t), -sp.sin(t), 0],
                          [sp.sin(t), sp.cos(t), 0],
                          [0, 0, 1]])

    def trans(x, y):
        return sp.Matrix([[1, 0, x], [0, 1, y], [0, 0, 1]])

    # Chain of homogeneous transforms: base offset -> joint q0 -> link ->
    # joint q1 -> link -> joint q2 -> link -> end effector.
    T = (trans(0, l0) * rot(q0) * trans(0, l)
         * rot(q1) * trans(0, l) * rot(q2) * trans(0, l))
    T = sp.simplify(T)
    pos = sp.Matrix([sp.simplify(T[0, 2]), sp.simplify(T[1, 2])])
    J = sp.simplify(pos.jacobian([q0, q1, q2]))

    print("=== Forward kinematics (symbolic) ===")
    print("x =", pos[0])
    print("y =", pos[1])
    print("\n=== Motion model  xdot = T qdot,  T = d(x,y)/d(q) ===")
    sp.pprint(J)
    return (q0, q1, q2, l0, l), pos, J


def fk_points(q):
    """Return the (x, y) of base, q0, q1, q2 joints and end effector."""
    pts = [np.array([0.0, 0.0]), np.array([0.0, L0])]
    theta = 0.0
    for qi in q:
        theta += qi
        pts.append(pts[-1] + L * np.array([-np.sin(theta), np.cos(theta)]))
    return np.array(pts)


def fk(q):
    return fk_points(q)[-1]


def jacobian(q):
    q0, q1, q2 = q
    a, b, c = q0, q0 + q1, q0 + q1 + q2
    return -L * np.array([
        [np.cos(a) + np.cos(b) + np.cos(c), np.cos(b) + np.cos(c), np.cos(c)],
        [np.sin(a) + np.sin(b) + np.sin(c), np.sin(b) + np.sin(c), np.sin(c)],
    ])


# --------------------------------------------------------------------------
# (b) Inverse kinematics
# --------------------------------------------------------------------------
def ik_closed_form():
    """A symmetric closed-form solution for target (10, 15).

    Relative to joint q0 the target is (10, 10), i.e. the three unit link
    directions must sum to (1, 1). Pick absolute link angles (from +y,
    clockwise) 45-a, 45, 45+a: the sum points along 45 deg with magnitude
    1 + 2cos(a) = sqrt(2)  ->  a = arccos((sqrt(2) - 1) / 2).
    In our CCW convention this gives q = [a - 45, -a, -a] (degrees).
    """
    a = np.arccos((np.sqrt(2) - 1) / 2)
    return np.array([a - np.pi / 4, -a, -a])


def ik_nullspace(target, q_init=(0.0, -0.5, -0.5), iters=2000, tol=1e-10,
                 k_null=0.5, alpha=0.2, damping=1e-2):
    """Resolved-rate IK with a null-space term.

    dq = T^+ (x_d - x) + (I - T^+ T) * dq_0,  dq_0 = -k_null * q
    The primary task drives the end effector to the target; the secondary
    task, projected into the null space of T (so it does not move the end
    effector), pulls the joints toward 0 to stay away from the +/-90 limits.
    T^+ is a damped pseudo-inverse and steps are scaled by alpha, since the
    straight-arm pose q = 0 is singular.
    """
    q = np.array(q_init, float)
    for _ in range(iters):
        err = target - fk(q)
        J = jacobian(q)
        Jp = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(2))
        N = np.eye(3) - Jp @ J
        dq = Jp @ err + N @ (-k_null * q)
        q = q + alpha * dq
        if np.linalg.norm(err) < tol and np.linalg.norm(N @ q) < 1e-8:
            break
    return q


def report_ik(name, q, target):
    p = fk(q)
    ok = np.all(np.abs(q) <= LIMIT + 1e-12)
    print(f"\n--- {name} ---")
    print("q (deg)        =", np.round(np.degrees(q), 4))
    print("FK(q)          =", np.round(p, 8))
    print("error |x - x_d| = %.2e" % np.linalg.norm(p - target))
    print("within +/-90 deg:", ok)
    J = jacobian(q)
    _, _, Vt = np.linalg.svd(J)
    print("null space of T at q:", np.round(Vt[-1], 4))


# --------------------------------------------------------------------------
# Figure: world/joint frames and IK solution
# --------------------------------------------------------------------------
def draw_frame(ax, origin, theta, label, scale=3.0, color="k"):
    """Draw a 2D frame whose y-axis points along the link (angle theta)."""
    xa = np.array([np.cos(theta), np.sin(theta)])
    ya = np.array([-np.sin(theta), np.cos(theta)])
    ax.arrow(*origin, *(scale * xa), head_width=0.6, color="tab:red",
             length_includes_head=True)
    ax.arrow(*origin, *(scale * ya), head_width=0.6, color="tab:green",
             length_includes_head=True)
    ax.text(origin[0] - 1.5, origin[1] - 2.0, label, fontsize=9, color=color)


def plot(q_zero, q_sol, target, fname="q1_frames_ik.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, q, title in [(axes[0], q_zero, "Frames, q = [0, 0, 0]"),
                         (axes[1], q_sol, "IK solution for (10, 15)")]:
        pts = fk_points(q)
        ax.plot(pts[:, 0], pts[:, 1], "-o", color="0.4", lw=3, ms=6)
        draw_frame(ax, pts[0], 0.0, "{W}")
        theta = 0.0
        for i, qi in enumerate(q):
            theta += qi
            draw_frame(ax, pts[i + 1], theta, f"{{{i}}} q{i}")
        draw_frame(ax, pts[-1], theta, "{E}")
        if title.startswith("IK"):
            ax.plot(*target, "r*", ms=15, label="target (10, 15)")
            ax.legend(loc="lower right")
            ax.set_title(title + "\nq = [%.2f, %.2f, %.2f] deg"
                         % tuple(np.degrees(q)))
        else:
            ax.set_title(title + "\n(red = x-axis, green = y-axis)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(-15, 22)
        ax.set_ylim(-3, 38)
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"\nSaved figure to {fname}")


if __name__ == "__main__":
    derive_symbolic()

    target = np.array([10.0, 15.0])
    print("\n=== Inverse kinematics, target (10, 15) ===")
    q_cf = ik_closed_form()
    report_ik("Closed-form symmetric solution", q_cf, target)
    q_ns = ik_nullspace(target)
    report_ik("Pseudo-inverse + null-space IK (from q = [0, -0.5, -0.5] rad)", q_ns, target)

    plot(np.zeros(3), q_cf, target)
