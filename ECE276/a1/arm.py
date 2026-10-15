"""Kinematics of the planar three-link arm (l1 = l2 = 0.5 m, l3 = 0.2 m)."""
import numpy as np

L = np.array([0.5, 0.5, 0.2])


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def fk(q, L=L):
    """End-effector pose [x, y, theta] from joint angles q = [q1, q2, q3].

    x     = l1 c1 + l2 c12 + l3 c123
    y     = l1 s1 + l2 s12 + l3 s123
    theta = q1 + q2 + q3
    """
    c = np.cumsum(q)
    return np.array([np.sum(L * np.cos(c)), np.sum(L * np.sin(c)), c[-1]])


def joint_points(q, L=L):
    """Planar positions of joint 1, joint 2, joint 3 and the end effector (4x2)."""
    c = np.cumsum(q)
    seg = np.stack([L * np.cos(c), L * np.sin(c)], axis=1)
    return np.vstack([[0.0, 0.0], np.cumsum(seg, axis=0)])


def jacobian(q, L=L):
    """3x3 Jacobian d[x, y, theta]/dq. Rows 0:2 are the position Jacobian."""
    c = np.cumsum(q)
    J = np.ones((3, 3))
    for i in range(3):
        J[0, i] = -np.sum(L[i:] * np.sin(c[i:]))
        J[1, i] = np.sum(L[i:] * np.cos(c[i:]))
    return J


def ik(pose, elbow=1.0, q_near=None, L=L):
    """Analytic inverse kinematics for pose [x, y, theta].

    The wrist point (x, y) - l3 [cos theta, sin theta] is solved as a 2-link arm.
    elbow = +1 / -1 picks the elbow branch. If q_near is given, each angle is
    shifted by multiples of 2 pi to the value closest to q_near (continuity).
    """
    x, y, th = pose
    xw, yw = x - L[2] * np.cos(th), y - L[2] * np.sin(th)
    c2 = (xw**2 + yw**2 - L[0]**2 - L[1]**2) / (2 * L[0] * L[1])
    if abs(c2) > 1.0 + 1e-10:
        raise ValueError(f'Unreachable end-effector pose: {pose}')
    q2 = elbow * np.arccos(np.clip(c2, -1.0, 1.0))
    q1 = np.arctan2(yw, xw) - np.arctan2(L[1] * np.sin(q2), L[0] + L[1] * np.cos(q2))
    q = np.array([q1, q2, th - q1 - q2])
    if q_near is not None:
        q = q_near + wrap(q - q_near)
    return q


def pinv_and_null(J):
    """SVD pseudoinverse and orthogonal nullspace projector (Lecture 4.5).

    Damping must not be used in I - J^dagger J: it would leak the secondary
    motion into the Cartesian task. Retain all nonzero singular values.
    """
    u, s, vt = np.linalg.svd(J, full_matrices=True)
    tol = np.finfo(float).eps * max(J.shape) * s[0]
    rank = int(np.count_nonzero(s > tol))
    J_dagger = (vt[:rank].T / s[:rank]) @ u[:, :rank].T
    basis = vt[rank:].T
    return J_dagger, basis @ basis.T
