"""CSE276C HW1: reproducible kinematics and linear camera calibration.

Run: python solve_hw1.py
Dependencies: numpy, matplotlib. Input: imgpoints.txt beside this script.
Outputs: results/*.pdf, results/calibration.npz, results/summary.json,
and numerical LaTeX fragments used by report.tex.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def fk_points(q):
    directions = np.c_[-np.sin(np.cumsum(q)), np.cos(np.cumsum(q))]
    return np.vstack(([0., 0.], [0., 5.],
                      [0., 5.] + 10. * np.cumsum(directions, axis=0)))


def jacobian(q):
    a = np.cumsum(q)
    return -10. * np.array([np.cumsum(np.cos(a)[::-1])[::-1],
                           np.cumsum(np.sin(a)[::-1])[::-1]])


def inverse_kinematics():
    a = np.arccos((np.sqrt(2.) - 1.) / 2.)
    return np.array([a - np.pi / 4., -a, -a])


def world_points():
    # File order: eight points across each row, then the next row down.
    # Origin: bottom-left detected corner; X right, Y up, Z = 0 (mm).
    return np.array([(25. * c, 25. * (5 - r))
                     for r in range(6) for c in range(8)])


def homogeneous(p):
    return np.c_[p, np.ones(len(p))]


def project(H, p):
    z = homogeneous(p) @ H.T
    return z[:, :2] / z[:, 2, None]


def normalize(p):
    center = p.mean(axis=0)
    scale = np.sqrt(2.) / np.linalg.norm(p - center, axis=1).mean()
    N = np.array([[scale, 0., -scale * center[0]],
                  [0., scale, -scale * center[1]], [0., 0., 1.]])
    return project(N, p), N


def homography(p, uv):
    pn, Np = normalize(p)
    un, Nu = normalize(uv)
    ph = homogeneous(pn)
    M = np.zeros((2 * len(p), 9))
    M[0::2, :3] = ph
    M[1::2, 3:6] = ph
    M[0::2, 6:] = -un[:, 0, None] * ph
    M[1::2, 6:] = -un[:, 1, None] * ph
    _, singular, Vt = np.linalg.svd(M, full_matrices=(len(M) < 9))
    H = np.linalg.solve(Nu, Vt[-1].reshape(3, 3)) @ Np
    return H / H[2, 2], singular


def vrow(H, i, j):
    # Assignment order: b11, b12, b13, b22, b23, b33.
    x, y = H[:, i], H[:, j]
    return np.array([x[0] * y[0], x[0] * y[1] + x[1] * y[0],
                     x[0] * y[2] + x[2] * y[0], x[1] * y[1],
                     x[1] * y[2] + x[2] * y[1], x[2] * y[2]])


def intrinsic(Hs, zero_skew=False):
    L = np.vstack([row for H in Hs for row in
                   (vrow(H, 0, 1), vrow(H, 0, 0) - vrow(H, 1, 1))])
    cols = [0, 2, 3, 4, 5] if zero_skew else list(range(6))
    _, singular, Vt = np.linalg.svd(L[:, cols], full_matrices=False)
    b = np.zeros(6)
    b[cols] = Vt[-1]
    b /= b[-1]  # Report scale: B33 = 1.
    b11, b12, b13, b22, b23, b33 = b
    B = np.array([[b11, b12, b13], [b12, b22, b23],
                  [b13, b23, b33]])
    assert np.linalg.eigvalsh(B).min() > 0., "B must be positive definite"
    d = b11 * b22 - b12**2
    cy = (b12 * b13 - b11 * b23) / d
    lam = b33 - (b13**2 + cy * (b12 * b13 - b11 * b23)) / b11
    fx = np.sqrt(lam / b11)
    fy = np.sqrt(lam * b11 / d)
    skew = -b12 * fx**2 * fy / lam
    cx = skew * cy / fy - b13 * fx**2 / lam
    A = np.array([[fx, skew, cx], [0., fy, cy], [0., 0., 1.]])
    # Independent recovery from B = U.T @ U, where U is upper triangular.
    Ac = np.linalg.inv(np.linalg.cholesky(B).T)
    np.testing.assert_allclose(A, Ac / Ac[2, 2], atol=1e-9, rtol=0)
    return A, b, L, singular


def extrinsic(H, A):
    Q = np.linalg.solve(A, H)
    # Symmetric estimate of the common scale; choose positive board depth.
    scale = 2. / (np.linalg.norm(Q[:, 0]) + np.linalg.norm(Q[:, 1]))
    scale *= np.sign(Q[2, 2])
    r1, r2, t = (scale * Q).T
    R0 = np.column_stack((r1, r2, np.cross(r1, r2)))
    U, _, Vt = np.linalg.svd(R0)
    R = U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt
    return R, t, -R.T @ t


def error_stats(errors):
    return dict(total=float(errors.sum()), mean=float(errors.mean()),
                rmse=float(np.sqrt(np.mean(errors**2))),
                sse=float(np.sum(errors**2)), maximum=float(errors.max()))


def validate_math():
    """Check derivatives and recover known noiseless synthetic cameras."""
    rng = np.random.default_rng(276)
    eps = 1e-6
    for q in rng.uniform(-np.pi / 2., np.pi / 2., (30, 3)):
        Jfd = np.column_stack([
            (fk_points(q + eps * e)[-1] - fk_points(q - eps * e)[-1])
            / (2. * eps) for e in np.eye(3)])
        np.testing.assert_allclose(jacobian(q), Jfd, atol=1e-7, rtol=0)
    q = inverse_kinematics()
    assert np.all(np.abs(q) <= np.pi / 2.)
    np.testing.assert_allclose(fk_points(q)[-1], [10., 15.], atol=1e-12,
                               rtol=0)
    p = world_points()
    for skew in (0., 2.):
        A = np.array([[700., skew, 320.], [0., 720., 240.], [0., 0., 1.]])
        Hs, poses = [], []
        for ax, ay in [(.2, .3), (-.4, .1), (.1, -.5), (.5, .4)]:
            Rx = np.array([[1., 0., 0.], [0., np.cos(ax), -np.sin(ax)],
                           [0., np.sin(ax), np.cos(ax)]])
            Ry = np.array([[np.cos(ay), 0., np.sin(ay)], [0., 1., 0.],
                           [-np.sin(ay), 0., np.cos(ay)]])
            R, t = Rx @ Ry, np.array([20., -40., 600.])
            H = A @ np.column_stack((R[:, :2], t))
            Hfit, _ = homography(p, project(H, p))
            np.testing.assert_allclose(Hfit, H / H[2, 2], atol=1e-10,
                                       rtol=0)
            corners = p[[0, 7, 40, 47]]
            Hfour, _ = homography(corners, project(H, corners))
            np.testing.assert_allclose(Hfour, Hfit, atol=1e-10, rtol=0)
            Hs.append(Hfit)
            poses.append((R, t))
        Afit, _, _, _ = intrinsic(Hs, zero_skew=(skew == 0.))
        np.testing.assert_allclose(Afit, A, atol=1e-8, rtol=0)
        for H, (R, t) in zip(Hs, poses):
            for sign in (-1., 1.):
                Rfit, tfit, _ = extrinsic(sign * H, Afit)
                np.testing.assert_allclose(Rfit, R, atol=1e-10, rtol=0)
                np.testing.assert_allclose(tfit, t, atol=1e-8, rtol=0)


def save_figures(p, uv, Hs, errors, Rs, centers):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, q, title in zip(axes, [np.radians([-25., -35., -50.]),
                                  inverse_kinematics()],
                            ["Reference frames (schematic pose)",
                             "Valid inverse-kinematics solution"]):
        pts = fk_points(q)
        ax.plot(*pts.T, "o-", color="#334155", lw=2.5)
        angles = np.r_[0., np.cumsum(q)]
        for k, (origin, angle) in enumerate(zip(pts[:4], angles)):
            for d, color in [(np.array([np.cos(angle), np.sin(angle)]),
                              "#c43b3b"),
                             (np.array([-np.sin(angle), np.cos(angle)]),
                              "#17834b")]:
                ax.annotate("", origin + 3.5 * d, origin,
                            arrowprops=dict(arrowstyle="->", color=color))
            ax.annotate("W" if k == 0 else f"J{k-1}", origin,
                        xytext=(-20, -10), textcoords="offset points")
        for j in range(4) if ax is axes[0] else []:
            midpoint = (pts[j] + pts[j + 1]) / 2.
            ax.annotate("5 cm" if j == 0 else "10 cm", midpoint,
                        xytext=[(6, 0), (8, 0), (0, -18), (6, 8)][j],
                        textcoords="offset points", fontsize=8)
        if ax is axes[1]:
            ax.plot(10., 15., "*", ms=13, color="#dc2626")
        ax.set(xlim=(-12, 29), ylim=(-4, 30), xlabel="x (cm)",
               ylabel="y (cm)", title=title, aspect="equal")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "kinematics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(10, 7.3), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        pred = project(Hs[i], p)
        ax.scatter(*uv[i].T, s=13, facecolors="none", edgecolors="#2563eb",
                   label="measured")
        ax.scatter(*pred.T, s=10, marker="+", color="#dc2626", label="projected")
        ax.set(title=f"View {i+1}: RMSE {error_stats(errors[i])['rmse']:.3f} px",
               xlim=(0, 640), ylim=(480, 0), aspect="equal")
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.supxlabel("u (pixels)")
    fig.supylabel("v (pixels, downward)")
    fig.tight_layout()
    fig.savefig(OUT / "reprojections.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(121, projection="3d")
    border = np.array([[0, 0, 0], [175, 0, 0], [175, 125, 0],
                       [0, 125, 0], [0, 0, 0]])
    ax.plot(*border.T, color="0.5", label="corner-grid boundary")
    ax.plot(*centers.T, "o-", color="#2563eb", label="camera centers")
    for i, (C, R) in enumerate(zip(centers, Rs), 1):
        ax.text(*C, str(i), fontsize=8)
        direction = R.T[:, 2]  # Optical +z axis expressed in board frame.
        ax.quiver(*C, *direction, length=55., color="#dc2626",
                  arrow_length_ratio=.2)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
           title="Camera poses relative to the board")
    all_pts = np.vstack((border, centers))
    ax.set_box_aspect(np.ptp(all_pts, axis=0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(5))
    ax.view_init(elev=22., azim=-62.)
    ax.legend(fontsize=7, loc="upper left")
    ax = fig.add_subplot(122)
    for k, name in enumerate("XYZ"):
        ax.plot(range(1, 10), centers[:, k], "o-", label=name)
    ax.set(xlabel="Image index (file order)", ylabel="Position (mm)",
           title="Components of camera center", xticks=range(1, 10))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory.pdf")
    plt.close(fig)


def tex_matrix(A, digits=6, scientific=False):
    fmt = f".{digits}e" if scientific else f".{digits}f"
    def cell(x):
        if not scientific or x == 0.:
            return format(x if x != 0. else 0., f".{digits}f")
        mantissa, power = format(x, fmt).split("e")
        return mantissa + r"\!\times\!10^{" + str(int(power)) + "}"
    rows = [" & ".join(cell(x) for x in row) for row in np.atleast_2d(A)]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def write_tables(Hs, Rs, ts, centers, errors, A, b, Afree, bfree):
    pieces = [r"\begin{align*}"]
    for i, H in enumerate(Hs, 1):
        pieces.append(f"H_{{{i}}} &= " + tex_matrix(H) + r" \\")
    pieces.append(r"\end{align*}")
    (OUT / "homographies.tex").write_text("\n".join(pieces))
    for start in (0, 5):
        pieces = [r"\begin{align*}"]
        for j in range(start, min(start + 5, 9)):
            E = np.column_stack((Rs[j], ts[j]))
            pieces.append(f"[R_{{{j+1}}}\\mid t_{{{j+1}}}] &= "
                          + tex_matrix(E, 5) + r" \\")
        pieces.append(r"\end{align*}")
        (OUT / f"extrinsics_{start}.tex").write_text("\n".join(pieces))
    rows = []
    for i, (C, e) in enumerate(zip(centers, errors), 1):
        s = error_stats(e)
        rows.append(f"{i} & {s['total']:.6f} & {s['mean']:.6f} & "
                    f"{s['rmse']:.6f} & {s['maximum']:.6f} " + r"\\")
    header = (r"\begin{tabular}{rrrrr}\toprule" + "\n"
              r"View & Sum (px) & Mean (px) & RMS (px) & Maximum (px)"
              r"\\\midrule" + "\n")
    footer = "\n" + r"\bottomrule\end{tabular}" + "\n"
    (OUT / "errors.tex").write_text(header + "\n".join(rows) + footer)
    rows = [str(i) + " & " + " & ".join(f"{x:.4f}" for x in C) + r" \\"
            for i, C in enumerate(centers, 1)]
    header = (r"\begin{tabular}{rrrr}\toprule" + "\n"
              r"View & $C_X$ (mm) & $C_Y$ (mm) & $C_Z$ (mm)"
              r"\\\midrule" + "\n")
    (OUT / "centers.tex").write_text(header + "\n".join(rows) + footer)
    pieces = []
    for name, value in [("Afinal", A), ("Afree", Afree),
                        ("bfinal", b[:, None]), ("bfree", bfree[:, None])]:
        pieces.append("\\newcommand{\\" + name + "}{"
                      + tex_matrix(value, 6, name.startswith("b")) + "}")
    (OUT / "intrinsics.tex").write_text("\n".join(pieces))


def main():
    OUT.mkdir(exist_ok=True)
    validate_math()
    data = np.loadtxt(ROOT / "imgpoints.txt")
    assert data.shape == (432, 2) and np.isfinite(data).all()
    uv, p = data.reshape(9, 48, 2), world_points()
    fits = [homography(p, points) for points in uv]
    Hs = np.array([fit[0] for fit in fits])
    errors = np.array([np.linalg.norm(points - project(H, p), axis=1)
                       for H, points in zip(Hs, uv)])
    Afree, bfree, L, sfree = intrinsic(Hs)
    A, b, _, singular = intrinsic(Hs, zero_skew=True)
    Rs, ts, centers = map(np.array, zip(*(extrinsic(H, A) for H in Hs)))
    X = np.c_[p, np.zeros(len(p))]
    pose_errors, depths = [], []
    for R, t, C, points in zip(Rs, ts, centers, uv):
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.linalg.det(R), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(R @ C + t, 0., atol=1e-10, rtol=0)
        camera_points = X @ R.T + t
        depths.extend(camera_points[:, 2])
        pixels = camera_points @ A.T
        pixels = pixels[:, :2] / pixels[:, 2, None]
        pose_errors.append(np.linalg.norm(points - pixels, axis=1))
    assert min(depths) > 0.
    q = inverse_kinematics()
    null = np.linalg.svd(jacobian(q), full_matrices=True)[2][-1]
    summary = dict(ik_degrees=np.degrees(q).tolist(),
                   ik_position=fk_points(q)[-1].tolist(),
                   ik_null_vector=null.tolist(),
                   homography_error=error_stats(errors),
                   per_view_error=[error_stats(e) for e in errors],
                   intrinsic_zero_skew=A.tolist(),
                   intrinsic_unconstrained=Afree.tolist(),
                   b_zero_skew=b.tolist(), b_unconstrained=bfree.tolist(),
                   camera_centers_mm=centers.tolist(),
                   pose_error=error_stats(np.array(pose_errors)),
                   minimum_depth_mm=float(min(depths)),
                   intrinsic_singular_values=singular.tolist(),
                   unconstrained_singular_values=sfree.tolist(),
                   max_rotation_error=float(max(np.linalg.norm(R.T @ R
                       - np.eye(3)) for R in Rs)),
                   validation="All numerical assertions passed.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(OUT / "calibration.npz", world_points_mm=p, image_points=uv,
             homographies=Hs, A=A, b=b, A_unconstrained=Afree,
             b_unconstrained=bfree, L=L, rotations=Rs, translations_mm=ts,
             camera_centers_mm=centers, homography_errors_px=errors)
    save_figures(p, uv, Hs, errors, Rs, centers)
    write_tables(Hs, Rs, ts, centers, errors, A, b, Afree, bfree)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
