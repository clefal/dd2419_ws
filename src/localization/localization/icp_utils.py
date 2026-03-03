#!/usr/bin/env python3
import math
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as e:
    cKDTree = None


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def pose_to_T(tx: float, ty: float, theta: float) -> np.ndarray:
    T = np.eye(3, dtype=np.float64)
    T[0:2, 0:2] = rot2(theta)
    T[0:2, 2] = [tx, ty]
    return T


def T_to_pose(T: np.ndarray):
    tx = float(T[0, 2])
    ty = float(T[1, 2])
    theta = math.atan2(T[1, 0], T[0, 0])
    return tx, ty, theta


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[0:2, 0:2]
    t = T[0:2, 2]
    Ti = np.eye(3, dtype=np.float64)
    Ti[0:2, 0:2] = R.T
    Ti[0:2, 2] = -R.T @ t
    return Ti


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts shape (N,2) -> returns (N,2)"""
    return (pts @ T[0:2, 0:2].T) + T[0:2, 2]


def best_fit_transform_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Find T that minimizes || (T*src) - dst || in least squares.
    src,dst: (N,2) matched pairs
    Returns 3x3 T
    """
    assert src.shape == dst.shape
    N = src.shape[0]
    if N < 3:
        return np.eye(3, dtype=np.float64)

    mu_s = np.mean(src, axis=0)
    mu_d = np.mean(dst, axis=0)

    X = src - mu_s
    Y = dst - mu_d

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection fix
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = mu_d - (R @ mu_s)

    T = np.eye(3, dtype=np.float64)
    T[0:2, 0:2] = R
    T[0:2, 2] = t
    return T


def icp_2d_point_to_point(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    init_T: np.ndarray = None,
    max_iter: int = 25,
    max_corr_dist: float = 0.35,
    tol: float = 1e-4,
    min_inliers: int = 40,
):
    """
    Align src_pts to dst_pts. Returns:
      T (3x3) mapping src -> dst,
      info dict: {converged, rmse, inliers}
    """
    if init_T is None:
        T = np.eye(3, dtype=np.float64)
    else:
        T = init_T.copy()

    if cKDTree is None:
        raise RuntimeError("scipy not available. Install scipy or use a custom KD-tree.")

    if src_pts.shape[0] < min_inliers or dst_pts.shape[0] < min_inliers:
        return T, {"converged": False, "rmse": float("inf"), "inliers": 0}

    tree = cKDTree(dst_pts)

    prev_rmse = None
    converged = False
    best = (T.copy(), float("inf"), 0)

    for _ in range(max_iter):
        src_trans = apply_T(T, src_pts)

        dists, idx = tree.query(src_trans, k=1)
        mask = dists < max_corr_dist

        inliers = int(np.sum(mask))
        if inliers < min_inliers:
            break

        matched_src = src_trans[mask]
        matched_dst = dst_pts[idx[mask]]

        # Compute incremental transform that maps matched_src -> matched_dst
        dT = best_fit_transform_2d(matched_src, matched_dst)

        # Update: src_pts -> (dT * T) -> dst
        T = dT @ T

        rmse = float(np.sqrt(np.mean(dists[mask] ** 2)))

        if rmse < best[1]:
            best = (T.copy(), rmse, inliers)

        if prev_rmse is not None and abs(prev_rmse - rmse) < tol:
            converged = True
            break
        prev_rmse = rmse

    T_best, rmse_best, inliers_best = best
    return T_best, {"converged": converged, "rmse": rmse_best, "inliers": inliers_best}