"""
Notes :
- `fluxobs`, `lambdaobs`, `transmod`, `lammod` are NumPy arrays (not strings).
- Data/model must be normalized in the fitting window (continuum ≈ 1). If needed,
  adjust model EW beforehand (with adjust_model from auxiliaries.py).
- Gaussian width is estimated from second moments (robust, no external optimizer),

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BuildDESOutput:
    DES: np.ndarray              # (NN, m) design matrix
    B_obs: np.ndarray            # (NN,) observed flux vector (window)
    L_obs: np.ndarray            # (NN,) observed wavelength vector (window)
    L_grid: np.ndarray           # (m,) local PSF offset grid
    dl_grid: float               # scalar grid step for L_grid
    U: np.ndarray                # (NN, m)
    S: np.ndarray                # (m,)
    VT: np.ndarray               # (m, m)
    vect_instrum: np.ndarray     # (m,) full (least-squares) LSF solution
    conv_spect: np.ndarray       # (NN,) reconstructed spectrum DES @ vect_instrum
    b_profiles: np.ndarray       # (m, m) columns = LSF using 1..m singular values
    first_max_index: int         # index from the profile-length heuristic
    second_min_index: int        # index of secondary minimum after first max
    best_profile: np.ndarray     # (m,) chosen LSF normalized to sum = 1
    sigma_best: float            # (in wavelength units of L_grid)
    resolution_R: float          # ≈ lambda_center / FWHM (Gaussian)
    lambda_center: float         # central wavelength used in R estimate


def build_matrix_DES(
    p_start: int,
    p_end: int,
    N_psf: int,
    l_span: float,
    fluxobs: np.ndarray,
    lambdaobs: np.ndarray,
    transmod: np.ndarray,
    lammod: np.ndarray,
    *,
    fill_value: Optional[float] = None,
) -> BuildDESOutput:
    """
    Parameters
    ----------
    p_start, p_end : int
        First/last indices (inclusive) of the observed window.
    N_psf : int
        Number of points describing the LSF (recommended odd).
    l_span : float
        Total span of the local PSF support (same units as wavelengths).
    fluxobs, lambdaobs : array-like
        Observed (normalized) flux and wavelength arrays.
    transmod, lammod : array-like
        High-resolution model transmission and its wavelength grid.
    fill_value : float or None
        Value used outside model grid during interpolation. If None, edge values are used.

    Returns
    -------
    BuildDESOutput
        Structured result containing DES, solutions, chosen PSF and resolution.
    """
    # Shapes & window
    assert 0 <= p_start <= p_end < len(lambdaobs), "Bad p_start/p_end"
    NN = int(p_end - p_start + 1)
    m = int(N_psf)
    if m < 2 or NN < 1:
        raise ValueError("Invalid sizes for N_psf or selected window")

    # Local PSF offset grid (regular, centered)
    dl_grid = float(l_span) / (m - 1)
    L_grid = dl_grid * np.arange(m, dtype=float) - l_span / 2.0

    # Slice observed window
    L_obs = np.asarray(lambdaobs[p_start : p_end + 1], dtype=float)
    B_obs = np.asarray(fluxobs[p_start : p_end + 1], dtype=float)

    lammod = np.asarray(lammod, dtype=float)
    transmod = np.asarray(transmod, dtype=float)
    if fill_value is None:
        left = float(transmod[0])
        right = float(transmod[-1])
    else:
        left = right = float(fill_value)

    # Build DES
    DES = np.empty((NN, m), dtype=float)
    for i in range(NN):
        lam_0 = L_obs[i]
        lam_k = lam_0 + L_grid
        v_tapas = np.interp(lam_k, lammod, transmod, left=left, right=right)
        DES[i, :] = v_tapas * dl_grid

    # SVD solve DES * x ≈ B_obs
    U, S, VT = np.linalg.svd(DES, full_matrices=False)
    # Back-substitution (Moore–Penrose)
    x_full = VT.T @ ((U.T @ B_obs) / S)
    vect_instrum = x_full.copy()
    conv_spect = DES @ vect_instrum

    # Truncated SVD sweep: columns j = solution using first (j+1) singular values
    b_profiles = np.empty((m, m), dtype=float)
    for j in range(m):
        k = j + 1
        Uk = U[:, :k]
        Sk = S[:k]
        Vk = VT[:k, :].T  # (m, k)
        x_k = Vk @ ((Uk.T @ B_obs) / Sk)
        b_profiles[:, j] = x_k

    # Heuristic search for optimal PSF profile among columns of b_profiles
    first_max_index, second_min_index, best_profile = _search_best_line_and_psf(b_profiles)

    # Normalize best_profile to unit area on L_grid
    s = best_profile.sum()
    if s != 0:
        best_profile = best_profile / s

    # Gaussian-moment width & spectral resolution
    sigma_best = _sigma_from_moments(L_grid, best_profile)
    lambda_center = float(lambdaobs[int(round(p_start + 0.5 * (p_end - p_start)))])
    R = _resolution_from_sigma(lambda_center, sigma_best)

    return BuildDESOutput(
        DES=DES,
        B_obs=B_obs,
        L_obs=L_obs,
        L_grid=L_grid,
        dl_grid=float(dl_grid),
        U=U,
        S=S,
        VT=VT,
        vect_instrum=vect_instrum,
        conv_spect=conv_spect,
        b_profiles=b_profiles,
        first_max_index=int(first_max_index),
        second_min_index=int(second_min_index),
        best_profile=best_profile,
        sigma_best=float(sigma_best),
        resolution_R=float(R),
        lambda_center=lambda_center,
    )


# ----------------------
# Helper implementations
# ----------------------

def _total_length(wavex: np.ndarray, wavey: np.ndarray) -> float:
    """Length of `wavey(wavex)` """
    x = np.asarray(wavex, dtype=float)
    y = np.asarray(wavey, dtype=float)
    if x.size < 2:
        return 0.0
    ascal = (y.mean() / x.mean()) if x.mean() != 0 else 1.0
    dx = np.diff(x)
    dy = np.diff(y) / ascal
    return float(np.sqrt(dx * dx + dy * dy).sum())


def _search_best_line_and_psf(D: np.ndarray):
    """ Search for the best PSF profile. Returns indexes and profile.
    """
    D = np.asarray(D, dtype=float)
    NN, m = D.shape
    wavex = np.arange(NN, dtype=float)

    leline = np.empty(m, dtype=float)
    for j in range(m):
        col = D[:, j].astype(float)
        s = col.sum()
        if s != 0:
            col /= s
        leline[j] = _total_length(wavex, col)

    # First maximum while leline increases with j
    first_max = 1
    for j in range(1, min(100, m)):
        if leline[j] > leline[j - 1]:
            first_max = j
        else:
            break

    # Secondary minimum after first_max
    tail = leline[first_max:]
    second_min_rel = int(np.argmin(tail))
    second_min = first_max + second_min_rel

    best_profile = D[:, second_min].astype(float)
    s = best_profile.sum()
    if s != 0:
        best_profile /= s
    return int(first_max), int(second_min), best_profile


def _sigma_from_moments(x: np.ndarray, y: np.ndarray) -> float:
    """Second-moment standard deviation of y(x) assuming y ≥ 0 and ∑y = 1."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = y.sum()
    if s <= 0:
        return float("nan")
    mu = float((x * y).sum() / s)
    var = float(((x - mu) ** 2 * y).sum() / s)
    return float(np.sqrt(max(var, 0.0)))


def _resolution_from_sigma(lambda_center: float, sigma: float) -> float:
    """R ≈ λ / FWHM, with FWHM = 2 * sqrt(ln 2) * sigma for a Gaussian PSF."""
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    fwhm = 2.0 * np.sqrt(np.log(2.0)) * float(sigma)
    return float(lambda_center / fwhm)


# ---------
# Example
# ---------
if __name__ == "__main__":
    # Tiny synthetic demo showing the function call; replace with real arrays.
    rng = np.random.default_rng(0)
    lam_obs = np.linspace(630.0, 630.5, 4001)
    true_sigma = 0.02
    center = 630.25
    depth = 0.5
    flux_obs = 1.0 - depth * np.exp(-0.5 * ((lam_obs - center) / true_sigma) ** 2)

    out = build_matrix_DES(
        p_start=307441,
        p_end=308850,
        N_psf=101,
        l_span=0.25,
        fluxobs=flux_obs,
        lambdaobs=lam_obs,
        transmod=trans_mod,
        lammod=lam_mod,
    )
    print({
        "DES": out.DES.shape,
        "best_profile_len": out.best_profile.size,
        "first_max_index": out.first_max_index,
        "second_min_index": out.second_min_index,
        "R": out.resolution_R,
    })
