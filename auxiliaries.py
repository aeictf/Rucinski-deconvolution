"""
- ew_abs_lines: computes EW of absorption features for normalized spectra.
- adjust_model : iteratively scales a normalized transmission model
  so that its equivalent width (EW) over [lamdeb, lamend] matches the data's EW.
- convol_by_psf: local convolution of a model spectrum with a point-spread
  function (PSF) sampled on a local wavelength offset grid.


"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable, Optional
import numpy as np
from scipy.interpolate import UnivariateSpline

ArrayLike = Iterable[float]


def ew_abs_lines(x: ArrayLike, y: ArrayLike, lamdeb: float, lamend: float) -> float:
    """Compute equivalent width (EW) of absorption lines on a *normalized* spectrum.

    Parameters
    ----------
    x, y : array-like
        Wavelengths and normalized fluxes (continuum ≈ 1).
    lamdeb, lamend : float
        Integration limits in same units as ``x``.

    Returns
    -------
    float
        Equivalent width: :math:`\int_{lamdeb}^{lamend} (1 - y(\lambda)) d\lambda`.

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if lamdeb > lamend:
        lamdeb, lamend = lamend, lamdeb

    mask = (x >= lamdeb) & (x <= lamend)
    if mask.sum() < 2:
        # Try to pad by including closest bracketing points for safe trapezoid integration
        # if the interval lies inside x's span.
        if lamdeb < x.min() or lamend > x.max():
            return float("nan")
        # Insert the bounds by linear interpolation
        xi = np.array([lamdeb, lamend])
        yi = np.interp(xi, x, y)
        x_int = np.r_[lamdeb, lamend]
        y_int = yi
    else:
        # Ensure the exact bounds are included for accurate integration
        xi = np.array([lamdeb, lamend])
        yi = np.interp(xi, x, y)
        x_int = np.r_[lamdeb, x[mask], lamend]
        y_int = np.r_[yi[0], y[mask], yi[1]]

    return float(np.trapz(1.0 - y_int, x_int))


def adjust_model(
    wavexmod, waveymod, wavexdata, waveydata, lamdeb, lamend,
    *,
    tol1=0.01, tol2=0.001, max_iter1=200, max_iter2=200,
    verbose=False, return_history=True
):
    """
    -  iteratively raises the model transmission to a power of <1 (0.95/0.99) if EW(model) > EW(data) or to a power of >1 (1.05/1.01) if EW(model) < EW(data), only for elements > 0 (otherwise sets 0), until |EW(model)/EW(data) - 1| becomes < tol.
    -   TWO passes: coarse (0.95/1.05) and fine (0.99/1.01).

    """
    xmod = np.asarray(wavexmod, float)
    ymod = np.asarray(waveymod, float).copy()
    xdat = np.asarray(wavexdata, float)
    ydat = np.asarray(waveydata, float)

    # Проверки/подготовка
    ew_data = ew_abs_lines(xdat, ydat, lamdeb, lamend)
    if not np.isfinite(ew_data) or ew_data == 0.0:
        raise ValueError("EW(data) 0 or NAN. Impossible to adjust.")

    def _iterate(y, exp_less, exp_more, tol, max_iter, phase_name):
        history = []
        for ii in range(max_iter):
            ew_mod = ew_abs_lines(xmod, y, lamdeb, lamend)
            rat = ew_mod / ew_data
            history.append(float(rat))
            if verbose:
                print(f"{phase_name} iter {ii}: RAT={rat:.6f}")
            if abs(rat - 1.0) < tol:
                break
            # IGOR: wave = (wave>0) ? wave^p : 0
            mask = y > 0
            if ew_mod > ew_data:
                # уменьшить EW → сделать поток выше: степень < 1
                y[~mask] = 0.0
                y[mask] = y[mask] ** exp_less
            else:
                # увеличить EW → сделать поток ниже: степень > 1
                y[~mask] = 0.0
                y[mask] = y[mask] ** exp_more
        return y, history

    # Первый цикл (грубая подгонка): 0.95 / 1.05, порог 1%
    ymod, h1 = _iterate(ymod, exp_less=0.95, exp_more=1.05, tol=tol1, max_iter=max_iter1, phase_name="coarse")

    # Второй цикл (точная подгонка): 0.99 / 1.01, порог 0.1‰
    ymod, h2 = _iterate(ymod, exp_less=0.99, exp_more=1.01, tol=tol2, max_iter=max_iter2, phase_name="fine")

    # Итоговые значения
    ew_mod_final = ew_abs_lines(xmod, ymod, lamdeb, lamend)
    rat_final = ew_mod_final / ew_data

    result = {
        "y_model_scaled": ymod,
        "EW_model": float(ew_mod_final),
        "EW_data": float(ew_data),
        "ratio": float(rat_final),
        "iterations": {"coarse": len(h1), "fine": len(h2)},
    }
    if return_history:
        result["ratio_history"] = {"coarse": h1, "fine": h2}
    return result

def convol_by_psf(
    W_psf: ArrayLike,
    w_grid: ArrayLike,
    lmd: float,
    trmodel: ArrayLike,
    lammodel: ArrayLike,
    *,
    normalize_psf: bool = False,
    fill_value: Optional[float] = None,
) -> float:
    """Convolution of TAPAS-like model with a local PSF sampled on a *local* grid.

    - Let ``w_grid`` be the *local* wavelength offsets (not necessarily regular).
    - Define ``lam_k = lmd + w_grid``.
    - Sample the model at ``lam_k`` via linear interpolation.
    - Return ``∫ W_psf(w) * model(lmd + w) dw`` over ``w_grid``.

    Parameters
    ----------
    W_psf : array-like
        Discrete PSF values on ``w_grid``.
    w_grid : array-like
        Offsets (in wavelength units) relative to center ``lmd``.
    lmd : float
        Central wavelength where the convolution is evaluated.
    trmodel, lammodel : array-like
        Model values and their wavelengths.
    normalize_psf : bool, default False
        If True, normalize PSF so ``∫ W_psf dw = 1`` over ``w_grid``.
    fill_value : float or None
        Value to use outside the model grid. If None, edge values are used.

    Returns
    -------
    float
        Convolved model value at ``lmd``.
    """
    W_psf = np.asarray(W_psf, dtype=float)
    w_grid = np.asarray(w_grid, dtype=float)
    lammodel = np.asarray(lammodel, dtype=float)
    trmodel = np.asarray(trmodel, dtype=float)

    lam_k = lmd + w_grid
    if fill_value is None:
        left = trmodel[0]
        right = trmodel[-1]
    else:
        left = right = float(fill_value)

    v_tapas = np.interp(lam_k, lammodel, trmodel, left=left, right=right)

    if normalize_psf:
        area_psf = np.trapz(W_psf, w_grid)
        if area_psf != 0.0:
            W_psf = W_psf / area_psf

    return float(np.trapz(W_psf * v_tapas, w_grid))


def convolve_model_with_psf(
    lammodel: ArrayLike,
    trmodel: ArrayLike,
    w_grid: ArrayLike,
    W_psf: ArrayLike,
    lam_eval: ArrayLike,
    *,
    normalize_psf: bool = False,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """Vectorized helper to evaluate ``convol_by_psf`` on many centers ``lam_eval``."""
    lam_eval = np.asarray(lam_eval, dtype=float)
    out = np.empty_like(lam_eval)
    for i, lmd in enumerate(lam_eval):
        out[i] = convol_by_psf(
            W_psf=W_psf,
            w_grid=w_grid,
            lmd=float(lmd),
            trmodel=trmodel,
            lammodel=lammodel,
            normalize_psf=normalize_psf,
            fill_value=fill_value,
        )
    return out



# --------------------------
# Example usage / quick test
# --------------------------
if __name__ == "__main__":
    # Create a toy normalized Gaussian absorption line as the *model*.
    lam = np.linspace(6296.0, 6313.5, 4096)
    depth = 0.5
    center = 6305.0
    sigma = 0.2
    model = 1.0 - depth * np.exp(-0.5 * ((lam - center) / sigma) ** 2)

    # Make a "data" variant with slightly deeper line
    data = 1.0 - 0.65 * np.exp(-0.5 * ((lam - center) / sigma) ** 2)

    lamdeb, lamend = 6296.8, 6312.9
    ew_m = ew_abs_lines(lam, model, lamdeb, lamend)
    ew_d = ew_abs_lines(lam, data, lamdeb, lamend)

    res = adjust_model(lam, model, lam, data, lamdeb, lamend, mode="scale_depth")
    print({"EW_model": ew_m, "EW_data": ew_d, "scale": res.scale})
    # Convolution with a simple PSF on a local grid
    w = np.linspace(-0.5, 0.5, 101)
    psf = np.exp(-0.5 * (w / 0.1) ** 2)
    val = convol_by_psf(psf, w, center, model, lam, normalize_psf=True)
    print({"convolved_at_center": val})
