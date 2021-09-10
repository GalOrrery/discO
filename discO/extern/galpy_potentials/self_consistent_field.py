# -*- coding: utf-8 -*-
"""Galpy's SCF Potential

.. warning::

  This will be deprecated as soon as https://github.com/jobovy/galpy/pull/444/
  is merged.


"""

# THIRD PARTY
import astropy.units as u
import numpy as np
from scipy.special import gamma, gegenbauer, lpmv

##############################################################################
# CODE
##############################################################################


def _RToxi(r, a=1):
    return np.divide((r / a - 1.0), (r / a + 1.0))


def _C(xi, N, L, alpha=lambda x: 2 * x + 3.0 / 2):
    """
    NAME:
       _C
    PURPOSE:
       Evaluate C_n,l (the Gegenbauer polynomial) for 0 <= l < L and 0<= n < N
    INPUT:
       xi - radial transformed variable
       N - Size of the N dimension
       L - Size of the L dimension
       alpha = A lambda function of l. Default alpha = 2l + 3/2

    OUTPUT:
       An LxN Gegenbauer Polynomial
    HISTORY:
       2016-05-16 - Written - Aladdin

    """
    CC = np.zeros((N, L), float)

    for ll in range(L):
        for n in range(N):
            a = alpha(ll)
            if n == 0:
                CC[n][ll] = 1.0
                continue
            elif n == 1:
                CC[n][ll] = 2.0 * a * xi
            if n + 1 != N:
                CC[n + 1][ll] = (n + 1.0) ** -1.0 * (
                    2 * (n + a) * xi * CC[n][ll] - (n + 2 * a - 1) * CC[n - 1][ll]
                )
    return CC


def scf_compute_coeffs_nbody(
    pos,
    mass,
    N,
    L,
    a=1.0,
    radial_order=None,
    costheta_order=None,
    phi_order=None,
):
    """Compute SCF Coefficients

    Numerically compute the expansion coefficients for a given triaxial
    density

    Parameters
    ----------
    pos : (3, N) array
        Positions of particles
    m : scalar or (N,) array
        mass of particles
    N : int
        size of the Nth dimension of the expansion coefficients
    L : int
        size of the Lth and Mth dimension of the expansion coefficients
    a : float or Quantity
        parameter used to shift the basis functions

    Returns
    -------
    Acos, Asin : array
        Expansion coefficients for density dens that can be given to
        ``SCFPotential.__init__``

    .. versionadded:: 1.7
       2020-11-18 - Written - Morgan Bennett

    """
    mass = mass.to_value(1e12 * u.solMass)  # :(
    ns = np.arange(0, N)
    ls = np.arange(0, L)
    ms = np.arange(0, L)

    ra = (np.sqrt(np.sum(np.square(pos), axis=0)) / a).to_value(u.one)
    phi = np.arctan2(pos[1], pos[0])
    costheta = (pos[2] / ra / a).to_value(u.one)

    Anlm = np.zeros([2, N, L, L])
    for i, nn in enumerate(ns):
        for j, ll in enumerate(ls):
            for k, mm in enumerate(ms[: j + 1]):

                Plm = lpmv(mm, ll, costheta)

                cosmphi = np.cos(phi * mm)
                sinmphi = np.sin(phi * mm)

                Ylm = (np.sqrt((2.0 * ll + 1) * gamma(ll - mm + 1) / gamma(ll + mm + 1)) * Plm)[
                    None,
                    :,
                ] * np.array([cosmphi, sinmphi])
                Ylm = np.nan_to_num(Ylm)

                C = gegenbauer(nn, 2.0 * ll + 1.5)
                Cn = C(
                    u.Quantity((ra - 1) / (ra + 1), copy=False).to_value(
                        u.one,
                    ),
                )

                phinlm = (-np.power(ra, ll) / np.power(ra + 1, (2.0 * ll + 1)) * Cn)[None, :] * Ylm

                Sum = np.sum(mass[None, :] * phinlm, axis=1)

                Knl = 0.5 * nn * (nn + 4.0 * ll + 3.0) + (ll + 1) * (2.0 * ll + 1.0)
                Inl = (
                    -Knl
                    * 4.0
                    * np.pi
                    / 2.0 ** (8.0 * ll + 6.0)
                    * gamma(nn + 4.0 * ll + 3.0)
                    / gamma(nn + 1)
                    / (nn + 2.0 * ll + 1.5)
                    / gamma(2.0 * ll + 1.5) ** 2
                )

                Anlm[:, i, j, k] = Inl ** (-1) * Sum

    return 2.0 * Anlm


# /def
