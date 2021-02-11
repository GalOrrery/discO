"""Galpy's SCF Potential

.. warning::

  This will be deprecated as soon as https://github.com/jobovy/galpy/pull/444/
  is merged.


"""

# THIRD PARTY
import astropy.units as u
import numpy
from scipy.special import gamma, gegenbauer, lpmv

##############################################################################
# CODE
##############################################################################


def _RToxi(r, a=1):
    return numpy.divide((r / a - 1.0), (r / a + 1.0))


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
    CC = numpy.zeros((N, L), float)

    for l in range(L):
        for n in range(N):
            a = alpha(l)
            if n == 0:
                CC[n][l] = 1.0
                continue
            elif n == 1:
                CC[n][l] = 2.0 * a * xi
            if n + 1 != N:
                CC[n + 1][l] = (n + 1.0) ** -1.0 * (
                    2 * (n + a) * xi * CC[n][l]
                    - (n + 2 * a - 1) * CC[n - 1][l]
                )
    return CC


def scf_compute_coeffs_spherical_nbody(pos, m, N, a=1.0):
    """Compute SCF Coefficients for Spherical NBody.

    Parameters
    ----------
    pos : Quantity array
        position of particles in your nbody snapshot
    m : Quantity
        masses of particles
    N : int
        size of expansion coefficients
    a : Quantity
        parameter used to shift the basis functions

    """
    Acos = numpy.zeros((N, 1, 1), float)
    Asin = None

    r = numpy.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    Cs = numpy.array([_C(_RToxi(ri, a=a), N, 1)[:, 0] for ri in r])
    RhoSum = numpy.sum(
        (m / (4.0 * numpy.pi) / (r / a + 1))[:, None] * Cs, axis=0
    )
    n = numpy.arange(0, N)
    K = (
        16
        * numpy.pi
        * (n + 3.0 / 2)
        / ((n + 2) * (n + 1) * (1 + n * (n + 3.0) / 2.0))
    )
    Acos[n, 0, 0] = 2 * K * RhoSum

    return Acos, Asin


# /def


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
    """        
        NAME:

           scf_compute_coeffs

        PURPOSE:

           Numerically compute the expansion coefficients for a given triaxial density

        INPUT:

           pos - Positions of particles

           m - mass of particles

           N - size of the Nth dimension of the expansion coefficients

           L - size of the Lth and Mth dimension of the expansion coefficients

           a - parameter used to shift the basis functions

           radial_order - Number of sample points of the radial integral. If None, radial_order=max(20, N + 3/2L + 1)

           costheta_order - Number of sample points of the costheta integral. If None, If costheta_order=max(20, L + 1)

           phi_order - Number of sample points of the phi integral. If None, If costheta_order=max(20, L + 1)

        OUTPUT:

           (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

        HISTORY:

           2020-11-18 - Written - Morgan Bennett

        """

    n = numpy.arange(0, N)
    l = numpy.arange(0, L)
    m = numpy.arange(0, L)

    r = numpy.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    phi = numpy.arctan2(pos[1], pos[0])
    costheta = (pos[2] / r).to_value(u.one)
    """
        Plm= numpy.zeros([len(costheta),1,L,L])
        for i,ll in enumerate(l):
            for j,mm in enumerate(m):
                Plm[:,0,j,i]= lpmv(ll,mm,costheta)

        cosmphi= numpy.cos(phi[:,None]*m[None,:])[:,None,None,:]
        sinmphi= numpy.sin(phi[:,None]*m[None,:])[:,None,None,:]
        
        Ylm= (numpy.sqrt((2.*l[:,None]+1)*gamma(l[:,None]-m[None,:]+1)/gamma(l[:,None]+m[None,:]+1))[None,None,:,:]*Plm)[None,:,:,:,:]*numpy.array([cosmphi,sinmphi])
        Ylm= numpy.nan_to_num(Ylm)

        C= [[gegenbauer(nn,2.*ll+1.5) for ll in l] for nn in n] 
        Cn= numpy.zeros((1,len(r),N,L,1))
        for i in range(N):
            for j in range(L):
                Cn[0,:,i,j,0]= C[i][j]((r/a-1)/(r/a+1))

        rl= ((r[:,None]/a)**l[None,:])[None,:,None,:,None]
        r12l1= ((1+(r[:,None]/a))**(2.*l[None,:]+1))[None,:,None,:,None]

        phinlm= -rl/r12l1*Cn*Ylm

        Sum= numpy.sum(mass[None,:,None,None,None]*phinlm,axis=1)
        Knl= 0.5*n[:,None]*(n[:,None]+4.*l[None,:]+3.)+(l[None,:]+1)*(2.*l[None,:]+1.)

        Inl= (-Knl*4.*numpy.pi/2.**(8.*l[None,:]+6.)*gamma(n[:,None]+4.*l[None,:]+3.)/gamma(n[:,None]+1)/(n[:,None]+2.*l[None,:]+1.5)/gamma(2.*l[None,:]+1.5)**2)[None,:,:,None]

        Anlm= Inl**(-1)*Sum"""
    Anlm = numpy.zeros([2, L, L, L])
    for i, nn in enumerate(n):
        for j, ll in enumerate(l):
            for k, mm in enumerate(m[: j + 1]):

                Plm = lpmv(mm, ll, costheta)

                cosmphi = numpy.cos(phi * mm)
                sinmphi = numpy.sin(phi * mm)

                Ylm = (
                    numpy.sqrt(
                        (2.0 * ll + 1)
                        * gamma(ll - mm + 1)
                        / gamma(ll + mm + 1)
                    )
                    * Plm
                )[None, :] * numpy.array([cosmphi, sinmphi])
                Ylm = numpy.nan_to_num(Ylm)

                C = gegenbauer(nn, 2.0 * ll + 1.5)
                Cn = C(
                    u.Quantity((r / a - 1) / (r / a + 1), copy=False).to_value(
                        u.one
                    )
                )

                phinlm = (
                    -((r / a) ** ll) / (r / a + 1) ** (2.0 * ll + 1) * Cn
                )[None, :] * Ylm

                Sum = numpy.sum(mass[None, :] * phinlm, axis=1)

                Knl = 0.5 * nn * (nn + 4.0 * ll + 3.0) + (ll + 1) * (
                    2.0 * ll + 1.0
                )
                Inl = (
                    -Knl
                    * 4.0
                    * numpy.pi
                    / 2.0 ** (8.0 * ll + 6.0)
                    * gamma(nn + 4.0 * ll + 3.0)
                    / gamma(nn + 1)
                    / (nn + 2.0 * ll + 1.5)
                    / gamma(2.0 * ll + 1.5) ** 2
                )

                Anlm[:, i, j, k] = Inl ** (-1) * Sum

    return 2.0 * Anlm


# /def
