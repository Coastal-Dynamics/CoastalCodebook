import numpy as np


def disper(w, h, g=9.8):
    """
    DISPER  Linear dispersion relation.

    absolute error in k*h < 5.0e-16 for all k*h

    Syntax:
    k = disper(w, h, [g])

    Input:
    w = 2*pi/T, were T is wave period
    h = water depth
    g = gravity constant

    Output:
    k = wave number

    Example
    k = disper(2*pi/5,5,g = 9.81);

    Copyright notice
    --------------------------------------------------------------------
    Copyright (C)
    G. Klopman, Delft Hydraulics, 6 Dec 1994
    M. van der Lugt conversion to python, 11 Jan 2021

    """
    # make sure numpy array
    listType = type([1, 2])
    Type = type(w)

    w = np.atleast_1d(w)

    # check to see if warning disappears
    wNul = w == 0
    w[w == 0] = np.nan

    w2 = w**2 * h / g
    q = w2 / (1 - np.exp(-(w2 ** (5 / 4)))) ** (2 / 5)

    for j in np.arange(0, 2):
        thq = np.tanh(q)
        thq2 = 1 - thq**2
        aa = (1 - q * thq) * thq2

        # prevent warnings, we don't apply aa<0 anyway
        aa[aa < 0] = 0

        bb = thq + q * thq2
        cc = q * thq - w2

        D = bb**2 - 4 * aa * cc

        # initialize argument with the exception
        arg = -cc / bb

        # only execute operation on those entries where no division by 0
        ix = np.abs(aa * cc) >= 1e-8 * bb**2
        arg[ix] = (-bb[ix] + np.sqrt(D[ix])) / (2 * aa[ix])

        q = q + arg

    k = np.sign(w) * q / h

    # set 0 back to 0
    k = np.where(wNul, 0, k)

    # if input was a list, return also as list
    if Type == listType:
        k = list(k)
    elif len(k) == 1:
        k = k[0]

    return k


def find_cb_phib(
    phi0, H0, T, g=9.81, gamma=0.8, hb=np.arange(0.1, 5.0, 0.01), print_report=False
):
    """
    Returns breaking wave celerity cb [m/s] and angle of incidence at breaking phib [degrees] for given:
    - phi0 : angle of incidence [degrees]
    - H0   : deep water wave height [m]
    - T    : period [s]

    The parameter hb_guess is used as guessed values for the breaking depth.
    From this array, the best-fitting value is chosen in the end. You can adjust this
    array to make estimates more accurate at the cost of computational efficiency.
    """
    # First convert the angle of incidence to radians
    phi_rad = phi0 / 360 * 2 * np.pi

    # We start with calculating deep water celerity, wavelength, and angular frequency
    c0 = g * T / (2 * np.pi)
    L0 = c0 * T
    w = T / (2 * np.pi)

    # For every value of hb_guess, the wavenumber k is determined using the dispersion relation
    k = disper(w, hb, g=g)  # Feel free to use your own implementation from week 2!

    # Next we calculate the celerity and group celerity for each breaking depth
    c = np.sqrt(g / k * np.tanh(k * hb))
    n = 1 / 2 * (1 + (2 * k * hb) / (np.sinh(2 * k * hb)))
    cg = n * c

    # In order to correctly shoal the waves, we also need the deep water group celerity
    n0 = 1 / 2
    cg0 = n0 * c0

    # And to account for refraction we need the angle of incidence at breaking using Snell's law
    phi = np.arcsin(np.sin(phi_rad) / c0 * c)

    # Shoaling & refraction coefficients
    Ksh = np.sqrt(cg0 / cg)
    Kref = np.sqrt(np.cos(phi_rad) / np.cos(phi))

    # Wave heights Hb at depth hb
    Hb = Ksh * Kref * H0

    # We are looking for an hb where the breaker parameter is 0.8
    # We can determine which value of hb in our array gets closest using the
    # following line of code:
    i = np.argmin(np.abs(Hb / hb - gamma))
    Hb_pred, hb_pred = Hb[i], hb[i]

    # Let's print what we found
    if print_report:
        print(f"predicted breaking depth: {hb_pred:.2f} m")
        print(f"predicted breaking wave height: {Hb_pred:.2f} m")
        print(f"gamma = {Hb_pred / hb_pred:.2f} [-]")

    # And finally return the associated value cb for the celerity at breaking, as well as the angle of incidence at breaking phib
    return c[i], phi[i] / (2 * np.pi) * 360


def CERC(cb, phi0, H0, K=0.7, s=2.65, p=0.4):
    """
    cb:   celerity at breaking
    phi0: offshore angle of incidence (degrees)
    H0:   offshore wave height

    K:    coefficient
    s:    relative density
    p:    porosity
    """

    return (
        K
        / (32 * (s - 1) * (1 - p))
        * cb
        * np.sin(2 * (phi0 / 360 * 2 * np.pi))
        * H0**2
    )


def get_S_coastline(coastline_orientation, wave_climate):
    """
    Returns yearly transport for angle phi [degrees]

    Transport is already scaled for the relative occurrence of the conditions.
    """
    total_transport = 0

    for index, row in wave_climate.iterrows():
        Hs, angle, days = row

        # Our formulation of the CERC formula (the choice of K) was based on Hrms, so we determine Hrms from Hs
        Hrms = Hs / np.sqrt(2)
        cb, phib = find_cb_phib(angle - coastline_orientation, Hrms, 7)
        S = CERC(cb, angle - coastline_orientation, Hrms)

        total_transport += days / 365.25 * S
    return total_transport * 365.25 * 24 * 3600
