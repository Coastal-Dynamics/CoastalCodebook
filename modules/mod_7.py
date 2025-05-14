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
