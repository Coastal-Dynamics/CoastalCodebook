import numpy as np


def load_table_h_L0():
    # load first column of table A.3, page 519, of the open textbook
    h_L0_series = np.array(
        [
            0,
            0.002,
            0.004,
            0.006,
            0.008,
            0.01,
            0.015,
            0.02,
            0.025,
            0.03,
            0.035,
            0.04,
            0.045,
            0.05,
            0.055,
            0.06,
            0.065,
            0.07,
            0.075,
            0.08,
            0.085,
            0.09,
            0.095,
            0.1,
            0.11,
            0.12,
            0.13,
            0.14,
            0.15,
            0.16,
            0.17,
            0.18,
            0.19,
            0.2,
            0.21,
            0.22,
            0.23,
            0.24,
            0.25,
            0.26,
            0.27,
            0.28,
            0.29,
            0.30,
            0.31,
            0.32,
            0.33,
            0.34,
            0.35,
            0.36,
            0.37,
            0.38,
            0.39,
            0.4,
            0.41,
            0.42,
            0.43,
            0.44,
            0.45,
            0.46,
            0.47,
            0.48,
            0.49,
            0.5,
            1,
        ]
    )
    return h_L0_series


def load_table_tanhkh():
    # load 2nd column of table A.3, page 519, of the open textbook
    tanhkh_series = np.array(
        [
            0,
            0.112,
            0.158,
            0.193,
            0.222,
            0.248,
            0.302,
            0.347,
            0.386,
            0.420,
            0.452,
            0.48,
            0.507,
            0.531,
            0.554,
            0.575,
            0.595,
            0.614,
            0.632,
            0.649,
            0.665,
            0.681,
            0.695,
            0.709,
            0.735,
            0.759,
            0.780,
            0.8,
            0.818,
            0.835,
            0.85,
            0.864,
            0.877,
            0.888,
            0.899,
            0.909,
            0.918,
            0.926,
            0.933,
            0.940,
            0.946,
            0.952,
            0.957,
            0.961,
            0.965,
            0.969,
            0.972,
            0.975,
            0.978,
            0.980,
            0.983,
            0.984,
            0.986,
            0.988,
            0.989,
            0.990,
            0.991,
            0.992,
            0.993,
            0.994,
            0.995,
            0.995,
            0.996,
            0.996,
            1,
        ]
    )

    return tanhkh_series


def load_table_h_L():
    # load third column of table A.3, page 519, of the open textbook
    h_L_series = np.array(
        [
            0,
            0.0179,
            0.0253,
            0.0311,
            0.0360,
            0.0403,
            0.0496,
            0.0576,
            0.0648,
            0.0713,
            0.0775,
            0.0833,
            0.0888,
            0.0942,
            0.0993,
            0.104,
            0.109,
            0.114,
            0.119,
            0.123,
            0.128,
            0.132,
            0.137,
            0.141,
            0.150,
            0.158,
            0.167,
            0.175,
            0.183,
            0.192,
            0.2,
            0.208,
            0.217,
            0.225,
            0.234,
            0.242,
            0.251,
            0.259,
            0.268,
            0.277,
            0.285,
            0.294,
            0.303,
            0.312,
            0.321,
            0.330,
            0.339,
            0.349,
            0.358,
            0.367,
            0.377,
            0.386,
            0.395,
            0.405,
            0.415,
            0.424,
            0.434,
            0.433,
            0.453,
            0.463,
            0.472,
            0.482,
            0.492,
            0.502,
            1,
        ]
    )
    return h_L_series


def load_table_n():
    # load 8th column of table A.3, page 519, of the open textbook
    n_series = np.array(
        [
            1,
            0.9958,
            0.9917,
            0.9875,
            0.9834,
            0.9793,
            0.969,
            0.9588,
            0.9488,
            0.9389,
            0.9289,
            0.9193,
            0.9095,
            0.8998,
            0.8905,
            0.8812,
            0.8719,
            0.8627,
            0.8538,
            0.8448,
            0.8358,
            0.8272,
            0.8188,
            0.8102,
            0.7937,
            0.7775,
            0.7611,
            0.7468,
            0.7329,
            0.7195,
            0.7041,
            0.6918,
            0.68,
            0.6687,
            0.6559,
            0.6458,
            0.6362,
            0.6253,
            0.6169,
            0.6073,
            0.5999,
            0.5915,
            0.5851,
            0.5778,
            0.5711,
            0.5649,
            0.5602,
            0.5549,
            0.55,
            0.5455,
            0.5414,
            0.5377,
            0.5348,
            0.5316,
            0.5287,
            0.526,
            0.5232,
            0.5211,
            0.5191,
            0.5173,
            0.5156,
            0.5141,
            0.5128,
            0.5116,
            0.5,
        ]
    )
    return n_series


def find_interpolate_table(h_L0, tanhkh, h_L, n):
    h_L0_series = load_table_h_L0()
    tanhkh_series = load_table_tanhkh()
    h_L_series = load_table_h_L()
    n_series = load_table_n()

    if h_L0 != None:
        in_series = h_L0_series
        in_value = h_L0
        out1_series = tanhkh_series
        out2_series = h_L_series
        out3_series = n_series
        known = 1
    if tanhkh != None:
        in_series = tanhkh_series
        in_value = tanhkh
        out1_series = h_L0_series
        out2_series = h_L_series
        out3_series = n_series
        known = 2
    if h_L != None:
        in_series = h_L_series
        in_value = h_L
        out1_series = h_L0_series
        out2_series = tanhkh_series
        out3_series = n_series
        known = 3

    id_lower = np.where(in_series <= in_value)[0][-1]

    # get the (last) id where the value is below input value
    out1_lower = out1_series[id_lower]
    out1_upper = out1_series[id_lower + 1]
    out2_lower = out2_series[id_lower]
    out2_upper = out2_series[id_lower + 1]
    out3_lower = out3_series[id_lower]
    out3_upper = out3_series[id_lower + 1]

    # linear interpolation for first output
    slope1 = (out1_upper - out1_lower) / (in_series[id_lower + 1] - in_series[id_lower])
    out1_value = out1_lower + slope1 * (in_value - in_series[id_lower])

    # linear interpolation for second output
    slope2 = (out2_upper - out2_lower) / (in_series[id_lower + 1] - in_series[id_lower])
    out2_value = out2_lower + slope2 * (in_value - in_series[id_lower])

    # linear interpolation for 3rd output
    slope3 = (out3_upper - out3_lower) / (in_series[id_lower + 1] - in_series[id_lower])
    out3_value = out3_lower + slope3 * (in_value - in_series[id_lower])

    if known == 1:
        return1 = in_value
        return2 = out1_value
        return3 = out2_value
        return4 = out3_value
    if known == 2:
        return2 = in_value
        return1 = out1_value
        return3 = out2_value
        return4 = out3_value
    if known == 3:
        return3 = in_value
        return1 = out1_value
        return2 = out2_value
        return4 = out3_value

    return return1, return2, return3, return4
