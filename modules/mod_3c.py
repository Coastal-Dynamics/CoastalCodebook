from warnings import filterwarnings

import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts

# import ipywidgets as ipw
# from IPython.display import display

hv.extension("bokeh")
pn.extension()


def wave_length(T, h):
    d = h

    # based on waveNumber_Fenton(T,d) from Jaime in computerlab
    g = 9.81
    omega = 2 * np.pi / T
    k0 = omega * omega / g
    alpha = k0 * d
    beta = alpha * (np.tanh(alpha)) ** -0.5
    k = (
        (alpha + beta**2 * np.cosh(beta) ** -2)
        / (np.tanh(beta) + beta * np.cosh(beta) ** -2)
        / d
    )

    L = 2 * np.pi / k

    return L


def depth_xrange(slope, h0):
    m = 1 / slope
    step = 1

    steps = int(h0 * m * 1 / step)
    x_range = np.linspace(0, h0 * m, num=steps, endpoint=False)
    x = np.array(x_range, dtype=float)

    zbed = -(h0 - x * slope)
    h = -zbed
    x_rev = x - np.max(x) - step
    return h, x_rev, x_range, zbed


def check_wave_transformation(input_values, student_func):
    H0, T, h0, slope, rho, angle0, gamma = input_values

    def oblique_wave(H0, T, h0, slope, rho, angle0, gamma):
        h, x_rev, x_range, zbed = depth_xrange(slope, h0)

        L = np.array([wave_length(T, h_val) for h_val in h], dtype=float)
        c = L / T
        k = 2 * np.pi / L
        n = 0.5 + (k * h / np.sinh(2 * k * h))
        cg = n * c
        Ksh = np.sqrt(cg[0] / cg)

        snell_constant = np.sin(np.deg2rad(angle0)) / c[0]
        theta_radians = np.arcsin(snell_constant * c)
        theta = np.rad2deg(theta_radians)
        Kr = (np.cos(theta_radians[0]) / np.cos(theta_radians)) ** 0.5

        H = H0 * Ksh * Kr
        Hbreaking = gamma * h
        H[H > Hbreaking] = Hbreaking[H > Hbreaking]
        g = 9.81
        E = 1 / 8 * rho * g * H**2

        return L, c, n, cg, theta, Ksh, Kr, H, E

    correct_func = [oblique_wave]

    def check_answer(input_values, student_func, correct_func):
        H0, T, h0, slope, rho, angle0, gamma = input_values.values()
        h, x_rev, x_range, zbed = depth_xrange(slope, h0)

        answer_list = ["h", "L", "c", "n", "cg", "θ", "Ksh", "Kr", "H", "E"]

        title_list = [
            "h",  # not used
            "wave length L",
            "phase celerity c",
            "n",
            "group velocity cg",
            "wave angle θ",
            "shoaling coefficient Ksh",
            "refraction coefficient Kr",
            "wave height H",
            "wave energy E",
        ]
        unit_list = [
            "[m]",
            "[m]",
            "[m/s]",
            "[-]",
            "[m/s]",
            "[degrees]",
            "[-]",
            "[-]",
            "[m]",
            "[J/m2]",
        ]
        legendpos_list = [
            "bottom_right",
            "bottom_left",
            "bottom_left",
            "top_left",
            "bottom_left",
            "bottom_left",
            "top_left",
            "bottom_left",
            "bottom_left",
            "bottom_left",
        ]

        # student_func and correct_func shoud be provided as a list element, so that multiple functions
        # can be checked at once. This method remains applicable for one function
        student_answers = []
        correct_answers = []

        # loop through the functions and store the student and correct answers
        # student_output = student_func[0](H0, T, h0, slope, rho, x_range, angle0, gamma)
        # correct_output = correct_func[0](H0, T, h0, slope, rho, x_range, angle0, gamma)

        student_output = student_func[0](H0, T, h0, slope, rho, angle0, gamma)
        correct_output = correct_func[0](H0, T, h0, slope, rho, angle0, gamma)

        for i in range(len(student_output)):
            student_answers.append(
                student_output[i]
                if student_output[i] is not None
                else np.zeros(len(x_range))
            )
            correct_answers.append(correct_output[i])

        # Provide feedback if student answers are correct or not
        return_text = "Your function is: " + "\n"
        bed_lvl = hv.Curve(
            (x_rev, zbed),
            label="Bed (1:" + str(round(1 / slope)) + ")",
        )
        bed_lvl.opts(color="black", show_legend=True, padding=((0, 0.05), 0.1))
        water_lvl = hv.Curve(([np.min(x_rev), 0], [0, 0]), label="MSL")
        water_lvl.opts(color="gray", show_legend=True, padding=((0, 0.05), 0.1))
        figure_h = hv.Overlay(bed_lvl * water_lvl)
        figure_h.opts(
            title="cross-shore profile",
            shared_axes=True,
            aspect=1.7,
            responsive=True,
            legend_position=f"{legendpos_list[0]}",
            ylabel="z = -h [m]",
            xlabel="cross-shore location [m]",
        )
        figures = hv.Empty() + figure_h + hv.Empty()
        for i in range(len(student_answers)):
            if np.allclose(student_answers[i], correct_answers[i], atol=1e-5):
                return_text += f"Values for {answer_list[i+1]} are correct.\n"
            else:
                return_text += f"Values for {answer_list[i+1]} are incorrect.\n"

            # Plot the answers with holoviews and visualize the differences
            correct_plot = hv.Curve(
                (x_rev, correct_answers[i]),
                label="Correct",
            ).opts(
                xlabel="cross-shore location [m]",
                ylabel=f"{answer_list[i+1]} {unit_list[i+1]}",
                padding=((0, 0.05), 0.1),
            )
            student_plot = hv.Curve(
                (x_rev, student_answers[i]),
                label="Your code",
            ).opts(
                xlabel="cross-shore location [m]",
                ylabel=f"{answer_list[i+1]} {unit_list[i+1]}",
                line_dash="dashed",
            )
            figures += hv.Overlay(correct_plot * student_plot).opts(
                title=f"{title_list[i+1]}",
                shared_axes=True,
                aspect=1.7,
                responsive=True,
                legend_position=f"{legendpos_list[i+1]}",
            )
        # The following commented lines were used to create the files including the correct output of the code, which are used later on.
        # These files are added to the database on Teams
        # ## Specify the file name
        # file_name = f"{answer_list[i+1]}_values.txt"

        # ## Write the list to the file
        # with open(file_name, "w") as file:
        #     for value in correct_answers[i]:
        #         file.write(f"{value} ")

        figure = hv.Layout(figures).cols(3).opts(shared_axes=False)
        filterwarnings("ignore", category=FutureWarning)
        return return_text, figure

    filterwarnings("ignore", category=RuntimeWarning)
    return check_answer(input_values, student_func, correct_func)


def pop_wave_transformation():
    # Define widgets
    H0_1 = pn.widgets.FloatInput(name="H₁ [m]", value=1.5, start=0, end=50, step=0.5)
    T1 = pn.widgets.FloatInput(name="T₁ [s]", value=6, start=0.5, end=50, step=0.5)
    angle1 = pn.widgets.FloatInput(name="θ₁ [°]", value=15, start=0, end=90, step=1)

    H0_2 = pn.widgets.FloatInput(name="H₂ [m]", value=2.0, start=0, end=50, step=0.5)
    T2 = pn.widgets.FloatInput(name="T₂ [s]", value=7, start=0.5, end=500, step=0.5)
    angle2 = pn.widgets.FloatInput(name="θ₂ [°]", value=30, start=0, end=90, step=1)

    h0 = pn.widgets.FloatInput(
        name="Depth at boundary [m]", value=20, start=1, end=100, step=1
    )
    slope_in = pn.widgets.FloatInput(
        name="Slope: 1:...", value=100, start=1, end=500, step=1
    )
    gamma = pn.widgets.FloatInput(
        name="Breaking parameter [-]", value=0.8, start=0.4, end=1.5, step=0.1
    )
    rho = pn.widgets.FloatInput(
        name="Water density [kg/m³]", value=1025, start=1000, end=1050, step=1
    )

    # Layout the widgets
    wave1_controls = pn.Column(
        pn.pane.Markdown("### Offshore conditions wave 1"), H0_1, T1, angle1
    )
    wave2_controls = pn.Column(
        pn.pane.Markdown("### Offshore conditions wave 2"), H0_2, T2, angle2
    )
    bathymetry_controls = pn.Column(pn.pane.Markdown("### Profile"), h0, slope_in)
    input_controls = pn.Column(pn.pane.Markdown("### Other input"), gamma, rho)

    widgets = pn.Row(
        wave1_controls,
        wave2_controls,
        bathymetry_controls,
        input_controls,
        align="start",
    )

    # Function to calculate wave properties
    def calc_wave_properties(h, T, H0, angle, gamma, rho):
        g = 9.81
        if T <= 0 or h.size == 0:
            raise ValueError("Wave period T and depth h must be valid.")
        L = wave_length(T, h)
        c = L / T
        k = 2 * np.pi / L
        n = 0.5 + (k * h / np.sinh(2 * k * h))
        cg = n * c
        Ksh = np.sqrt(cg[0] / cg)
        theta_r = np.arcsin(np.sin(np.radians(angle)) / c[0] * c)
        theta_d = np.rad2deg(theta_r)
        Kr = np.sqrt(np.cos(theta_r[0]) / np.cos(theta_r))
        H = H0 * Ksh * Kr
        H = np.minimum(H, gamma * h)
        E = 0.125 * rho * g * H**2
        return {
            "L": L,
            "c": c,
            "n": n,
            "cg": cg,
            "θ": theta_d,
            "Ksh": Ksh,
            "Kr": Kr,
            "H": H,
            "E": E,
        }

    # Interactive plot function
    @pn.depends(
        H0_1.param.value,
        T1.param.value,
        angle1.param.value,
        H0_2.param.value,
        T2.param.value,
        angle2.param.value,
        h0.param.value,
        slope_in.param.value,
        gamma.param.value,
        rho.param.value,
    )
    def interactiveplot_wave_transformation(
        H0_1, T1, angle1, H0_2, T2, angle2, h0, slope_in, gamma, rho
    ):
        slope = 1.0 / slope_in  # Bed slope [-]

        h, x_rev, x_range, zbed = depth_xrange(slope, h0)

        wave1 = calc_wave_properties(h, T1, H0_1, angle1, gamma, rho)
        wave2 = calc_wave_properties(h, T2, H0_2, angle2, gamma, rho)

        ylabels = {
            "L": "Wavelength \n (L) [m]",
            "c": "Wave celerity \n (c) [m/s]",
            "n": "Parameter \n (n) [-]",
            "cg": "Group celerity \n (cg) [m/s]",
            "θ": "Theta \n (θ) [degrees]",
            "Ksh": "Shoaling coeff. \n (Ksh) [-]",
            "Kr": "Refraction coeff. \n (Kr) [-]",
            "H": "Wave height \n (H) [m]",
            "E": "Wave energy \n (E) [J/m²]",
        }

        legend_pos = {
            "L": "bottom_left",
            "c": "bottom_left",
            "n": "top_left",
            "cg": "bottom_left",
            "θ": "bottom_left",
            "Ksh": "top_left",
            "Kr": "bottom_left",
            "H": "bottom_left",
            "E": "bottom_left",
        }

        wave_plots = [
            hv.Overlay(
                hv.Curve((x_rev, wave1[key]), label=f"{key}₁").opts(
                    ylabel=ylabels[key],
                    xlabel="cross-shore location [m]",
                    responsive=True,
                    padding=((0, 0.05), 0.1),
                )
                * hv.Curve((x_rev, wave2[key]), label=f"{key}₂").opts(
                    ylabel=ylabels[key],
                    xlabel="cross-shore location [m]",
                    responsive=True,
                    padding=((0, 0.05), 0.1),
                )
            ).opts(legend_position=legend_pos[key], legend_cols=2)
            for key in wave1
        ]

        return (
            hv.Layout(wave_plots)
            .cols(3)
            .opts(shared_axes=False, legend_position="left")
        )

    app = pn.Column(widgets, interactiveplot_wave_transformation)

    return app


def check_radiation_stresses(
    input_values, L, c, n, cg, angle, Ksh, Kr, H, E, student_func
):
    H0, T, h0, slope, rho, angle0, gamma, cf = input_values.values()

    def correct_radiation_stresses(input_values, L, c, n, cg, angle, Ksh, Kr, H, E):
        H0, T, h0, slope, rho, angle0, gamma, cf = input_values.values()

        g = 9.81

        h, x_rev, x_range, zbed = depth_xrange(slope, h0)

        k = 2 * np.pi / L  # The wave number
        snell_constant = np.sin(np.deg2rad(angle0)) / c[0]
        theta_radians = np.arcsin(snell_constant * c)

        Hbreaking = gamma * h  # The wave-breaking height

        Sxx = (n - 0.5 + n * np.cos(theta_radians) ** 2) * E
        Syx = n * np.cos(theta_radians) * np.sin(theta_radians) * E

        eta = np.zeros(Sxx.shape, dtype=float)
        for i in range(len(eta) - 1):  # key here is that setup[0] = 0
            eta[i + 1] = eta[i] - (Sxx[i + 1] - Sxx[i]) / (rho * g * h[i])

        Fx = np.zeros(eta.shape, dtype=float)
        for i in range(len(Fx) - 1):
            Fx[i] = -(Sxx[i + 1] - Sxx[i]) / (x_rev[i + 1] - x_rev[i])

        Fy = np.zeros(Syx.shape)
        for i in range(len(Fy) - 1):
            Fy[i] = -(Syx[i + 1] - Syx[i]) / (x_rev[i + 1] - x_rev[i])

        dh_dx = np.zeros(len(h))  # dh/dx
        for i in range(len(dh_dx) - 1):
            dh_dx[i] = (h[i + 1] - h[i]) / (x_rev[i + 1] - x_rev[i])

        # formula 5.82 on page 226 of the book
        V = (
            -5
            / 16
            * np.pi
            * gamma
            / cf
            * g
            * np.sin(theta_radians[0])
            / c[0]
            * h
            * dh_dx
        )
        V[H != Hbreaking] = 0  # function valid where waves are breaking

        omega = 2 * np.pi / T
        u0 = 0.5 * omega * H / np.sinh(k * h)

        return Sxx, Fx, eta, Syx, Fy, V, u0

    correct_func = [correct_radiation_stresses]

    def check_answer(input_values, student_func, correct_func):
        H0, T, h0, slope, rho, angle0, gamma, cf = input_values.values()

        h, x_rev, x_range, zbed = depth_xrange(slope, h0)

        # student_func and correct_func shoud be provided as a list element, so that multiple functions
        # can be checked at once. This method remains applicable for one function
        student_answers = []
        correct_answers = []

        answer_list = ["h", "Sxx", "Fx", "η", "Syx", "Fy", "V", "u0"]

        title_list = [
            "h",  # not used
            "radiation normal stress Sxx",
            "wave force Fx",
            "set-down and set-up η",
            "radiation shear stress Syx",
            "wave force Fy",
            "longshore current velocity V",
            "near-bed orbital velocity ampl. u0",
        ]

        unit_list = [
            "[m]",
            "[N/m]",
            "[N/m²]",
            "[m]",
            "[N/m]",
            "[N/m²]",
            "[m/s]",
            "[m/s]",
        ]

        legendpos_list = [
            "bottom_right",
            "top_left",
            "top_left",
            "top_left",
            "bottom_left",
            "top_left",
            "top_left",
            "top_left",
        ]

        # loop through the functions and store the student and correct answers
        student_output = student_func[0](
            input_values, L, c, n, cg, angle, Ksh, Kr, H, E
        )
        correct_output = correct_func[0](
            input_values, L, c, n, cg, angle, Ksh, Kr, H, E
        )

        for i in range(len(student_output)):
            student_answers.append(
                student_output[i]
                if student_output[i] is not None
                else np.zeros(len(x_range))
            )
            correct_answers.append(correct_output[i])

        # Provide feedback if student answers are correct or not
        return_text = "Your function is: " + "\n"
        bed_lvl = hv.Curve(
            (x_rev, zbed), label="Bed (1:" + str(round(1 / slope, 2)) + ")"
        )
        bed_lvl.opts(color="black", show_legend=True)
        water_lvl = hv.Curve(([np.min(x_rev), 0], [0, 0]), label="MSL")
        water_lvl.opts(color="gray", show_legend=True)
        figure_h = hv.Overlay(bed_lvl * water_lvl)
        figure_h.opts(
            legend_position=legendpos_list[0],
            title="cross-shore profile",
            shared_axes=True,
            aspect=1.7,
            responsive=True,
            padding=((0, 0.05), 0.1),
            ylabel="z = -h [m]",
            xlabel="cross-shore location [m]",
        )

        cor_u0 = hv.Curve((x_rev, correct_answers[-1]), label="Correct").opts(
            xlabel="cross-shore location [m]",
            ylabel=f"{answer_list[-1]} {unit_list[-1]}",
        )
        stud_u0 = hv.Curve((x_rev, student_answers[-1]), label="Your code").opts(
            xlabel="cross-shore location [m]",
            ylabel=f"{answer_list[-1]} {unit_list[-1]}",
            line_dash="dashed",
        )
        figure_u0 = hv.Overlay(cor_u0 * stud_u0).opts(
            legend_position=legendpos_list[-1],
            title=f"{title_list[i+1]}",
            shared_axes=True,
            aspect=1.7,
            responsive=True,
            padding=((0, 0.05), 0.1),
        )

        figures = figure_h + figure_u0 + hv.Empty()
        for i in range(len(student_answers) - 1):
            # Check if each student answer matches the correct answer
            if np.allclose(student_answers[i][:-2], correct_answers[i][:-2], atol=1e-5):
                return_text += f"Values for {answer_list[i+1]} are correct.\n"
            else:
                return_text += f"Values for {answer_list[i+1]} are incorrect.\n"

            # Plot the answers with holoviews and visualize the differences
            correct_plot = hv.Curve((x_rev, correct_answers[i]), label="Correct").opts(
                xlabel="cross-shore location [m]",
                ylabel=f"{answer_list[i+1]} {unit_list[i+1]}",
                padding=((0, 0.05), 0.1),
            )
            student_plot = hv.Curve(
                (x_rev, student_answers[i]), label="Your code"
            ).opts(
                xlabel="cross-shore location [m]",
                ylabel=f"{answer_list[i+1]} {unit_list[i+1]}",
                line_dash="dashed",
            )
            figures += hv.Overlay(correct_plot * student_plot).opts(
                legend_position=legendpos_list[i + 1],
                title=f"{title_list[i+1]}",
                shared_axes=True,
                aspect=1.7,
                responsive=True,
            )

        figure = hv.Layout(figures).cols(3).opts(shared_axes=False)
        filterwarnings("ignore", category=FutureWarning)
        return return_text, figure

    filterwarnings("ignore", category=RuntimeWarning)
    return check_answer(input_values, student_func, correct_func)


def pop_radiation_stresses():
    # Define widgets
    H0_1 = pn.widgets.FloatInput(name="H₁ [m]", value=1.5, start=0, end=50, step=0.5)
    T1 = pn.widgets.FloatInput(name="T₁ [s]", value=6, start=0.5, end=50, step=0.5)
    angle1 = pn.widgets.FloatInput(name="θ₁ [°]", value=15, start=0, end=90, step=1)

    H0_2 = pn.widgets.FloatInput(name="H₂ [m]", value=2.0, start=0, end=50, step=0.5)
    T2 = pn.widgets.FloatInput(name="T₂ [s]", value=5, start=0.5, end=500, step=0.5)
    angle2 = pn.widgets.FloatInput(name="θ₂ [°]", value=30, start=0, end=90, step=1)

    h0 = pn.widgets.FloatInput(
        name="Depth at boundary [m]", value=30, start=1, end=100, step=1
    )
    slope_in = pn.widgets.FloatInput(
        name="Slope: 1:...", value=100, start=1, end=500, step=1
    )

    gamma = pn.widgets.FloatInput(
        name="Breaking parameter [-]", value=0.8, start=0.4, end=1.5, step=0.1
    )
    rho = pn.widgets.FloatInput(
        name="Water density [kg/m³]", value=1025, start=1000, end=1050, step=1
    )

    cf = pn.widgets.FloatInput(
        name="Friction coefficient [-]", value=0.01, start=0, end=1, step=0.005
    )

    # Layout the widgets
    wave1_controls = pn.Column(
        pn.pane.Markdown("### Offshore conditions wave 1"), H0_1, T1, angle1
    )
    wave2_controls = pn.Column(
        pn.pane.Markdown("### Offshore conditions wave 2"), H0_2, T2, angle2
    )
    bathymetry_controls = pn.Column(pn.pane.Markdown("### Profile"), h0, slope_in)
    input_controls = pn.Column(pn.pane.Markdown("### Other input"), gamma, rho, cf)

    widgets = pn.Row(
        wave1_controls,
        wave2_controls,
        bathymetry_controls,
        input_controls,
        align="start",
    )

    # Function to calculate wave properties
    def calc_radiation(h, T, H0, angle, x, gamma, rho, cf):
        g = 9.81
        if T <= 0 or h.size == 0:
            raise ValueError("Wave period T and depth h must be valid.")
        L = np.array([wave_length(T, h) for h in h])
        c = L / T
        k = 2 * np.pi / L
        n = 0.5 + (k * h / np.sinh(2 * k * h))
        cg = n * c
        Ksh = np.sqrt(cg[0] / cg)
        omega = 2 * np.pi / T
        snell_constant = np.clip(np.sin(np.deg2rad(angle)) / c[0], -1, 1)
        theta_radians = np.arcsin(snell_constant * c)
        Kr = np.sqrt(np.cos(theta_radians[0]) / np.cos(theta_radians))
        H = H0 * Ksh * Kr
        Hbreaking = gamma * h  # The wave-breaking height
        H[H > Hbreaking] = Hbreaking[H > Hbreaking]
        u0 = 0.5 * omega * H / np.sinh(k * h)
        E = 0.125 * rho * g * H**2
        Sxx = (n - 0.5 + n * np.cos(theta_radians) ** 2) * E
        Syx = n * np.cos(theta_radians) * np.sin(theta_radians) * E
        eta = np.zeros(Sxx.shape, dtype=float)
        for i in range(len(eta) - 1):  # key here is that setup[0] = 0
            eta[i + 1] = eta[i] - (Sxx[i + 1] - Sxx[i]) / (rho * g * h[i])
        Fx = np.zeros(eta.shape, dtype=float)
        for i in range(len(Fx) - 1):
            Fx[i] = -(Sxx[i + 1] - Sxx[i]) / (x[i + 1] - x[i])
        Fy = np.zeros(Syx.shape)
        for i in range(len(Fy) - 1):
            Fy[i] = -(Syx[i + 1] - Syx[i]) / (x[i + 1] - x[i])
        dh_dx = np.zeros(len(h))  # dh/dx
        for i in range(len(dh_dx) - 1):
            dh_dx[i] = (h[i + 1] - h[i]) / (x[i + 1] - x[i])
        V = (
            -5
            / 16
            * np.pi
            * gamma
            / cf
            * g
            * np.sin(theta_radians[0])
            / c[0]
            * h
            * dh_dx
        )
        V[H != Hbreaking] = 0

        return {
            "u0": u0,
            "Sxx": Sxx,
            "Fx": Fx,
            "η": eta,
            "Syx": Syx,
            "Fy": Fy,
            "V": V,
        }

    # Interactive plot function
    @pn.depends(
        H0_1.param.value,
        T1.param.value,
        angle1.param.value,
        H0_2.param.value,
        T2.param.value,
        angle2.param.value,
        h0.param.value,
        slope_in.param.value,
        gamma.param.value,
        rho.param.value,
        cf.param.value,
    )
    def interactiveplot_radiation_stress(
        H0_1, T1, angle1, H0_2, T2, angle2, h0, slope_in, gamma, rho, cf
    ):
        slope = 1.0 / slope_in  # Bed slope [-]

        h, x_rev, x_range, zbed = depth_xrange(slope, h0)

        # Initial bed and still water plots
        bed_lvl = hv.Curve((x_rev, zbed), label="Bed level").opts(
            color="black",
            padding=((0, 0.05), 0.1),
        )
        water_lvl = hv.Curve(([np.min(x_rev), 0], [0, 0]), label="MSL]").opts(
            color="gray",
            padding=((0, 0.05), 0.1),
        )
        plot0 = hv.Overlay([bed_lvl, water_lvl]).opts(
            responsive=True,
            legend_position="bottom_right",
            ylabel="z = -h [m]",
        )

        wave1 = calc_radiation(h, T1, H0_1, angle1, x_range, gamma, rho, cf)
        wave2 = calc_radiation(h, T2, H0_2, angle2, x_range, gamma, rho, cf)

        ylabels = {
            "u0": "Near-bed orbital \n velocity (u0) [m/s]",
            "Sxx": "Radiation normal \n stress (Sxx) [N/m]",
            "Fx": "Cross-shore wave \n force (Fx) [N/m²]",
            "η": "Set-down and \n set-up (η) [m]",
            "Syx": "Radiation shear \n stress (Syx) [N/m]",
            "Fy": "Along-shore wave \n force (Fy) [N/m²]",
            "V": "Longshore velocity \n (V) [m/s]",
        }

        legend_pos = {
            "u0": "top_left",
            "Sxx": "top_left",
            "Fx": "top_left",
            "η": "top_left",
            "Syx": "bottom_left",
            "Fy": "top_left",
            "V": "top_left",
        }

        wave_plots = [
            hv.Overlay(
                hv.Curve((x_rev, wave1[key]), label=f"{key}₁").opts(
                    ylabel=ylabels[key],
                    xlabel="Cross-shore location (x) [m]",
                    responsive=True,
                    padding=((0, 0.05), 0.1),
                )
                * hv.Curve((x_rev, wave2[key]), label=f"{key}₂").opts(
                    ylabel=ylabels[key],
                    xlabel="Cross-shore location (x) [m]",
                    responsive=True,
                    padding=((0, 0.05), 0.1),
                )
            ).opts(legend_position=legend_pos[key], legend_cols=2)
            for key in wave1
        ]

        return (
            hv.Layout([plot0 + wave_plots[0] + hv.Empty()] + wave_plots[1:])
            .cols(3)
            .opts(shared_axes=False, legend_position="left")
        )

    app = pn.Column(widgets, interactiveplot_radiation_stress)

    return app
