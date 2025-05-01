import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts
from warnings import filterwarnings

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


def group_stats(k1, k2, w1, w2):
    Delta_k = k2 - k1
    Delta_w = w2 - w1
    L = 2 * np.pi / abs(Delta_k)
    T = 2 * np.pi / abs(Delta_w)
    cg = Delta_w / Delta_k
    return L, T, cg


# Ranges for horizontal axes
def ranges(h, T1, T2, ng):
    k1 = 2 * np.pi / wave_length(T1, h)
    k2 = 2 * np.pi / wave_length(T2, h)
    w1 = 2 * np.pi / T1
    w2 = 2 * np.pi / T2
    L, T, cg = group_stats(k1, k2, w1, w2)
    xmax = ng * L
    tmax = ng * T
    x = np.linspace(0, xmax, 1000)  # x-range for spatial plot
    t = np.linspace(0, tmax, 1000)  # t-range for time plot
    return x, t


def time_eta(a, T1, T2, tp, xp, h, t, x, rho):
    L1 = wave_length(T1, h)
    L2 = wave_length(T2, h)

    w1 = 2 * np.pi / T1
    w2 = 2 * np.pi / T2
    k1 = 2 * np.pi / L1
    k2 = 2 * np.pi / L2
    Delta_w = w2 - w1
    Delta_k = k2 - k1
    eta_t = np.abs(2 * a * np.cos(0.5 * Delta_w * t - 0.5 * Delta_k * xp))
    return eta_t


def space_eta(a, T1, T2, tp, xp, h, t, x, rho):
    L1 = wave_length(T1, h)
    L2 = wave_length(T2, h)

    # eta = 2a cos( Delta omega/2  t- Delta k/2 x)
    w1 = 2 * np.pi / T1
    w2 = 2 * np.pi / T2
    k1 = 2 * np.pi / L1
    k2 = 2 * np.pi / L2
    Delta_w = w2 - w1
    Delta_k = k2 - k1
    eta_x = np.abs(2 * a * np.cos(0.5 * Delta_w * tp - 0.5 * Delta_k * x))
    return eta_x


def check_envelope_E(input_values, student_func):
    a, T1, T2, tp, xp, t, x, h, n_groups, rho, g = input_values.values()

    def wave_E_t(eta_t, T1, T2, h, rho, g):
        E = 1 / 2 * rho * g * eta_t**2  # The wave energy
        return E

    def wave_E_x(eta_x, T1, T2, h, rho, g):
        E = 1 / 2 * rho * g * eta_x**2  # The wave energy
        return E

    def wave_Sxx_t(eta_t, T1, T2, h, rho, g):
        E = wave_E_t(eta_t, T1, T2, h, rho, g)
        L1 = wave_length(T1, h)
        L2 = wave_length(T2, h)

        w1 = 2 * np.pi / T1
        w2 = 2 * np.pi / T2
        k1 = 2 * np.pi / L1
        k2 = 2 * np.pi / L2
        Delta_w = w2 - w1
        Delta_k = k2 - k1

        c_average = (w1 + w2) / (k1 + k2)
        cg = Delta_w / Delta_k
        n = cg / c_average

        Sxx = (2 * n - 0.5) * E
        return Sxx

    def wave_Sxx_x(eta_x, T1, T2, h, rho, g):
        E = wave_E_x(eta_x, T1, T2, h, rho, g)
        L1 = wave_length(T1, h)
        L2 = wave_length(T2, h)

        w1 = 2 * np.pi / T1
        w2 = 2 * np.pi / T2
        k1 = 2 * np.pi / L1
        k2 = 2 * np.pi / L2
        Delta_w = w2 - w1
        Delta_k = k2 - k1

        c_average = (w1 + w2) / (k1 + k2)
        cg = Delta_w / Delta_k
        n = cg / c_average

        Sxx = (2 * n - 0.5) * E
        return Sxx

    E_S_func = [wave_E_t, wave_E_x, wave_Sxx_t, wave_Sxx_x]

    def check_answer(input_values, student_func, E_S_func):
        a, T1, T2, tp, xp, t, x, h, n_groups, rho, g = input_values.values()
        answer_list = [
            "E_t = E(x = xₚ, t)",
            "E_x = E(x, t = tₚ)",
            "Sxx_t = Sₓₓ(x = xₚ, t)",
            "Sxx_x = Sₓₓ(x, t = tₚ)",
        ]

        # # Additional calculations to show the components constructing eta
        L1 = wave_length(T1, h)
        L2 = wave_length(T2, h)

        x, t = ranges(h, T1, T2, n_groups)

        eta_t = time_eta(a, T1, T2, tp, xp, h, t, x, rho)
        eta_x = space_eta(a, T1, T2, tp, xp, h, t, x, rho)
        eta_tx = [eta_t, eta_x]

        eta1_T = a * np.sin(2 * np.pi / T1 * t - 2 * np.pi / L1 * xp)
        eta2_T = a * np.sin(2 * np.pi / T2 * t - 2 * np.pi / L2 * xp)
        eta_T = eta1_T + eta2_T

        eta1_X = a * np.sin(2 * np.pi / T1 * tp - 2 * np.pi / L1 * x)
        eta2_X = a * np.sin(2 * np.pi / T2 * tp - 2 * np.pi / L2 * x)
        eta_X = eta1_X + eta2_X

        # student_func and correct_func shoud be provided as a list element, so that multiple functions
        # can be checked at once. This method remains applicable for one function
        student_answers = []
        E_S_answers = []

        # loop through the functions and store the student and correct answers
        for i in range(4):
            # the even numbers, i%2=0, are the time based calculations and odd numbers, i%2=1, are space based
            student_answers.append(
                student_func(a, T1, T2, tp, xp, t, x, h, rho, g)[i]
                if student_func(a, T1, T2, tp, xp, t, x, h, rho, g)[i] is not None
                else np.zeros(len(x))
            )
            E_S_answers.append(E_S_func[i](eta_tx[i % 2], T1, T2, h, rho, g))

        # Provide feedback if student answers are correct or not
        return_text = "Your function is: " + "\n"

        student_plot = hv.Curve(
            (t, student_answers[0]), label=f"Student {answer_list[0]}"
        )
        E_S_plot = hv.Curve((t, E_S_answers[0]), label=f"Correct {answer_list[0]}")
        E_S_plot.opts(
            xlabel="time [s]", ylabel="E [N/m]", line_dash="dashed", color="green"
        )

        figures = hv.Overlay(student_plot * E_S_plot).opts(
            title=answer_list[0],
            aspect=2,
            responsive=True,
            shared_axes=True,
            legend_position="right",
        )

        for i in range(len(student_answers)):
            # Check if each student answer matches the correct answer
            if np.allclose(student_answers[i], E_S_answers[i], atol=1e-5):
                return_text += f"Function {i + 1} is correct.\n"
            else:
                return_text += f"Function {i + 1} is incorrect.\n"

            # Plot the answers with holoviews and visualize the differences
            if i >= 1:
                # the odd numbers are the space based calculations
                if i % 2 == 1:
                    E_S_plot = hv.Curve(
                        (x, E_S_answers[i]), label=f"Correct {answer_list[i]}"
                    )
                    E_S_plot.opts(
                        xlabel="space [m]",
                        ylabel="E [N/m]",
                        line_dash="dashed",
                        color="green",
                    )
                    if i == 3:
                        E_S_plot.opts(ylabel="Sₓₓ [N/m]")
                    student_plot = hv.Curve(
                        (x, student_answers[i]), label=f"Student {answer_list[i]}"
                    )

                    figures += hv.Overlay(student_plot * E_S_plot).opts(
                        title=answer_list[i],
                        aspect=2,
                        responsive=True,
                        shared_axes=True,
                        legend_position="right",
                    )
                # the even numbers are the time based calculations
                else:
                    E_S_plot = hv.Curve(
                        (t, E_S_answers[i]), label=f"Correct {answer_list[i]}"
                    )
                    E_S_plot.opts(
                        xlabel="time [s]",
                        ylabel="Sₓₓ [N/m]",
                        line_dash="dashed",
                        color="green",
                    )
                    student_plot = hv.Curve(
                        (t, student_answers[i]), label=f"Student {answer_list[i]}"
                    )

                    figures += hv.Overlay(student_plot * E_S_plot).opts(
                        title=answer_list[i],
                        aspect=2,
                        responsive=True,
                        shared_axes=True,
                        legend_position="right",
                    )

        component1 = hv.Curve((t, eta_T), label="η")
        t_envelope = hv.Curve((t, eta_tx[0]), label="Upper envelope")
        t_neg_envelope = hv.Curve((t, -1 * eta_tx[0]), label="Lower envelope")
        plot1 = hv.Overlay(component1 * t_envelope * t_neg_envelope).opts(
            xlabel="t [s]",
            ylabel="η [m]",
            aspect=2,
            responsive=True,
            legend_position="right",
        )

        component2 = hv.Curve((x, eta_X), label="η")
        x_envelope = hv.Curve((x, eta_tx[1]), label="Upper envelope")
        x_neg_envelope = hv.Curve((x, -1 * eta_tx[1]), label="Lower envelope")
        plot2 = hv.Overlay(component2 * x_envelope * x_neg_envelope).opts(
            xlabel="x [m]",
            ylabel="η [m]",
            aspect=2,
            responsive=True,
            legend_position="right",
        )

        figure = hv.Layout(plot1 + plot2 + figures).cols(2).opts(shared_axes=False)
        filterwarnings("ignore", category=FutureWarning)
        return return_text, figure

    return check_answer(input_values, student_func, E_S_func)


def check_boundwave_eta(input_values, student_func):
    a, T1, T2, tp, xp, t, x, h, n_groups, rho, g = input_values.values()

    def correct_etab_t(a, T1, T2, tp, xp, h, t, x, rho):
        g = 9.81  # m/s^2

        L1 = wave_length(T1, h)
        L2 = wave_length(T2, h)

        w1 = 2 * np.pi / T1
        w2 = 2 * np.pi / T2
        k1 = 2 * np.pi / L1
        k2 = 2 * np.pi / L2
        Delta_w = w2 - w1
        Delta_k = k2 - k1

        c_average = (w1 + w2) / (k1 + k2)
        cg = Delta_w / Delta_k
        n = cg / c_average

        eta_b = -g * a**2 * (2 * n - 0.5) / (g * h - cg**2)
        eta = eta_b * np.cos(Delta_w * t - Delta_k * xp)

        return eta

    def correct_etab_x(a, T1, T2, tp, xp, h, t, x, rho):
        g = 9.81  # m/s^2
        L1 = wave_length(T1, h)
        L2 = wave_length(T2, h)

        w1 = 2 * np.pi / T1
        w2 = 2 * np.pi / T2
        k1 = 2 * np.pi / L1
        k2 = 2 * np.pi / L2

        Delta_w = w2 - w1
        Delta_k = k2 - k1

        c_average = (w1 + w2) / (k1 + k2)
        cg = Delta_w / Delta_k
        n = cg / c_average

        eta_b = -g * a**2 * (2 * n - 0.5) / (g * h - cg**2)
        eta = eta_b * np.cos(Delta_w * tp - Delta_k * x)
        return eta

    correct_func = [correct_etab_t, correct_etab_x]

    def check_answer(input_values, student_func, correct_func):
        a, T1, T2, tp, xp, t, x, h, n_groups, rho, g = input_values.values()
        answer_list = ["etab_t = ηb(x = xₚ, t)", "etab_x = ηb(x, t = tₚ)"]

        # Additional calculations to show the components constructing eta
        L1 = wave_length(T1, h)
        L2 = wave_length(T2, h)

        x, t = ranges(h, T1, T2, n_groups)

        w1 = 2 * np.pi / T1
        w2 = 2 * np.pi / T2
        k1 = 2 * np.pi / L1
        k2 = 2 * np.pi / L2

        a_T = time_eta(a, T1, T2, tp, xp, h, t, x, rho)
        a_X = space_eta(a, T1, T2, tp, xp, h, t, x, rho)

        eta1_T = a * np.sin(2 * np.pi / T1 * t - 2 * np.pi / L1 * xp)
        eta2_T = a * np.sin(2 * np.pi / T2 * t - 2 * np.pi / L2 * xp)
        eta_T = eta1_T + eta2_T

        eta1_X = a * np.sin(2 * np.pi / T1 * tp - 2 * np.pi / L1 * x)
        eta2_X = a * np.sin(2 * np.pi / T2 * tp - 2 * np.pi / L2 * x)
        eta_X = eta1_X + eta2_X

        # student_func and correct_func shoud be provided as a list element, so that multiple functions
        # can be checked at once. This method remains applicable for one function
        student_answers = []
        correct_answers = []

        # loop through the functions and store the student and correct answers
        for i in range(2):
            student_answers.append(
                student_func(a, T1, T2, tp, xp, t, x, h, rho, g)[i]
                if student_func(a, T1, T2, tp, xp, t, x, h, rho, g)[i] is not None
                else np.zeros(len(x))
            )
            correct_answers.append(correct_func[i](a, T1, T2, tp, xp, h, t, x, rho))

        # Provide feedback if student answers are correct or not
        return_text = "Your function is: " + "\n"

        student_plot = hv.Curve(
            (t, student_answers[0]), label=f"Student {answer_list[0]}"
        ).opts(xlabel="t [s]", ylabel="η [m]")
        correct_plot = hv.Curve(
            (t, correct_answers[0]), label=f"Correct {answer_list[0]}"
        ).opts(line_dash="dashed", color="green")
        figures = hv.Overlay(student_plot * correct_plot).opts(
            aspect=2,
            responsive=True,
            shared_axes=True,
            legend_position="right",
            title=answer_list[0],
        )

        component_plot = hv.Curve((t, eta_T), label="η")
        envelope_plot = hv.Curve((t, a_T), label="Upper envelope")
        neg_envelope = hv.Curve((t, -1 * a_T), label="Lower envelope")
        surface_elev = hv.Overlay(
            component_plot * envelope_plot * neg_envelope * correct_plot
        ).opts(
            title=answer_list[0],
            xlabel="t [s]",
            ylabel="η [m]",
            aspect=2,
            responsive=True,
            shared_axes=True,
            legend_position="right",
        )

        for i in range(len(student_answers)):
            # Check if each student answer matches the correct answer
            if np.allclose(student_answers[i], correct_answers[i], atol=1e-5):
                return_text += f"Function {i + 1} is correct.\n"
            else:
                return_text += f"Function {i + 1} is incorrect.\n"

            # Plot the answers with holoviews and visualize the differences
            if i >= 1:
                correct_plot = hv.Curve(
                    (x, correct_answers[i]), label=f"Correct {answer_list[i]}"
                ).opts(
                    xlabel="x [m]", ylabel="η [m]", line_dash="dashed", color="green"
                )
                student_plot = hv.Curve(
                    (x, student_answers[i]), label=f"Student {answer_list[i]}"
                )
                figures += hv.Overlay(student_plot * correct_plot).opts(
                    aspect=2,
                    responsive=True,
                    shared_axes=True,
                    legend_position="right",
                    title=answer_list[i],
                )

                component_plot = hv.Curve((x, eta_X), label="η")
                envelope_plot = hv.Curve((x, a_X), label="Envelope")
                neg_envelope = hv.Curve((x, -1 * a_X), label="Negative envelope")
                surface_elev += hv.Overlay(
                    component_plot * envelope_plot * neg_envelope * correct_plot
                ).opts(
                    title=answer_list[i],
                    xlabel="x[m]",
                    ylabel="η [m]",
                    aspect=2,
                    responsive=True,
                    shared_axes=True,
                    legend_position="right",
                )

        figure = hv.Layout(surface_elev + figures).cols(2).opts(shared_axes=False)
        filterwarnings("ignore", category=FutureWarning)
        return return_text, figure

    return check_answer(input_values, student_func, correct_func)
