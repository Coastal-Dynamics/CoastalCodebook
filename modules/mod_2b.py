import timeit

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import holoviews as hv
import ipywidgets as ipw
from ipywidgets import interact, HBox, VBox
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from scipy.signal import hilbert
from IPython.display import display
from warnings import filterwarnings

pn.extension()
hv.extension("bokeh")


def dispersion(k, h):  # calculate omega
    return (9.81 * k * np.tanh(k * h)) ** 0.5


def nfactor(k, h):  # calculate n
    return 0.5 + k * h / np.sinh(2 * k * h)


def wave_length(T, h):
    g = 9.81
    omega = 2 * np.pi / T
    k0 = omega * omega / g
    alpha = k0 * h
    beta = alpha * (np.tanh(alpha)) ** -0.5
    k = (
        (alpha + beta**2 * np.cosh(beta) ** -2)
        / (np.tanh(beta) + beta * np.cosh(beta) ** -2)
        / h
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


def W2_tsunami_L(L1, L2, L3, L4):
    h = 4000

    if L1 == None:
        L1 = 0
    if L2 == None:
        L2 = 0
    if L3 == None:
        L3 = 0
    if L4 == None:
        L4 = 0

    kh2 = np.pi
    k2 = kh2 / h
    w2 = (9.81 * k2 * np.tanh(kh2)) ** 0.5
    T2 = 2 * np.pi / w2

    kh3 = np.pi / 10
    k3 = kh3 / h
    w3 = (9.81 * k3 * np.tanh(kh3)) ** 0.5
    T3 = 2 * np.pi / w3

    T1 = 5 * 60  # 5 min
    T4 = 60 * 60  # 60 min

    w1 = 2 * np.pi / T1
    w4 = 2 * np.pi / T4

    p1 = hv.Points([[w1, L1]], label="L1").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="L [m]",
        color="red",
        marker="circle",
        size=6,
    )
    p2 = hv.Points([[w2, L2]], label="L2").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="L [m]",
        color="orange",
        marker="circle",
        size=6,
    )
    p3 = hv.Points([[w3, L3]], label="L3").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="L [m]",
        color="blue",
        marker="circle",
        size=6,
    )
    p4 = hv.Points([[w4, L4]], label="L4").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="L [m]",
        color="green",
        marker="circle",
        size=6,
    )

    L = np.arange(0.1, 750000)
    k = 2 * np.pi / L
    w = dispersion(k, h)
    L_line = hv.Curve((w, L), label="L").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="L [m]",
        xlim=(0, 0.125),
        ylim=(0, 750000),
    )
    vline1 = hv.VLine((w2), label="deep water limit").opts(
        color="orange", show_legend=True, line_width=1
    )
    vline2 = hv.VLine((w3), label="shallow water limit").opts(
        color="blue", show_legend=True, line_width=1
    )
    figure = hv.Overlay([vline1, vline2, L_line, p1, p2, p3, p4]).opts(
        title="Wave length (h = 4000 m)", aspect=2, frame_width=400, shared_axes=True
    )
    filterwarnings("ignore")
    return display(figure)


def W2_tsunami_c_cg(c1, c2, c3, c4, cg1, cg2, cg3, cg4):
    h = 4000

    if c1 == None:
        c1 = 0
    if c2 == None:
        c2 = 0
    if c3 == None:
        c3 = 0
    if c4 == None:
        c4 = 0

    if cg1 == None:
        cg1 = 0
    if cg2 == None:
        cg2 = 0
    if cg3 == None:
        cg3 = 0
    if cg4 == None:
        cg4 = 0

    kh2 = np.pi
    k2 = kh2 / h
    w2 = (9.81 * k2 * np.tanh(kh2)) ** 0.5
    T2 = 2 * np.pi / w2

    kh3 = np.pi / 10
    k3 = kh3 / h
    w3 = (9.81 * k3 * np.tanh(kh3)) ** 0.5
    T3 = 2 * np.pi / w3

    T1 = 5 * 60  # 5 min
    T4 = 60 * 60  # 60 min

    w1 = 2 * np.pi / T1
    w4 = 2 * np.pi / T4

    p1 = hv.Points([[w1, c1]], label="c₁").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="c [m/s]",
        color="red",
        marker="circle",
        size=6,
    )
    p2 = hv.Points([[w2, c2]], label="c₂").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="c [m/s]",
        color="orange",
        marker="circle",
        size=6,
    )
    p3 = hv.Points([[w3, c3]], label="c₃").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="c [m/s]",
        color="blue",
        marker="circle",
        size=6,
    )
    p4 = hv.Points([[w4, c4]], label="c₄").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="c [m/s]",
        color="green",
        marker="circle",
        size=6,
    )

    p5 = hv.Points([[w1, cg1]], label="cg₁").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="cg [m/s]",
        color="red",
        marker="circle",
        size=6,
    )
    p6 = hv.Points([[w2, cg2]], label="cg₂").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="cg [m/s]",
        color="orange",
        marker="circle",
        size=6,
    )
    p7 = hv.Points([[w3, cg3]], label="cg₃").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="cg [m/s]",
        color="blue",
        marker="circle",
        size=6,
    )
    p8 = hv.Points([[w4, cg4]], label="cg₄").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="cg [m/s]",
        color="green",
        marker="circle",
        size=6,
    )

    L = np.arange(0.1, 750000)
    k = 2 * np.pi / L
    w = dispersion(k, h)
    c = w / k
    n = nfactor(k, h)
    cg = n * c
    c_line = hv.Curve((w, c), label="c").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="c [m/s]",
        xlim=(0, 0.125),
        ylim=(25, 225),
    )
    cg_line = hv.Curve((w, cg), label="cg").opts(
        show_legend=True,
        xlabel="ω [rad/s]",
        ylabel="cg [m/s]",
        xlim=(0, 0.125),
        ylim=(25, 225),
    )
    vline1 = hv.VLine((w2), label="deep water limit").opts(
        color="orange", show_legend=True, line_width=1
    )
    vline2 = hv.VLine((w3), label="shallow water limit").opts(
        color="blue", show_legend=True, line_width=1
    )
    figure1 = hv.Overlay([vline1, vline2, c_line, p1, p2, p3, p4]).opts(
        title="Wave celerity (h = 4000 m)", aspect=2, responsive=True, shared_axes=True
    )

    figure2 = hv.Overlay([vline1, vline2, cg_line, p5, p6, p7, p8]).opts(
        title="Wave group celerity (h = 4000 m)",
        aspect=2,
        responsive=True,
        shared_axes=True,
    )

    figure = figure1 + figure2
    filterwarnings("ignore")
    return display(figure.opts())


def hv_W2_Q7_graph(T1, T2, T3, slope_in, d0):
    # bed profile

    slope = 1.0 / slope_in  # bed slope [-]
    d0  # offshore water depth [m]
    x_max = round((d0 + 2) / slope)
    x = np.arange(0, x_max + 1, 1)  # cross-shore coordinate [m]
    # x = np.linspace(0, x_max + x_max/100, 100)# <-- should be used to reduce computational demands, influences xticks
    zbed = -(d0 - slope * x)  # bed elevation [m]
    h = -zbed  # still water depth [m]
    h[h < 0] = 0  # no negative depths

    # w is zero when h is 0, causing a divide by zero.
    # shorten the lists if a water depth of 0 is reached.
    x0_id = np.argwhere(h == 0)[0][0]  # first location where water depth = 0
    h_water = h[0:x0_id]
    x_water = x[0:x0_id]

    # To solve the xticks problem and shift the axis for plotting so that x=0 at water line.
    x_shift_rev = -x_water[::-1]

    # wavelength through profile
    L1 = [wave_length(T1, h) for h in h_water]
    L2 = [wave_length(T2, h) for h in h_water]
    L3 = [wave_length(T3, h) for h in h_water]

    # phase velocity cross-shore distribution
    # 9.81 * T / (2 * np.pi) * np.tanh(2 * np.pi * h / L)

    def calc_c(T, h):
        L = wave_length(T, h)
        return L / T

    c1 = [calc_c(T1, h) for h in h_water]
    c2 = [calc_c(T2, h) for h in h_water]
    c3 = [calc_c(T3, h) for h in h_water]

    bed = hv.Curve(
        (x_shift_rev, -h_water), label="bed (1:" + str(round(slope_in, 2)) + ")"
    ).opts(color="black", padding=((0, 0.05), 0.1))

    still_water_s = hv.HLine((0), label="still water").opts(
        color="grey",
    )

    bx0 = hv.VLine((0), label="shoreline").opts(color="grey", line_dash="dashed")

    plot0 = (bed * still_water_s * bx0).opts(
        legend_position="right", title="bed profile (z = 0 is still water level)"
    )

    plot0.opts(
        width=500,
        height=300,
        ylabel="z [m]",
        xlabel="cross-shore location x [m]",
    )

    # wavelength

    #    y_max = np.max(([L1], [L2], [L3])) * 1.1

    wL1 = hv.Curve((x_shift_rev, L1), label="wave 1")
    wL2 = hv.Curve((x_shift_rev, L2), label="wave 2")
    wL3 = hv.Curve((x_shift_rev, L3), label="wave 3")

    wx0 = hv.VLine((0), label="shoreline").opts(color="grey", line_dash="dashed")

    plot1 = (wL1 * wL2 * wL3 * wx0).opts(legend_position="right", title="wave length")
    plot1.opts(
        #        ylim=(0, y_max),
        width=500,
        height=300,
        ylabel="wave length L [m]",
        xlabel="cross-shore location x [m]",
        padding=((0, 0.05), 0.1),
    )

    # wave celerity
    #    y_max = np.max(([c1], [c2], [c3])) * 1.1

    wc1 = hv.Curve((x_shift_rev, c1), label="wave 1")
    wc2 = hv.Curve((x_shift_rev, c2), label="wave 2")
    wc3 = hv.Curve((x_shift_rev, c3), label="wave 3")

    plot2 = (wc1 * wc2 * wc3 * wx0).opts(legend_position="right", title="wave celerity")
    plot2.opts(
        # ylim=(0, y_max),
        width=500,
        height=300,
        ylabel="celerity c [m/s]",
        xlabel="cross-shore location x [m]",
        padding=((0, 0.05), 0.1),
    )

    filterwarnings("ignore", category=FutureWarning)
    plot = (
        hv.Layout(plot1 + plot2 + plot0 + plot0)
        .cols(2)
        .opts(
            title="Cross-shore distribution of wave length and wave celerity",
            shared_axes=False,
        )
    )

    display(plot)


def hv_W2_Q7():
    # Create interactive widgets, which require IPY Widgets, widgets from panel do not work
    T1 = ipw.FloatSlider(value=4, min=1, max=20, step=0.01, description="T1 [s]")
    T2 = ipw.FloatSlider(value=7, min=1, max=20, step=0.01, description="T2 [s]")
    T3 = ipw.FloatSlider(value=25, min=1, max=20, step=0.01, description="T3 [s]")

    slope = ipw.FloatSlider(
        value=75, min=50, max=200, step=0.1, description="slope 1:..."
    )
    d0 = ipw.FloatSlider(
        value=50, min=0.1, max=500, step=0.1, description="max. depth [m]"
    )

    # Setup widget layout (User Interface) for the graph input
    vbox1 = ipw.VBox(
        [
            ipw.Label("Waves", layout=ipw.Layout(align_self="center")),
            T1,
            T2,
            T3,
        ]
    )
    vbox2 = ipw.VBox(
        [ipw.Label("Bed profile", layout=ipw.Layout(align_self="center")), slope, d0]
    )
    UI = ipw.HBox([vbox1, vbox2])

    # Use the interactive function to update the plot
    graph = ipw.interactive_output(
        hv_W2_Q7_graph, {"T1": T1, "T2": T2, "T3": T3, "slope_in": slope, "d0": d0}
    )

    filterwarnings("ignore", category=FutureWarning)
    #    display(UI, graph, intro_widget, *questions)
    display(UI, graph)


def W2_Q9_t(input_values, student_func):
    h, a, T1, T2, L1, L2, t, xp = input_values.values()

    def eta_t(a, T1, T2, L1, L2, t, xp):
        eta1_T = a * np.sin(2 * np.pi / T1 * t - 2 * np.pi / L1 * xp)
        eta2_T = a * np.sin(2 * np.pi / T2 * t - 2 * np.pi / L2 * xp)
        eta_T = eta1_T + eta2_T
        return eta_T

    def varying_amplitude_t(a, T1, T2, L1, L2, t, xp):
        w1 = 2 * np.pi / T1
        w2 = 2 * np.pi / T2
        k1 = 2 * np.pi / L1
        k2 = 2 * np.pi / L2
        Delta_w = w2 - w1
        Delta_k = k2 - k1
        var_amp = 2 * a * np.cos(0.5 * Delta_w * t - 0.5 * Delta_k * xp)
        return var_amp

    def envelope_t(a, T1, T2, L1, L2, t, xp):
        envelope = np.abs(varying_amplitude_t(a, T1, T2, L1, L2, t, xp))
        return envelope

    correct_func = [eta_t, varying_amplitude_t, envelope_t]

    def check_answer(input_values, student_func, correct_func):
        # Input values depends on the question at hand, so change accordingly
        h, a, T1, T2, L1, L2, t_val, xp = input_values.values()

        L1 = wave_length(T1, h)
        L2 = wave_length(T2, h)

        # student_func and correct_func shoud be provided as a list element, so that multiple functions
        # can be checked at once. This method remains applicable for one function
        student_answers = []
        correct_answers = []

        # loop through the functions and store the student and correct answers
        for i in range(len(student_func)):
            student_answers.append(
                student_func[i](a, T1, T2, L1, L2, t_val, xp)
                #                if student_func[i](a, T1, T2, L1, L2, t_val, xp) != None
                if student_func[i](a, T1, T2, L1, L2, t_val, xp) is not None
                else np.zeros(len(t_val))
            )
            correct_answers.append(correct_func[i](a, T1, T2, L1, L2, t_val, xp))

        # Provide feedback if student answers are correct or not
        return_text = "Your time functions: " + "\n"
        clabels = ["η", "Slowly-varying amplitude", "Upper envelope"]

        student_plot = hv.Curve(
            (t_val, student_answers[0]), label="Time function 1"
        ).opts(xlabel="time [s]", ylabel="η [m]", title="Student functions")
        correct_plot = hv.Curve((t_val, correct_answers[0]), label=clabels[0]).opts(
            xlabel="time [s]", ylabel="η [m]", title="Correct functions"
        )

        for i in range(len(student_answers)):
            # Check if each student answer matches the correct answer
            if np.allclose(student_answers[i], correct_answers[i], atol=1e-5):
                return_text += f"Function {i + 1} is correct.\n"
            else:
                return_text += f"Function {i + 1} is incorrect.\n"

            # Plot the answers with holoviews and visualize the differences
            if i == 1:
                correct_plot *= hv.Curve(
                    (t_val, correct_answers[i]), label=f"{clabels[i]}"
                )
                student_plot *= hv.Curve(
                    (t_val, student_answers[i]), label=f"Time function {i + 1}"
                )
            elif i == 2:
                correct_plot *= hv.Curve(
                    (t_val, correct_answers[i]), label=f"{clabels[i]}"
                ).opts(
                    line_dash="dashed",
                    aspect=2,
                    responsive=True,  # frame_width=450)
                )
                student_plot *= hv.Curve(
                    (t_val, student_answers[i]), label=f"Time function {i + 1}"
                ).opts(
                    line_dash="dashed",
                    aspect=2,
                    responsive=True,  # frame_width=450)
                )

        figure = hv.Layout(
            student_plot.opts(legend_position="right", aspect=2, responsive=True)
            + correct_plot.opts(legend_position="right", aspect=2, responsive=True)
        ).cols(2)
        return return_text, figure

    return check_answer(input_values, student_func, correct_func)


def W2_Q9_x(input_values, student_func):
    h, a, T1, T2, L1, L2, x, tp = input_values

    def eta_x(a, T1, T2, L1, L2, x, tp):
        eta1_X = a * np.sin(2 * np.pi / T1 * tp - 2 * np.pi / L1 * x)
        eta2_X = a * np.sin(2 * np.pi / T2 * tp - 2 * np.pi / L2 * x)
        eta_X = eta1_X + eta2_X
        return eta_X

    def varying_amplitude_x(a, T1, T2, L1, L2, x, tp):
        w1 = 2 * np.pi / T1
        w2 = 2 * np.pi / T2
        k1 = 2 * np.pi / L1
        k2 = 2 * np.pi / L2
        Delta_w = w2 - w1
        Delta_k = k2 - k1
        var_amp = 2 * a * np.cos(0.5 * Delta_w * tp - 0.5 * Delta_k * x)
        return var_amp

    def envelope_x(a, T1, T2, L1, L2, x, tp):
        envelope = np.abs(varying_amplitude_x(a, T1, T2, L1, L2, x, tp))
        return envelope

    correct_func = [eta_x, varying_amplitude_x, envelope_x]

    def check_answer(input_values, student_func, correct_func):
        h, a, T1, T2, L1, L2, x_val, tp = input_values.values()
        L1 = wave_length(T1, h)
        L2 = wave_length(T2, h)

        # student_func and correct_func shoud be provided as a list element, so that multiple functions
        # can be checked at once. This method remains applicable for one function
        student_answers = []
        correct_answers = []

        # loop through the functions and store the student and correct answers
        for i in range(len(student_func)):
            student_answers.append(
                student_func[i](a, T1, T2, L1, L2, x_val, tp)
                #                if student_func[i](a, T1, T2, L1, L2, x_val, tp) != None
                if student_func[i](a, T1, T2, L1, L2, x_val, tp) is not None
                else np.zeros(len(x_val))
            )
            correct_answers.append(correct_func[i](a, T1, T2, L1, L2, x_val, tp))

        # Provide feedback if student answers are correct or not
        return_text = "Your spatial functions: " + "\n"
        clabels = ["η", "Slowly-varying amplitude", "Upper envelope"]

        student_plot = hv.Curve(
            (x_val, student_answers[0]), label="Spatial function 1"
        ).opts(xlabel="x [m]", ylabel="η [m]", title="Student functions")
        correct_plot = hv.Curve((x_val, correct_answers[0]), label=clabels[0]).opts(
            xlabel="x [m]", ylabel="η [m]", title="Correct functions"
        )

        for i in range(len(student_answers)):
            # Check if each student answer matches the correct answer
            if np.allclose(
                student_answers[i], correct_answers[i], rtol=1e-5, atol=1e-5
            ):
                return_text += f"Function {i + 1} is correct.\n"
            else:
                return_text += f"Function {i + 1} is incorrect.\n"

            # Plot the answers with holoviews and visualize the differences
            if i == 1:
                correct_plot *= hv.Curve(
                    (x_val, correct_answers[i]), label=f"{clabels[i]}"
                )
                student_plot *= hv.Curve(
                    (x_val, student_answers[i]), label=f"Spatial function {i + 1}"
                )
            elif i == 2:
                correct_plot *= hv.Curve(
                    (x_val, correct_answers[i]), label=f"{clabels[i]}"
                ).opts(
                    line_dash="dashed",
                    aspect=2,
                    responsive=True,  # frame_width=450)
                )
                student_plot *= hv.Curve(
                    (x_val, student_answers[i]), label=f"Spatial function {i + 1}"
                ).opts(
                    line_dash="dashed",
                    aspect=2,
                    responsive=True,  # frame_width=450)
                )

        figure = hv.Layout(
            student_plot.opts(legend_position="right", aspect=2, responsive=True)
            + correct_plot.opts(legend_position="right", aspect=2, responsive=True)
        ).cols(2)
        return return_text, figure

    return check_answer(input_values, student_func, correct_func)


## JBO For review: why phi1 and phi2 in below function at default zero
def W2_wave_groups():  # by Lisa, modified by Kevin to use panel instead of ipywidgets
    # define widgets

    a1_widget = pn.widgets.FloatInput(name="a [m]", value=1, step=0.01, start=0, end=20)
    a2_widget = pn.widgets.FloatInput(
        name="a [m]", value=1.5, step=0.01, start=0, end=20
    )

    T1_widget = pn.widgets.FloatInput(
        name="T [s]", value=7, step=0.01, start=0.01, end=250000
    )
    T2_widget = pn.widgets.FloatInput(
        name="T [s]", value=6.2, step=0.01, start=0.01, end=250000
    )

    phi1_widget = pn.widgets.FloatInput(
        value=0, name="phi [2 pi rad]", step=0.01, start=-2, end=2
    )
    phi2_widget = pn.widgets.FloatInput(
        value=0, name="phi [2 pi rad]", step=0.01, start=-2, end=2
    )

    n_waves_widget = pn.widgets.FloatInput(
        name="# groups", value=3, step=0.1, start=0.1, end=10
    )

    depth_widget = pn.widgets.FloatInput(
        name="h [m]", value=20, start=1, end=250, step=0.01
    )
    xp_widget = pn.widgets.FloatInput(value=0, step=0.01, name="x [m]")
    tp_widget = pn.widgets.FloatInput(value=0, step=0.1, name="t [s]")

    L1 = wave_length(T1_widget.value, depth_widget.value)
    L2 = wave_length(T2_widget.value, depth_widget.value)

    c1_widget = pn.widgets.StaticText(
        value=f"{L1 / T1_widget.value:.2f}", name="c [m/s]"
    )
    c2_widget = pn.widgets.StaticText(
        value=f"{L2 / T2_widget.value:.2f}", name="c [m/s]"
    )

    h_L1_widget = pn.widgets.StaticText(
        value=f"{depth_widget.value / L1:.2f}", name="h/L [-]"
    )
    h_L2_widget = pn.widgets.StaticText(
        value=f"{depth_widget.value / L2:.2f}", name="h/L [-]"
    )

    Lgroup_v, Tgroup_v, cgroup_v = group_stats(
        k1=2 * np.pi / L1,
        k2=2 * np.pi / L2,
        w1=2 * np.pi / T1_widget.value,
        w2=2 * np.pi / T2_widget.value,
    )

    L_group_widget = pn.widgets.StaticText(
        value=f"{Lgroup_v:.2f}",
        name="L_group [m]",
    )

    T_group_widget = pn.widgets.StaticText(
        value=f"{Tgroup_v:.2f}",
        name="T_group [s]",
    )

    c_group_widget = pn.widgets.StaticText(
        value=f"{cgroup_v:.2f}",
        name="c_group [m/s]",
    )

    # Setup widget layout (User Interface)
    vbox1 = pn.Column(
        pn.pane.Markdown("### Wave 1"),
        a1_widget,
        T1_widget,
        # phi1_widget,
        c1_widget,
        h_L1_widget,
    )
    vbox2 = pn.Column(
        pn.pane.Markdown("### Wave 2"),
        a2_widget,
        T2_widget,
        # phi2_widget,
        c2_widget,
        h_L2_widget,
    )
    vbox3 = pn.Column(
        pn.pane.Markdown("### Wave group"),
        n_waves_widget,
        L_group_widget,
        T_group_widget,
        c_group_widget,
    )
    vbox4 = pn.Column(
        pn.pane.Markdown("### General"), depth_widget, xp_widget, tp_widget
    )

    ui = pn.Row(vbox1, vbox2, vbox3, vbox4)

    @pn.depends(
        a1_widget.param.value,
        T1_widget.param.value,
        phi1_widget.param.value,
        a2_widget.param.value,
        T2_widget.param.value,
        phi2_widget.param.value,
        n_waves_widget.param.value,
        xp_widget.param.value,
        tp_widget.param.value,
        depth_widget.param.value,
    )
    def calc_eta(
        a1,
        T1,
        phi_1,
        a2,
        T2,
        phi_2,
        n_waves,
        xp,
        tp,
        depth,
    ):
        # compute new values
        L1 = wave_length(T1, depth)
        L2 = wave_length(T2, depth)
        c1 = L1 / T1
        c2 = L2 / T2
        h_L1 = depth / L1
        h_L2 = depth / L2

        L_group, T_group, c_group = group_stats(
            k1=2 * np.pi / L1,
            k2=2 * np.pi / L2,
            w1=2 * np.pi / T1,
            w2=2 * np.pi / T2,
        )
        # set new values in widgets
        c1_widget.value = f"{c1:.2f}"
        c2_widget.value = f"{c2:.2f}"
        h_L1_widget.value = f"{h_L1:.2f}"
        h_L2_widget.value = f"{h_L2:.2f}"

        L_group_widget.value = f"{L_group:.2f}"
        T_group_widget.value = f"{T_group:.2f}"
        c_group_widget.value = f"{c_group:.2f}"

        ################# CREATE PLOT FROM NEW VALUES #################
        T = np.min([T1, T2])
        L = np.max([L1, L2])
        # requires additional x and t values to get a correct Hilbert transformation at the graph boundaries
        # t = np.arange(0,n_waves*T_group+T/30,T/30)
        # x = np.arange(0,n_waves*L_group+L/30,L/30)
        t = np.arange(-0.5 * n_waves * T_group, (n_waves + 0.5) * T_group, T / 30)
        x = np.arange(-0.5 * n_waves * L_group, (n_waves + 0.5) * L_group, L / 30)

        # calculate surface, including phase change
        eta1_T = a1 * np.sin(
            2 * np.pi / T1 * t - 2 * np.pi / L1 * xp - phi_1 * (2 * np.pi)
        )
        eta2_T = a2 * np.sin(
            2 * np.pi / T2 * t - 2 * np.pi / L2 * xp - phi_2 * (2 * np.pi)
        )
        eta_T = eta1_T + eta2_T

        eta1_x = a1 * np.sin(
            2 * np.pi / T1 * tp - 2 * np.pi / L1 * x - phi_1 * (2 * np.pi)
        )
        eta2_x = a2 * np.sin(
            2 * np.pi / T2 * tp - 2 * np.pi / L2 * x - phi_2 * (2 * np.pi)
        )
        eta_x = eta1_x + eta2_x

        # calculate surface, without phase change
        eta1_T_basic = a1 * np.sin(2 * np.pi / T1 * t - 2 * np.pi / L1 * xp)
        eta2_T_basic = a2 * np.sin(2 * np.pi / T2 * t - 2 * np.pi / L2 * xp)
        eta_T_basic = eta1_T_basic + eta2_T_basic

        eta1_x_basic = a1 * np.sin(2 * np.pi / T1 * tp - 2 * np.pi / L1 * x)
        eta2_x_basic = a2 * np.sin(2 * np.pi / T2 * tp - 2 * np.pi / L2 * x)
        eta_x_basic = eta1_x_basic + eta2_x_basic

        # calculate hilbert
        eta_T_envelope = np.abs(hilbert(eta_T_basic))
        eta_x_envelope = np.abs(hilbert(eta_x_basic))

        # carier wave
        k_bar = 2 * np.pi / L1 + 2 * np.pi / L2
        w_bar = (2 * np.pi / T1 + 2 * np.pi / T2) / 2
        car_wave_t = (a1 + a2) * np.sin(w_bar * t - k_bar * xp)
        car_wave_x = (a1 + a2) * np.sin(w_bar * tp - k_bar * x)

        # variable amplitude
        Delta_k = 2 * np.pi / L1 - 2 * np.pi / L2  # Delta k
        Delta_w = 2 * np.pi / T1 - 2 * np.pi / T2
        var_amp_t = (a1 + a2) * np.cos(Delta_w / 2 * t - Delta_k / 2 * xp)
        var_amp_x = (a1 + a2) * np.cos(Delta_w / 2 * tp - Delta_k / 2 * x)

        # plot surface including phase change
        plot3_1 = hv.Curve((t, eta_T), label="Surface elevation η")
        plot3_2 = hv.Curve((t, eta_T_envelope), label="Upper envelope")
        plot3_3 = hv.Curve((t, -1 * eta_T_envelope), label="Lower envelope")
        time_based = plot3_1 * plot3_2 * plot3_3

        plot6_1 = hv.Curve((x, eta_x), label="Surface elevation η")
        plot6_2 = hv.Curve((x, eta_x_envelope), label="Upper envelope")
        plot6_3 = hv.Curve((x, -1 * eta_x_envelope), label="Lower envelope")
        space_based = plot6_1 * plot6_2 * plot6_3

        # set vertical axis the same
        amp = (a1 + a2) * 1.1

        time_based.opts(
            xlabel="t/T_group",
            ylabel="η [m]",
            xlim=(0, n_waves * T_group),
            ylim=(-amp, amp),
            aspect=2,
            # responsive=True,
            frame_width=450,
            legend_position="right",
            title="Time-based (x = " + str(xp) + " m)",
        )

        space_based.opts(
            xlabel="x/L_group",
            ylabel="η [m]",
            xlim=(0, n_waves * L_group),
            ylim=(-amp, amp),
            show_legend=True,
            aspect=2,
            # responsive=True,
            frame_width=450,
            legend_position="right",
            title="Space-based (t = " + str(tp) + " s)",
        )

        # set scaled ticks
        if n_waves >= 1:
            time_based.opts(
                xticks=[(i * T_group, i) for i in np.arange(0, n_waves // 1 + 1, 1)]
            )
            space_based.opts(
                xticks=[(i * L_group, i) for i in np.arange(0, n_waves // 1 + 1, 1)]
            )

        else:  # 3 times when the scale is smaller than 1
            time_based.opts(
                xticks=[
                    (0, 0),
                    (0.5 * n_waves * T_group, 0.5 * n_waves),
                    (n_waves * T_group, n_waves),
                ]
            )
            space_based.opts(
                xticks=[
                    (0, 0),
                    (0.5 * n_waves * L_group, 0.5 * n_waves),
                    (n_waves * L_group, n_waves),
                ]
            )

        # plot legends
        layout = hv.Layout(time_based + space_based).cols(2).opts(shared_axes=False)

        return layout

    app = pn.Column(ui, calc_eta)

    return app


## JBO Since the wave animation is so slow it is not used in the notebook. We must review and speed it up.
def W2_Wave_animation():
    # Below is a graph that shows two wave components propagating. You can define the amplitude and period of two wave components and set the water depth.
    # (Not the wavelength) You can zoom in and out to obtain a specific number of waves of the wave group on the x-axis by setting "n_waves group".
    # The Hilbert Transform is applied to show the variable wave amplitude.
    # Can you find out when the waves are dispersive and when not? How is this visible on the graph? <br>
    # Unfortunately is the graph very computationally demanding, resulting in low frames per second, so it is a challenge to see the action,
    # but you might think of the correct reasoning.

    from scipy.signal import hilbert

    # adjusting the graph while it is on pause can be achieved by widget.observe(). See old Week_3_initialize.ipynb
    a1 = pn.widgets.FloatInput(
        name="a1 [m]", start=0, end=3, step=0.01, value=1, width=75
    )
    a2 = pn.widgets.FloatInput(
        name="a2 [m]", start=0, end=3, step=0.01, value=1, width=75
    )

    T1 = pn.widgets.FloatInput(
        name="T1 [s]", start=0, step=0.01, value=5, width=75
    )  # 3
    T2 = pn.widgets.FloatInput(
        name="T2 [s]", start=0, step=0.01, value=3, width=75
    )  # 5

    depth = pn.widgets.FloatInput(
        name="h [m]", start=0.01, end=250, step=0.01, value=2, width=75
    )
    n_waves = pn.widgets.FloatInput(
        name="n wₐᵥₑₛ group", start=0.02, end=20, step=0.01, value=10, width=75
    )

    time = pn.widgets.FloatInput(name="time [s]", start=0, value=0, width=75)
    f_time = pn.widgets.FloatInput(
        name="play speed", start=0.1, value=1, width=75, step=0.1
    )

    L1 = pn.widgets.FloatInput(
        name="L1 [m]",
        start=0,
        step=0.01,
        value=wave_length(T1.value, depth.value),
        width=75,
        disabled=True,
    )
    L2 = pn.widgets.FloatInput(
        name="L2 [m]",
        start=0,
        step=0.01,
        value=wave_length(T2.value, depth.value),
        width=75,
        disabled=True,
    )

    L_group, T_group, c_g = group_stats(
        k1=2 * np.pi / L1.value,
        k2=2 * np.pi / L2.value,
        w1=2 * np.pi / T1.value,
        w2=2 * np.pi / T2.value,
    )

    # Setup widget layout (User Interface) and display
    vbox1 = pn.Column("Wave 1", a1, T1, L1)
    vbox2 = pn.Column("Wave 2", a2, T2, L2)
    vbox3 = pn.Column(
        "General",
        depth,
        n_waves,
    )
    vbox4 = pn.Column("Play settings", f_time, time)

    # Define and display User Interface (UI)
    ui = pn.Row(vbox1, vbox2, vbox3, vbox4)
    display(ui)

    # Setup linear mesh (x) and duration before time (t) is reset
    x_max = L_group * n_waves.value
    x = np.linspace(0, x_max, 500)

    # Create figure, set structure and initial layout
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 5), sharex=True)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(bottom=0.2)  # Add some extra space for the axis at the bottom
    fig.subplots_adjust(left=0.1)  # Add some space for the labels

    # set grid distance
    grid_x = MultipleLocator(base=25)
    grid_y = MultipleLocator(base=5)

    # Set grid for multiple axes
    for ax in axs:
        ax.xaxis.set_minor_locator(grid_x)
        ax.yaxis.set_minor_locator(grid_y)
        ax.grid(which="both", linestyle="-", linewidth="0.5", color="gray", alpha=0.6)

    # Compute initial displacement
    eta = a1.value * np.sin(2 * np.pi / L1.value * x) + a2.value * np.sin(
        2 * np.pi / L2.value * x
    )
    eta1 = a1.value * np.sin(2 * np.pi / L1.value * x)
    eta2 = a2.value * np.sin(2 * np.pi / L2.value * x)

    k_bar = (2 * np.pi / L1.value + 2 * np.pi / L2.value) / 2
    car_wave = (a1.value + a2.value) * np.sin(k_bar * x)  # carrier wave

    k_dif = 2 * np.pi / L1.value - 2 * np.pi / L2.value  # Delta k
    var_amp = np.cos(-k_dif / 2 * x)  # variable amplitude
    var_amp_scaled = (a1.value + a2.value) * np.cos(
        -k_dif / 2 * x
    )  # variable amplitude

    # calculate hilbert
    eta_hilbert = np.abs(hilbert(eta))

    # Plot initial wave
    (line1,) = axs[0].plot(
        x, eta1, label="$\u03B7_1$ (wave 1)", linewidth=0.75, color="#0b5394"
    )
    (line2,) = axs[1].plot(
        x, eta2, label="$\u03B7_2$ (wave 2)", linewidth=0.75, color="#0b5394"
    )  # 03396c
    (line,) = axs[2].plot(
        x, eta, label="$\u03B7$ (wave 1+2)", linewidth=0.75, color="k"
    )
    (line_var,) = axs[2].plot(
        x, var_amp_scaled, label="Variable amplitude", linewidth=0.75, color="gray"
    )

    # set initial layout and make legends
    amp = a1.value + a2.value
    for ax in axs:
        ax.set_xlim(0, x_max)
        ax.set_ylim(-amp * 1.15, amp * 1.15)
        legend = ax.legend(loc="lower right")

        for text in legend.get_texts():
            text.set_fontsize(7)  # Set individual legend item text size

    start_time = timeit.default_timer()

    # adjust the graph when the animation is running
    def update_line(change):
        t = change.new
        t = (timeit.default_timer() - start_time) * f_time.value

        # adjust the widget
        time.value = t

        L1.value = wave_length(T1.value, h=depth.value)
        L2.value = wave_length(T2.value, h=depth.value)
        L_group, T_group, c_g = group_stats(
            k1=2 * np.pi / L1.value,
            k2=2 * np.pi / L2.value,
            w1=2 * np.pi / T1.value,
            w2=2 * np.pi / T2.value,
        )
        x_max = L_group * n_waves.value
        for ax in axs:
            ax.set_xlim(0, x_max)

        x = np.linspace(0, x_max, 750)

        # calculate sea surface elevation
        eta = a1.value * np.sin(
            2 * np.pi / T1.value * t - 2 * np.pi / L1.value * x
        ) + a2.value * np.sin(2 * np.pi / T2.value * t - 2 * np.pi / L2.value * x)
        eta1 = a1.value * np.sin(2 * np.pi / T1.value * t - 2 * np.pi / L1.value * x)
        eta2 = a2.value * np.sin(2 * np.pi / T2.value * t - 2 * np.pi / L2.value * x)

        omega_bar = (2 * np.pi / T1.value + 2 * np.pi / T2.value) / 2
        k_bar = (2 * np.pi / L1.value + 2 * np.pi / L2.value) / 2

        # calculate hilbert
        eta_hilbert = np.abs(hilbert(eta))

        # adjust sea surface elevation (line) in plot
        line.set_ydata(eta)
        line1.set_ydata(eta1)
        line2.set_ydata(eta2)
        line_var.set_ydata(eta_hilbert)
        line.set_xdata(x)
        line1.set_xdata(x)
        line2.set_xdata(x)
        line_var.set_xdata(x)

        amp = a1.value + a2.value
        for ax in axs:
            ax.set_ylim(-amp * 1.15, amp * 1.15)

        fig.canvas.draw()

    delta_t = 0.0005  # s
    discrete_player = pn.widgets.DiscretePlayer(
        name="Discrete Player",
        options=np.arange(0, 500, delta_t * f_time.value).tolist(),
        value=0,
        loop_policy="loop",
        interval=int(delta_t * 1000),
    )

    discrete_player.param.watch(update_line, "value")
    display(discrete_player)
