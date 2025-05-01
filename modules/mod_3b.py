import sys
from warnings import filterwarnings

import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts
import ipywidgets as ipw
from IPython.display import display

hv.extension("bokeh")
pn.extension()


def second_order_waves():  # second-order_Stokes_waves
    a1 = ipw.FloatText(value=1, min=0, max=20, step=0.01, description="a₁ [m]")
    a2 = ipw.FloatText(value=0, min=0, max=20, step=0.01, description="a₂ [m]")
    phi1 = ipw.FloatText(value=0, min=0, max=20, step=0.01, description="φ₁ [π rad]")
    phi2 = ipw.FloatText(value=0, min=0, max=20, step=0.01, description="φ₂ [π rad]")

    # Setup widget layout (User Interface)
    vbox1 = ipw.VBox(
        [ipw.Label("η₁", layout=ipw.Layout(align_self="center")), a1, phi1]
    )
    vbox2 = ipw.VBox(
        [ipw.Label("η₂", layout=ipw.Layout(align_self="center")), a2, phi2]
    )
    ui = ipw.HBox([vbox1, vbox2])

    def update_graph(a1, a2, phi1, phi2):
        S_min = -0.25 * np.pi
        S_max = 2.25 * np.pi

        S = np.linspace(S_min, S_max, 60)
        S1 = S - phi1 * np.pi
        S2 = S * 2 - phi2 * np.pi

        eta1 = a1 * np.cos(S1)
        eta2 = a2 * np.cos(S2)
        eta = eta1 + eta2

        x_tick_min = S_min // (0.5 * np.pi) * 0.5 * np.pi + 0.5 * np.pi
        x_ticks = np.arange(x_tick_min, S_max, 0.5 * np.pi)
        x_ticks_labels = [(angle, f"{angle/np.pi:.1f}π") for angle in x_ticks]

        part1 = hv.Curve((S, eta1), label="η₁").opts(
            color="grey",
            line_dash="dashed",
            show_legend=True,
            xticks=x_ticks_labels,
            xlim=(S_min, S_max),
        )
        part2 = hv.Curve((S, eta2), label="η₂").opts(
            line_dash="dashdot",
            color="grey",
            show_legend=True,
            xticks=x_ticks_labels,
            xlim=(S_min, S_max),
        )
        part3 = hv.Curve((S, eta), label="η₁+η₂").opts(
            color="k", show_legend=True, xticks=x_ticks_labels, xlim=(S_min, S_max)
        )

        figure = hv.Overlay(part1 * part2 * part3).opts(
            ylabel="η [m]",
            xlabel="phase [rad]",
            width=500,
            height=250,
            shared_axes=False,
            legend_position="right",
            #            aspect=1.3, responsive=True,
            show_grid=True,
        )
        filterwarnings("ignore", category=FutureWarning)
        display(figure)

    graph = ipw.interactive_output(
        update_graph, {"a1": a1, "a2": a2, "phi1": phi1, "phi2": phi2}
    )

    display(ui, graph)


def check_second_order_waves_x_L(input_values, student_func):
    x_L, eta1, eta2, phi1, phi2 = input_values

    def second_order_waves_x_L(xL, eta1, eta2, phi1, phi2):
        eta = eta1 * np.cos(-2 * np.pi * xL - phi1 * np.pi) + eta2 * np.cos(
            2 * (-2 * np.pi * xL) - phi2 * np.pi
        )
        return eta

    fig_xL_func = [second_order_waves_x_L]

    def check_answer(input_values, student_func, fig_xL_func):
        # Input values depends on the question at hand, so change accordingly
        x_L, eta1, eta2, phi1, phi2 = input_values.values()

        # student_func and correct_func shoud be provided as a list element, so that multiple functions
        # can be checked at once. This method remains applicable for one function
        student_answers = []
        fig_xL_answers = []

        # loop through the functions and store the student and correct answers
        for i in range(len(student_func)):
            student_answers.append(
                student_func[i](x_L, eta1, eta2, phi1, phi2)
                if student_func[i](x_L, eta1, eta2, phi1, phi2) is not None
                else np.zeros(len(x_L))
            )
            fig_xL_answers.append(fig_xL_func[i](x_L, eta1, eta2, phi1, phi2))

        # Provide feedback if student answers are correct or not
        return_text = "Your function is: " + "\n"

        student_plot = hv.Curve(
            (x_L, student_answers[0]), label="Student function"
        ).opts(xlabel="x/L", ylabel="η [m]", xlim=(-0.1, 1.1))
        fig_xL_plot = hv.Curve((x_L, fig_xL_answers[0]), label="Correct function").opts(
            xlabel="x/L", ylabel="η [m]", xlim=(-0.1, 1.1)
        )

        for i in range(len(student_answers)):
            # Check if each student answer matches the correct answer
            if np.allclose(student_answers[i], fig_xL_answers[i], atol=1e-5):
                return_text += f"Function {i + 1} is correct.\n"
            else:
                return_text += f"Function {i + 1} is incorrect.\n"

        figure = hv.Layout(
            student_plot.opts(aspect=1.3, responsive=True, show_grid=True)
            + fig_xL_plot.opts(aspect=1.3, responsive=True, show_grid=True)
        ).cols(2)
        return return_text, figure

    return check_answer(input_values, student_func, fig_xL_func)


def check_second_order_waves_t_T(input_values, student_func):
    t_T, eta1, eta2, phi1, phi2 = input_values.values()

    def second_order_waves_t_T(t_T, eta1, eta2, phi1, phi2):
        eta = eta1 * np.cos(2 * np.pi * t_T - phi1 * np.pi) + eta2 * np.cos(
            4 * np.pi * t_T - phi2 * np.pi
        )
        return eta

    fig_tT_func = [second_order_waves_t_T]

    def check_answer(input_values, student_func, fig_tT_func):
        # Input values depends on the question at hand, so change accordingly
        t_T, eta1, eta2, phi1, phi2 = input_values.values()

        # student_func and correct_func shoud be provided as a list element, so that multiple functions
        # can be checked at once. This method remains applicable for one function
        student_answers = []
        fig_tT_answers = []

        # loop through the functions and store the student and correct answers
        for i in range(len(student_func)):
            student_answers.append(
                student_func[i](t_T, eta1, eta2, phi1, phi2)
                if student_func[i](t_T, eta1, eta2, phi1, phi2) is not None
                else np.zeros(len(t_T))
            )
            fig_tT_answers.append(fig_tT_func[i](t_T, eta1, eta2, phi1, phi2))

        # Provide feedback if student answers are correct or not
        return_text = "Your function is: " + "\n"

        student_plot = hv.Curve(
            (t_T, student_answers[0]), label="Student function"
        ).opts(xlabel="t/T", ylabel="η [m]", xlim=(-0.1, 1.1))
        fig_tT_plot = hv.Curve((t_T, fig_tT_answers[0]), label="Correct function").opts(
            xlabel="t/T", ylabel="η [m]", xlim=(-0.1, 1.1)
        )

        for i in range(len(student_answers)):
            # Check if each student answer matches the correct answer
            if np.allclose(student_answers[i], fig_tT_answers[i], atol=1e-5):
                return_text += f"Function {i + 1} is correct.\n"
            else:
                return_text += f"Function {i + 1} is incorrect.\n"

        figure = hv.Layout(
            student_plot.opts(aspect=1.3, responsive=True, show_grid=True)
            + fig_tT_plot.opts(aspect=1.3, responsive=True, show_grid=True)
        ).cols(2)
        return return_text, figure

    return check_answer(input_values, student_func, fig_tT_func)


def asymmetric():
    a1 = ipw.FloatText(value=1, min=0, max=20, step=0.01, description="a₁ [m]")
    a2_1 = ipw.FloatText(value=0, min=0, max=20, step=0.01, description="a₂ [m]")
    a2_2 = ipw.FloatText(value=0, min=0, max=20, step=0.01, description="a₂ [m]")
    phi1 = ipw.FloatText(value=0, min=0, max=20, step=0.01, description="φ₁ [π rad]")
    phi2_1 = ipw.FloatText(value=0, min=0, max=20, step=0.01, description="φ₂ [π rad]")
    phi2_2 = ipw.FloatText(value=0, min=0, max=20, step=0.01, description="φ₂ [π rad]")

    # Setup widget layout (User Interface)
    vbox1 = ipw.VBox(
        [ipw.Label("η₁", layout=ipw.Layout(align_self="center")), a1, phi1]
    )
    vbox2 = ipw.VBox(
        [
            ipw.Label("η₂ for skewed waves", layout=ipw.Layout(align_self="center")),
            a2_1,
            phi2_1,
        ]
    )
    vbox3 = ipw.VBox(
        [
            ipw.Label(
                "η₂ for asymmetric waves", layout=ipw.Layout(align_self="center")
            ),
            a2_2,
            phi2_2,
        ]
    )
    ui = ipw.HBox([vbox1, vbox2, vbox3])

    def update_graph(a1, a2_1, a2_2, phi1, phi2_1, phi2_2):
        S_min = -0.25 * np.pi
        S_max = 2.25 * np.pi
        S = np.linspace(S_min, S_max, 100)

        # left panel
        eta_L = a1 * np.cos(S - phi1)
        eta_3_L = eta_L**3

        # Middle panel
        eta_M = a1 * np.cos(S - phi1) + a2_1 * np.cos(S * 2 - phi2_1 * np.pi)
        eta_3_M = eta_M**3

        # right panel
        eta_R = a1 * np.cos(S - phi1) + a2_2 * np.cos(S * 2 - phi2_2 * np.pi)
        eta_3_R = eta_R**3

        # define labels for x axis
        x_tick_min = S_min // (0.5 * np.pi) * 0.5 * np.pi + 0.5 * np.pi
        x_ticks = np.arange(x_tick_min, S_max, 0.5 * np.pi)
        x_ticks_labels = [(angle, f"{angle/np.pi:.1f}π") for angle in x_ticks]

        # construct left panel
        left1 = hv.Curve((S, eta_L), label="η=η₁").opts(
            color="grey",
            line_dash="dashed",
            show_legend=True,
            xlim=(S_min, S_max),
            xticks=x_ticks_labels,
        )
        left2 = hv.Curve((S, eta_3_L), label="η³").opts(
            color="k", show_legend=True, xlim=(S_min, S_max), xticks=x_ticks_labels
        )
        left3 = hv.Curve(([S_min, S_max], [0, 0])).opts(
            color="k",
            line_dash="dashed",
            line_width=0.7,
            show_legend=True,
            xlim=(S_min, S_max),
            xticks=x_ticks_labels,
        )
        left = hv.Overlay(left1 * left2 * left3).opts(
            xlabel="phase [rad]",
            ylabel="η [m]",
            legend_position="bottom_right",
            title="sinusoidal",
            aspect=1.3,
            responsive=True,
            show_grid=True,
        )

        # construct middle panel
        middle1 = hv.Curve((S, eta_M), label="η=η₁+η₂").opts(
            color="grey",
            line_dash="dashed",
            show_legend=True,
            xlim=(S_min, S_max),
            xticks=x_ticks_labels,
        )
        middle2 = hv.Curve((S, eta_3_M), label="η³").opts(
            color="k", show_legend=True, xlim=(S_min, S_max), xticks=x_ticks_labels
        )
        middle3 = hv.Curve(([S_min, S_max], [0, 0])).opts(
            color="k",
            line_dash="dashed",
            line_width=0.7,
            show_legend=True,
            xlim=(S_min, S_max),
            xticks=x_ticks_labels,
        )
        middle = hv.Overlay(middle1 * middle2 * middle3).opts(
            xlabel="phase [rad]",
            ylabel="η [m]",
            legend_position="bottom_right",
            title="skewed",
            aspect=1.3,
            responsive=True,
            show_grid=True,
        )

        # construct right panel
        right1 = hv.Curve((S, eta_R), label="η=η₁+η₂").opts(
            color="grey",
            line_dash="dashed",
            show_legend=True,
            xlim=(S_min, S_max),
            xticks=x_ticks_labels,
        )
        right2 = hv.Curve((S, eta_3_R), label="η³").opts(
            color="k", show_legend=True, xlim=(S_min, S_max), xticks=x_ticks_labels
        )
        right3 = hv.Curve(([S_min, S_max], [0, 0])).opts(
            color="k",
            line_dash="dashed",
            line_width=0.7,
            show_legend=True,
            xlim=(S_min, S_max),
            xticks=x_ticks_labels,
        )
        right = hv.Overlay(right1 * right2 * right3).opts(
            xlabel="phase [rad]",
            ylabel="η [m]",
            legend_position="bottom_right",
            title="asymmetric",
            aspect=1.3,
            responsive=True,
            show_grid=True,
        )

        # plot subplots
        filterwarnings("ignore", category=FutureWarning)
        display(left + middle + right)

    graph = ipw.interactive_output(
        update_graph,
        {
            "a1": a1,
            "a2_1": a2_1,
            "a2_2": a2_2,
            "phi1": phi1,
            "phi2_1": phi2_1,
            "phi2_2": phi2_2,
        },
    )

    display(ui, graph)
