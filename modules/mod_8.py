import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
import hvplot.pandas
from bokeh.models.formatters import PrintfTickFormatter


def image_app(images):
    romans = ["I", "II", "III", "IV", "V", "VI", "VII"]
    slider = pn.widgets.DiscreteSlider(
        name="X-position", options=romans, orientation="horizontal"
    )

    text = pn.widgets.StaticText(value=f"...")

    @pn.depends(slider.param.value)
    def image_field(slider_value):
        indices = np.argwhere(np.array(romans) == slider_value)[0][0]

        return pn.pane.Image(images[indices], sizing_mode="scale_width")

    app = pn.Column(
        slider,
        image_field,
    )

    return app


def check_answers(Sres, Sres_term1, Sres_term2, Sres_term3, Sres_term123):
    tp_components = [
        "Residual transport according to Eq. 3",
        "First term in RHS of Eq. 4: u₀ and u_M2",
        "Second term in RHS of Eq. 4: u_M2 and u_M4",
        "Third term in RHS of Eq. 4: u_M2, u_M4 and u_M6",
        "Residual transport according to Eq. 4",
    ]
    python_vars = ["Sres", "Sres_term1", "Sres_term2", "Sres_term3", "Sres_term123"]
    correct = np.array([-7.5, -5.95, -3.39, 1.46, -7.88])
    student = np.array([Sres, Sres_term1, Sres_term2, Sres_term3, Sres_term123])

    for i in range(len(student)):
        if student[i] is not None:
            student[i] = np.round(student[i] / 10**-6, 2)

    df = pd.DataFrame(
        data={
            "Residual transports": tp_components,
            "Python variable": python_vars,
            "Correct [1/10⁶ m³/m/s]": correct,
            "Student [1/10⁶ m³/m/s]": student,
        }
    )

    return df


def figure_u(u0, um2, um4, um6, phi42, phi62, u_3):
    T2 = 12 * 3600 + 25 * 60
    T4 = (12 * 3600 + 25 * 60) / 2
    T6 = (12 * 3600 + 25 * 60) / 3

    omega2 = 2 * np.pi / T2
    omega4 = 2 * np.pi / T4
    omega6 = 2 * np.pi / T6

    t = np.linspace(0, 24 * 3600 + 50 * 60, 200)

    M2 = um2 * np.cos(omega2 * t)
    M4 = um4 * np.cos(omega4 * t - phi42)
    M6 = um6 * np.cos(omega6 * t - phi62)

    u = u0 + M2 + M4 + M6
    u3 = u**3

    u_0 = u0 * np.ones(200)
    df1 = pd.DataFrame(
        {"t [hrs]": t / 3600, "u_0": u_0, "u_M2": M2, "u_M4": M4, "u_M6": M6, "u": u}
    )
    plot1 = df1.hvplot.line(
        x="t [hrs]",
        y=["u_0", "u_M2", "u_M4", "u_M6", "u"],
        value_label="u [m/s]",
        legend="right",
        group_label=" ",
        line_dash=["dashed", "dashed", "dashed", "dashed", "solid"],
        color=["blue", "red", "green", "orange", "grey"],
        responsive=True,
        height=300,
        max_width=800,
        title="Velocity components",
    ).opts(xticks=[0, 3, 6, 9, 12, 15, 18, 21, 24])

    df3 = pd.DataFrame(
        {
            "t [hrs]": t / 3600,
            "u_0": u_0,
            "u_M2": M2,
            "u_M4": M4,
            "u_M6": M6,
            "u": u,
            "u³": u3,
        }
    )
    plot3 = df3.hvplot.line(
        x="t [hrs]",
        y=["u", "u³"],
        value_label="u [m/s] / u³ [m³/s³]",
        legend="right",
        group_label=" ",
        line_dash=["solid", "solid"],
        color=["grey", "purple"],
        responsive=True,
        height=300,
        max_width=800,
        title="Velocity with 3rd power",
    ).opts(xticks=[0, 3, 6, 9, 12, 15, 18, 21, 24])

    if u_3 == 0:
        plot = plot1

    elif u_3 == 1:
        plot = (plot1 + plot3).opts(shared_axes=False).cols(2)

    return plot


def slider_u(u_3):
    # define sliders
    u0_slider = pn.widgets.FloatSlider(
        name="u_0 [m/s]", start=-2, end=2, step=0.01, value=0
    )
    um2_slider = pn.widgets.FloatSlider(
        name="u_M2 [m/s]", start=0, end=3, step=0.01, value=1
    )
    um4_slider = pn.widgets.FloatSlider(
        name="u_M4 [m/s]", start=0, end=3, step=0.01, value=0.2
    )
    um6_slider = pn.widgets.FloatSlider(
        name="u_M6 [m/s]", start=0, end=3, step=0.01, value=0.2
    )
    phi42_slider = pn.widgets.FloatSlider(
        name="φ₄₂ [rad]", start=-6.28, end=6.28, step=0.01, value=0
    )
    phi62_slider = pn.widgets.FloatSlider(
        name="φ₆₂ [rad]", start=-6.28, end=6.28, step=0.01, value=0
    )

    u_3 = pn.widgets.IntInput(value=u_3)

    @pn.depends(
        u0_slider.param.value,
        um2_slider.param.value,
        um4_slider.param.value,
        um6_slider.param.value,
        phi42_slider.param.value,
        phi62_slider.param.value,
        u_3.param.value,
    )
    def plot(u0, um2, um4, um6, phi42, phi62, u_3):
        return figure_u(u0, um2, um4, um6, phi42, phi62, u_3)

    app = pn.Column(
        pn.Row(
            pn.Column(
                u0_slider,
                phi42_slider,
                phi62_slider,
            ),
            pn.Column(
                um2_slider,
                um4_slider,
                um6_slider,
            ),
        ),
        pn.Row(plot, sizing_mode="stretch_width"),
        width_policy="max",
    )

    return app


def plot_S(total_transport):
    t = np.linspace(0, 2 * 24 * 3600 + 2 * 50 * 60, 250)

    try:
        total_transport
    except UnboundLocalError:
        total_transport = np.zeros(t.shape)

    c = 10**-4

    u0 = -3 / 100
    um2 = 115 / 100
    um4 = 10 / 100
    um6 = 9 / 100

    phi42 = 250 / 360 * (2 * np.pi)
    phi62 = 230 / 360 * (2 * np.pi)

    T2 = 12 * 3600 + 25 * 60
    T4 = (12 * 3600 + 25 * 60) / 2
    T6 = (12 * 3600 + 25 * 60) / 3

    omega2 = 2 * np.pi / T2
    omega4 = 2 * np.pi / T4
    omega6 = 2 * np.pi / T6

    M2 = um2 * np.cos(omega2 * t)
    M4 = um4 * np.cos(omega4 * t - phi42)
    M6 = um6 * np.cos(omega6 * t - phi62)

    S = c * (u0 + M2 + M4 + M6) ** 3

    df = pd.DataFrame({"t [hrs]": t / 3600, "Correct": S, "Student": total_transport})

    plot = df.hvplot.line(
        x="t [hrs]",
        y=["Correct", "Student"],
        value_label="S [m³/m/s]",
        legend="right",
        group_label="Answers",
        line_dash=["solid", "dashed"],
        title="Transport",
    )

    return plot


def plot_u_int(u):
    t = np.linspace(-1 * (12 * 3600 + 25 * 60), 2 * (12 * 3600 + 25 * 60), 1000)
    thrs = t / 3600

    try:
        u
    except UnboundLocalError:
        u = np.zeros(t.shape)

    u0 = 0 / 100
    um2 = 100 / 100
    um4 = 25 / 100

    phi42 = 270 / 360 * (2 * np.pi)

    T2 = 12 * 3600 + 25 * 60
    T4 = (12 * 3600 + 25 * 60) / 2

    omega2 = 2 * np.pi / T2
    omega4 = 2 * np.pi / T4

    M2 = um2 * np.cos(omega2 * t)
    M4 = um4 * np.cos(omega4 * t - phi42)

    u_correct = u0 + M2 + M4

    plot1 = hv.Curve(zip(thrs, u_correct), label="Velocity (correct)")
    plot2 = hv.Curve(zip(thrs, u), label="Velocity (student)").opts(line_dash="dashed")

    plot = (plot1 * plot2).opts(
        title="Tidal velocity",
        ylabel="Velocity [m/s]",
        xlabel="Time [hrs]",
        height=300,
        show_grid=True,
        responsive=True,
        legend_position="right",
        max_width=800,
    )

    return plot


def plot_c_eq(c_eq, beta, n):
    t = np.linspace(-1 * (12 * 3600 + 25 * 60), 2 * (12 * 3600 + 25 * 60), 1000)
    thrs = t / 3600

    u0 = 0 / 100
    um2 = 100 / 100
    um4 = 25 / 100

    phi42 = 270 / 360 * (2 * np.pi)

    T2 = 12 * 3600 + 25 * 60
    T4 = (12 * 3600 + 25 * 60) / 2

    omega2 = 2 * np.pi / T2
    omega4 = 2 * np.pi / T4

    M2 = um2 * np.cos(omega2 * t)
    M4 = um4 * np.cos(omega4 * t - phi42)

    u_correct = u0 + M2 + M4

    c_eq_correct = beta * np.abs(u_correct) ** (n - 1)

    plot1 = hv.Curve(zip(thrs, u_correct), label="Tidal velocity")
    plot2 = hv.Curve(
        zip(thrs, c_eq_correct / beta), label="Equilibrium concentration (correct)"
    )
    plot3 = hv.Curve(
        zip(thrs, c_eq / beta), label="Equilibrium concentration (student)"
    ).opts(line_dash="dashed", color="green")

    plot = (plot1 * plot2 * plot3).opts(
        title="Tidal velocity and equilibrium concentration",
        ylabel="Velocity [m/s]\nConcentration [β m³/m³]",
        xlabel="Time [hrs]",
        height=300,
        show_grid=True,
        responsive=True,
        legend_position="right",
        max_width=800,
    )

    return plot


def plot_c(student_c, beta, n):
    t = np.linspace(-1 * (12 * 3600 + 25 * 60), 2 * (12 * 3600 + 25 * 60), 1000)
    thrs = t / 3600

    u0 = 0 / 100
    um2 = 100 / 100
    um4 = 25 / 100

    phi42 = 270 / 360 * (2 * np.pi)

    T2 = 12 * 3600 + 25 * 60
    T4 = (12 * 3600 + 25 * 60) / 2

    omega2 = 2 * np.pi / T2
    omega4 = 2 * np.pi / T4

    M2 = um2 * np.cos(omega2 * t)
    M4 = um4 * np.cos(omega4 * t - phi42)

    u_correct = u0 + M2 + M4

    c_eq_correct = beta * np.abs(u_correct) ** (n - 1)
    c_correct = np.zeros(c_eq_correct.shape)

    dt = t[1] - t[0]
    Tsed = 0.2 * T2  # as done in book

    c_correct[0] = c_eq_correct[0]

    for n in range(len(student_c) - 1):
        c_correct[n + 1] = c_correct[n] + dt * 1 / Tsed * (
            c_eq_correct[n] - c_correct[n]
        )

    plot1 = hv.Curve(zip(thrs, c_eq_correct / beta), label="Equilibrium concentration")
    plot2 = hv.Curve(zip(thrs, c_correct / beta), label="Concentration (correct)")
    plot3 = hv.Curve(zip(thrs, student_c / beta), label="Concentration (student)").opts(
        line_dash="dashed", color="green"
    )

    plot = (plot1 * plot2 * plot3).opts(
        title="Actual and equilibrium concentration",
        ylabel="Concentration [β m³/m³]",
        xlabel="Time [hrs]",
        height=300,
        show_grid=True,
        responsive=True,
        legend_position="right",
        max_width=800,
    )

    return plot


def plot_fig932():
    t = np.linspace(-1 * (12 * 3600 + 25 * 60), 2 * (12 * 3600 + 25 * 60), 1000)
    thrs = t / 3600

    u0 = 0 / 100
    um2 = 100 / 100
    um4 = 25 / 100

    phi42 = 270 / 360 * (2 * np.pi)

    T2 = 12 * 3600 + 25 * 60
    T4 = (12 * 3600 + 25 * 60) / 2

    omega2 = 2 * np.pi / T2
    omega4 = 2 * np.pi / T4

    M2 = um2 * np.cos(omega2 * t)
    M4 = um4 * np.cos(omega4 * t - phi42)

    u_correct = u0 + M2 + M4

    beta = 10**-4
    n = 5

    c_eq_correct = beta * np.abs(u_correct) ** (n - 1)
    c_correct = np.zeros(c_eq_correct.shape)

    dt = t[1] - t[0]
    Tsed = 0.2 * T2  # as done in book

    for n in range(len(c_correct) - 1):
        c_correct[n + 1] = c_correct[n] + dt * 1 / Tsed * (
            c_eq_correct[n] - c_correct[n]
        )

    uc = u_correct * c_correct

    plot0 = hv.HLine(0).opts(color="black")
    plot1 = hv.Curve(zip(thrs, u_correct), label="Velocity")
    plot2 = hv.Curve(zip(thrs, c_eq_correct), label="Equilibrium concentration")
    plot2 = hv.Curve(zip(thrs, c_eq_correct / beta), label="Equilibrium concentration")

    plot3 = hv.Curve(zip(thrs, c_correct / beta), label="Concentration")
    plot4 = hv.Curve(zip(thrs, uc / beta), label="Flux")

    plot = (plot0 * plot1 * plot2 * plot3 * plot4).opts(
        title="Recreated Figure 9.32",
        ylabel="Velocity [m/s]\nConcentration [β m³/m³]\nFlux[β m/s]",
        xlabel="ωt [rad]",
        height=300,
        max_width=800,
        show_grid=True,
        responsive=True,
        legend_position="right",
        xlim=(
            3 / 2 * np.pi / (2 * np.pi) * T2 / 3600,
            7 / 2 * np.pi / (2 * np.pi) * T2 / 3600,
        ),
        xticks=[
            (-4 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "-2 π"),
            (-3 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "-3/2 π"),
            (-2 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "-π"),
            (-1 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "-1/2 π"),
            (0 * np.pi / (2 * np.pi) * T2 / 3600, "0"),
            (1 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "1/2 π"),
            (2 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "π"),
            (3 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "3/2 π"),
            (4 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "2 π"),
            (5 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "5/2 π"),
            (6 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "3 π"),
            (7 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "7/2 π"),
            (8 / 2 * np.pi / (2 * np.pi) * T2 / 3600, "4 π"),
        ],
    )

    return plot


def velocity_concentration_app():
    u0_slider = pn.widgets.EditableFloatSlider(
        name="u0 [cm/s]", start=-100, end=100, step=1, value=0, orientation="horizontal"
    )
    um2_slider = pn.widgets.EditableFloatSlider(
        name="ûₘ₂ [cm/s]", start=0, end=100, step=1, value=40, orientation="horizontal"
    )
    um4_slider = pn.widgets.EditableFloatSlider(
        name="ûₘ₄ [cm/s]", start=0, end=100, step=1, value=10, orientation="horizontal"
    )
    phi42_slider = pn.widgets.EditableFloatSlider(
        name="φ₄₂ [rad]",
        start=-6.28,
        end=6.28,
        step=0.01,
        value=3.14 + 1.57,
        orientation="horizontal",
        format=PrintfTickFormatter(format="%.2f"),
    )
    beta_slider = pn.widgets.EditableFloatSlider(
        name="β",
        start=10**-5,
        end=10**-3,
        step=10**-5,
        value=10**-4,
        orientation="horizontal",
        format=PrintfTickFormatter(format="%.5f"),
    )
    n_slider = pn.widgets.EditableIntSlider(
        name="n [-]", start=1, end=5, step=1, value=5, orientation="horizontal"
    )
    Tsed_slider = pn.widgets.EditableFloatSlider(
        name="Tˢᵉᵈ [s]",
        start=0,
        end=22350,
        step=200,
        value=2.8 * 3600,
        orientation="horizontal",
    )
    c0_slider = pn.widgets.EditableFloatSlider(
        name="c₀ [1/10⁵ m³/m³]",
        start=0,
        end=100,
        step=0.01,
        value=0,
        orientation="horizontal",
    )

    @pn.depends(
        u0_slider.param.value,
        um2_slider.param.value,
        um4_slider.param.value,
        phi42_slider.param.value,
        Tsed_slider.param.value,
        beta_slider.param.value,
        n_slider.param.value,
        c0_slider.param.value,
    )
    def velocity_concentration_plots(u0, um2, um4, phi42, Tsed, beta, n, c0):
        # define periods and compute radial frequency
        n_M2_periods = 3
        t_seconds = np.linspace(0, n_M2_periods * (12 * 3600 + 25 * 60), 10000)
        t_hours = t_seconds / 3600
        dt = t_seconds[1] - t_seconds[0]

        T2 = 12 * 3600 + 25 * 60
        T4 = (12 * 3600 + 25 * 60) / 2

        omega2 = 2 * np.pi / T2
        omega4 = 2 * np.pi / T4

        # compute velocity components
        M2 = um2 * np.cos(omega2 * t_seconds)
        M4 = um4 * np.cos(omega4 * t_seconds - phi42)

        # total velocity
        u = (u0 + M2 + M4) / 100

        # equilibrium concentration
        c_eq = beta * np.abs(u) ** (n - 1)

        # update beta name in slider
        superscript_dic = {
            -5: "⁻⁵",
            -4: "⁻⁴",
            -3: "⁻³",
            -2: "⁻²",
            -1: "⁻¹",
            0: "⁰",
            1: "¹",
            2: "²",
            3: "³",
            4: "⁴",
            5: "⁵",
        }
        beta_slider.name = (
            f"β [m{superscript_dic[int(1-n)]} s{superscript_dic[int(n-1)]}]"
        )

        # actual concentration
        c = np.zeros(c_eq.shape)
        c[0] = c0 / 10**5
        for i in range(len(c) - 1):
            c[i + 1] = c[i] + dt / Tsed * (c_eq[i] - c[i])
        if Tsed == 0:
            c = c_eq

        # flux
        uc = u * c
        uc_eq = u * c_eq

        # mask out pos/neg velocities
        uc_flo = uc * (uc >= 0)
        uc_ebb = uc * (uc < 0)

        # mask for final period
        t_vlines = [i * (12 * 3600 + 25 * 60) / 3600 for i in range(1, n_M2_periods)]
        t_boundary = t_vlines[-1] * 3600
        mask = t_seconds > t_boundary

        # area in graph
        A1 = np.abs(np.trapz(uc_flo[mask], t_seconds[mask]))
        A2 = np.abs(np.trapz(uc_ebb[mask], t_seconds[mask]))

        # create plots
        plot1 = (
            (
                hv.Curve(zip(t_hours, u), label="Velocity").opts(color="red")
                * hv.Curve([], [], label=f"uᵐⁱⁿ = {min(u):.2f} [m/s]")
                * hv.Curve([], [], label=f"uᵐᵃˣ = {max(u):.2f} [m/s]")
                * hv.HLine(0).opts(color="black")
                * hv.VLines(
                    [i * (12 * 3600 + 25 * 60) / 3600 for i in range(1, n_M2_periods)]
                ).opts(color="black", line_dash="dashed")
            )
            .opts(
                title="Tidal velocity",
                ylabel="Velocity [m/s]",
                xlabel="Time [hrs]",
                height=200,
                show_grid=True,
                responsive=True,
                max_width=800,
            )
            .opts(legend_position="right")
        )

        plot2 = (
            hv.Curve(
                zip(t_hours, c_eq * 10**5), label="Equilibrium concentration"
            ).opts(color="#FFCD00")
            * hv.Curve(zip(t_hours, c * 10**5), label="Concentration").opts(
                color="green"
            )
            * hv.HLine(0).opts(color="black")
            * hv.VLines(
                [i * (12 * 3600 + 25 * 60) / 3600 for i in range(1, n_M2_periods)]
            ).opts(color="black", line_dash="dashed")
        ).opts(
            title="Concentration",
            ylabel="Concentration [1/10⁵ m³/m³]",
            xlabel="Time [hrs]",
            height=200,
            show_grid=True,
            responsive=True,
            max_width=800,
            legend_position="right",
        )

        plot3 = (
            hv.Curve(zip(t_hours, uc * 10**5), label="Sediment flux").opts(
                color="blue"
            )
            * hv.Curve(
                zip(t_hours, uc_eq * 10**5), label="Equilibrium sediment flux"
            ).opts(color="magenta")
            * hv.Area(zip(t_hours[mask], uc_flo[mask] * 10**5), label=f"A₁").opts(
                color="#45b6fe"
            )
            * hv.Area(zip(t_hours[mask], uc_ebb[mask] * 10**5), label=f"A₂").opts(
                color="#daf0ff"
            )
            * hv.VLines(
                [i * (12 * 3600 + 25 * 60) / 3600 for i in range(1, n_M2_periods)]
            ).opts(color="black", line_dash="dashed")
            * hv.Curve([], [], label=f"A₁ = {A1/A2:.2f} A₂")
            * hv.HLine(0).opts(color="black")
        ).opts(
            title="Sediment flux",
            ylabel="u*c [1/10⁵ m/s]",
            xlabel="Time [hrs]",
            height=200,
            show_grid=True,
            responsive=True,
            max_width=800,
            legend_position="right",
        )

        layout = (plot1 + plot2 + plot3).cols(1).opts(shared_axes=False)

        return layout

    app = pn.Column(
        pn.Row(
            pn.Column(
                u0_slider,
                um2_slider,
                um4_slider,
                phi42_slider,
            ),
            pn.Column(
                beta_slider,
                n_slider,
                Tsed_slider,
                c0_slider,
            ),
            sizing_mode="stretch_width",
            max_width=400,
        ),
        pn.Row(velocity_concentration_plots),
    )

    return app


def plot_VcVod():
    # Show figures 9.35 and 9.36.
    P = np.logspace(5, 11, 100)
    Cv = 65 * 10**-6
    Cod = 65.7 * 10**-4

    Vod = Cod * P**1.23  # eq. 9.3
    Vc = Cv * P**1.5  # eq. 9.17

    curve1 = hv.Curve(zip(P, Vod), label="Vᵒᵈ (outer delta)")
    curve2 = hv.Curve(zip(P, Vc), label="Vᶜ (channels)")

    plot = (curve1 * curve2).opts(
        title="Volume of channels and outer delta",
        show_grid=True,
        logx=True,
        logy=True,
        xlim=(10 * 10**6, 5000 * 10**6),
        ylim=(2 * 10**6, 5000 * 10**6),
        xlabel="P [10⁶ m³]",
        ylabel="V [10⁶ m³]",
        width=600,
        height=600,
        legend_position="right",
    )

    return plot


def intermezzo_app():
    axes_type_dropdown = pn.widgets.Select(
        name="Axes type", options=["linear", "loglog"], value="loglog"
    )
    plot_range_x_slider = pn.widgets.RangeSlider(
        name="Horizontal axis range", start=1, end=10000, step=1, value=(10, 5000)
    )
    plot_range_y_slider = pn.widgets.RangeSlider(
        name="Vertical axis range (y)", start=1, end=10000, step=1, value=(2, 5000)
    )
    Cv_slider = pn.widgets.EditableFloatSlider(
        name="Cᵛ [10⁻⁶ m^{3-3n}]",
        start=1,
        end=100,
        step=0.1,
        value=65,
        format=PrintfTickFormatter(format="%.1f"),
    )
    Cod_slider = pn.widgets.EditableFloatSlider(
        name="Cᵒᵈ [10⁻⁴ m^{3-3n}]",
        start=1,
        end=100,
        step=0.1,
        value=65.7,
        format=PrintfTickFormatter(format="%.1f"),
    )
    power_c_slider = pn.widgets.EditableFloatSlider(
        name="Power n (Equation 9.17)",
        start=0.1,
        end=2,
        step=0.01,
        value=1.5,
        format=PrintfTickFormatter(format="%.2f"),
    )
    power_od_slider = pn.widgets.EditableFloatSlider(
        name="Power n (Equation 9.3)",
        start=0.1,
        end=2,
        step=0.01,
        value=1.23,
        format=PrintfTickFormatter(format="%.2f"),
    )

    @pn.depends(
        axes_type_dropdown.param.value,
        plot_range_x_slider.param.value,
        plot_range_y_slider.param.value,
        Cv_slider.param.value,
        Cod_slider.param.value,
        power_c_slider.param.value,
        power_od_slider.param.value,
    )
    def plot_VcVod_interactive(
        axes_type, plot_range_x, plot_range_y, Cv, Cod, power_c, power_od
    ):
        P = np.logspace(5, 11, 100)

        Vc = (Cv * 10**-6) * P**power_c
        Vod = (Cod * 10**-4) * P**power_od

        Cv_units = 3 - 3 * power_c_slider.value
        Cv_slider.name = f"Cᵛ [10⁻⁶ m^{Cv_units:,.2f}]"

        Cod_units = 3 - 3 * power_od_slider.value
        Cod_slider.name = f"Cᵒᵈ [10⁻⁴ m^{Cod_units:,.2f}]"

        curve1 = hv.Curve(zip(P / 10**6, Vod / 10**6), label="Vᵒᵈ (outer delta)")
        curve2 = hv.Curve(zip(P / 10**6, Vc / 10**6), label="Vᶜ (channels)")

        plot = (curve1 * curve2).opts(
            title="Volume of channels and outer delta",
            show_grid=True,
            logx=axes_type == "loglog",
            logy=axes_type == "loglog",
            xlim=plot_range_x,
            ylim=plot_range_y,
            xlabel="P [10⁶ m³]",
            ylabel="V [10⁶ m³]",
            width=600,
            height=600,
            legend_position="right",
        )

        return plot

    app = pn.Column(
        pn.Row(
            pn.Column(Cod_slider, power_od_slider, Cv_slider, power_c_slider),
            pn.Column(axes_type_dropdown, plot_range_x_slider, plot_range_y_slider),
        ),
        plot_VcVod_interactive,
        sizing_mode="stretch_width",
    )

    return app


def plot_fig935(
    P_before,
    P_after,
    dV_c,
    axes_type="loglog",
    plot_range=[[10, 5000], [2, 5000]],
    coefficients=[65 * 10**-6, 65.7 * 10**-4],
    powers=[1.5, 1.23],
    title="Correct plot",
    height=None,
    width=None,
):
    P = np.logspace(2, 20, 100)

    power_c, power_od = powers[0], powers[1]
    Cv, Cod = coefficients[0], coefficients[1]

    xmin, xmax = plot_range[0]
    ymin, ymax = plot_range[1]

    Vc = Cv * P**power_c
    Vod = Cod * P**power_od

    if axes_type == "loglog":
        loglog = True
    elif axes_type == "linear":
        loglog = False
    else:
        raise ValueError("Use either loglog or linear for the axes_type")

    curve1 = hv.Curve(zip(P / 10**6, Vod / 10**6), label="Vᵒᵈ").opts(
        color="#30a2da"
    )
    curve2 = hv.Curve(zip(P / 10**6, Vc / 10**6), label="Vᶜ").opts(color="#fc4f30")

    plot = curve1 * curve2

    dP = P_before - P_after

    V_c_before_correct = Cv * P_before**power_c
    V_c_new_correct = V_c_before_correct - dV_c
    V_c_after_correct = Cv * P_after**power_c

    V_od_before_correct = Cod * P_before**power_od
    V_od_after_correct = Cod * P_after**power_od

    a_correct = V_c_before_correct - dV_c - V_c_after_correct
    b_correct = V_od_before_correct - V_od_after_correct

    dP_curve = hv.Curve(
        [
            (P_before / 10**6, V_c_before_correct / 10**6),
            (P_after / 10**6, V_c_before_correct / 10**6),
        ]
    ).opts(color="black", apply_ranges=True, line_dash="solid", line_width=1) * hv.Text(
        (P_before + P_after) / 2 / 10**6,
        V_c_before_correct / 10**6 + 300,
        "ΔP",
        fontsize=9,
    )

    dV_c_curve = hv.Curve(
        [
            (P_after / 10**6, V_c_before_correct / 10**6),
            (P_after / 10**6, (V_c_before_correct - dV_c) / 10**6),
        ]
    ).opts(color="black", apply_ranges=True, line_dash="solid", line_width=1) * hv.Text(
        (P_after / 10**6) - 50,
        (V_c_before_correct + V_c_before_correct - dV_c) / 2 / 10**6,
        "ΔVᶜ",
        fontsize=9,
    )

    connection_curve = hv.Curve(
        [
            [P_before / 10**6, V_c_before_correct / 10**6],
            [(P_before - dP) / 10**6, (V_c_before_correct - dV_c) / 10**6],
        ]
    ).opts(color="black", line_dash="dashed")

    a_curves = (
        hv.Curve(
            [
                [10**-5, (V_c_before_correct - dV_c) / 10**6],
                [P_after / 10**6, (V_c_before_correct - dV_c) / 10**6],
            ]
        ).opts(color="#fc4f30", line_dash="dashed")
        * hv.Curve(
            [
                [10**-5, V_c_after_correct / 10**6],
                [P_after / 10**6, V_c_after_correct / 10**6],
            ]
        ).opts(color="#fc4f30", line_dash="dashed")
        * hv.Curve(
            [
                [10**2 + 10, V_c_after_correct / 10**6],
                [10**2 + 10, (V_c_before_correct - dV_c) / 10**6],
            ]
        ).opts(color="black", line_width=1)
        * hv.Text(
            10**2,
            (V_c_after_correct / 10**6 + (V_c_before_correct - dV_c) / 10**6) / 2,
            text="a",
            fontsize=9,
        )
    )

    b_curves = (
        hv.Curve(
            [
                [P_after / 10**6, V_od_after_correct / 10**6],
                [10**6, V_od_after_correct / 10**6],
            ]
        ).opts(color="#30a2da", line_dash="dashed")
        * hv.Curve(
            [
                [P_before / 10**6, V_od_before_correct / 10**6],
                [10**6, V_od_before_correct / 10**6],
            ]
        ).opts(color="#30a2da", line_dash="dashed")
        * hv.Curve(
            [
                [10**3, V_od_after_correct / 10**6],
                [10**3, (V_od_before_correct) / 10**6],
            ]
        ).opts(color="black", line_width=1)
        * hv.Text(
            10**3 + 70,
            (V_od_after_correct / 10**6 + V_od_before_correct / 10**6) / 2,
            text="b",
            fontsize=9,
        )
    )

    points_old = hv.Points(
        [
            [P_before / 10**6, V_c_before_correct / 10**6],
            [P_before / 10**6, V_od_before_correct / 10**6],
        ]
    ).opts(size=10, marker="x", color="white", line_color="black")
    points_bet = hv.Points([[P_after / 10**6, V_c_new_correct / 10**6]]).opts(
        size=10, marker="circle", color="black", line_color="black"
    )
    points_new = hv.Points(
        [
            [P_after / 10**6, V_c_after_correct / 10**6],
            [P_after / 10**6, V_od_after_correct / 10**6],
        ]
    ).opts(size=10, marker="circle", color="white", line_color="black")

    points = points_old * points_bet * points_new

    plot_correct = (
        plot * dP_curve * dV_c_curve * connection_curve * a_curves * b_curves * points
    ).opts(
        show_grid=True,
        logx=loglog,
        logy=loglog,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        title=title,
        xlabel="P [10⁶ m³]",
        ylabel="V [10⁶ m³]",
        legend_position="bottom_right",
    )

    if width:
        plot_correct = plot_correct.opts(width=width)
    if height:
        plot_correct = plot_correct.opts(height=height)

    return (
        plot_correct,
        V_c_before_correct,
        V_c_after_correct,
        V_od_before_correct,
        V_od_after_correct,
        a_correct,
        b_correct,
    )


def student_plot_fig935(
    P_before,
    P_after,
    dV_c,
    V_c_before,
    V_c_after,
    V_od_before,
    V_od_after,
    a,
    b,
    Cv=65 * 10**-6,
    Cod=65.7 * 10**-4,
):
    P = np.logspace(2, 20, 100)

    Vod = Cod * P**1.23
    Vc = Cv * P**1.5

    curve1 = hv.Curve(zip(P / 10**6, Vod / 10**6), label="Vᵒᵈ").opts(
        color="#30a2da"
    )
    curve2 = hv.Curve(zip(P / 10**6, Vc / 10**6), label="Vᶜ").opts(color="#fc4f30")

    plot = curve1 * curve2

    dP = P_before - P_after

    dP_curve = hv.Curve(
        [
            (P_before / 10**6, V_c_before / 10**6),
            (P_after / 10**6, V_c_before / 10**6),
        ]
    ).opts(color="black", apply_ranges=True, line_dash="solid", line_width=1) * hv.Text(
        (P_before + P_after) / 2 / 10**6, V_c_before / 10**6 + 300, "ΔP", fontsize=9
    )

    dV_c_curve = hv.Curve(
        [
            (P_after / 10**6, V_c_before / 10**6),
            (P_after / 10**6, (V_c_before - dV_c) / 10**6),
        ]
    ).opts(color="black", apply_ranges=True, line_dash="solid", line_width=1) * hv.Text(
        (P_after / 10**6) - 50,
        (V_c_before + V_c_before - dV_c) / 2 / 10**6,
        "ΔVᶜ",
        fontsize=9,
    )

    connection_curve = hv.Curve(
        [
            [P_before / 10**6, V_c_before / 10**6],
            [(P_before - dP) / 10**6, (V_c_before - dV_c) / 10**6],
        ]
    ).opts(color="black", line_dash="dashed")

    a_curves = (
        hv.Curve(
            [
                [10**-5, (V_c_before - dV_c) / 10**6],
                [P_after / 10**6, (V_c_before - dV_c) / 10**6],
            ]
        ).opts(color="#fc4f30", line_dash="dashed")
        * hv.Curve(
            [[10**-5, V_c_after / 10**6], [P_after / 10**6, V_c_after / 10**6]]
        ).opts(color="#fc4f30", line_dash="dashed")
        * hv.Curve(
            [
                [10**2 + 10, V_c_after / 10**6],
                [10**2 + 10, (V_c_before - dV_c) / 10**6],
            ]
        ).opts(color="black", line_width=1)
        * hv.Text(
            10**2,
            (V_c_after / 10**6 + (V_c_before - dV_c) / 10**6) / 2,
            text="a",
            fontsize=9,
        )
    )

    b_curves = (
        hv.Curve(
            [[P_after / 10**6, V_od_after / 10**6], [10**6, V_od_after / 10**6]]
        ).opts(color="#30a2da", line_dash="dashed")
        * hv.Curve(
            [
                [P_before / 10**6, V_od_before / 10**6],
                [10**6, V_od_before / 10**6],
            ]
        ).opts(color="#30a2da", line_dash="dashed")
        * hv.Curve(
            [[10**3, V_od_after / 10**6], [10**3, (V_od_before) / 10**6]]
        ).opts(color="black", line_width=1)
        * hv.Text(
            10**3 + 70,
            (V_od_after / 10**6 + V_od_before / 10**6) / 2,
            text="b",
            fontsize=9,
        )
    )

    points_old = hv.Points(
        [
            [P_before / 10**6, V_c_before / 10**6],
            [P_before / 10**6, V_od_before / 10**6],
        ]
    ).opts(size=10, marker="x", color="black", line_color="black")
    points_bet = hv.Points([[P_after / 10**6, (V_c_before - dV_c) / 10**6]]).opts(
        size=10, marker="circle", color="black", line_color="black"
    )
    points_new = hv.Points(
        [
            [P_after / 10**6, V_c_after / 10**6],
            [P_after / 10**6, V_od_after / 10**6],
        ]
    ).opts(size=10, marker="circle", color="white", line_color="black")

    points = points_old * points_bet * points_new

    student_plot = (
        plot * dP_curve * dV_c_curve * connection_curve * a_curves * b_curves * points
    ).opts(
        show_grid=True,
        logx=True,
        logy=True,
        xlim=(10, 5000),
        ylim=(2, 5000),
        title="Student plot",
        xlabel="P [10⁶ m³]",
        ylabel="V [10⁶ m³]",
        legend_position="bottom_right",
    )

    return student_plot


def intervention(V_c_before, V_od_before, V_c_after, V_od_after, a, b, i):
    P_before = [
        600 * 10**6,
        600 * 10**6,
        600 * 10**6,
        500 * 10**6,
        750 * 10**6,
    ]
    P_after = [
        300 * 10**6,
        300 * 10**6,
        500 * 10**6,
        250 * 10**6,
        750 * 10**6,
    ]
    dV_c = [300 * 10**6, 470 * 10**6, 300 * 10**6, 0 * 10**6, -200 * 10**6]

    (
        correct_plot,
        V_c_before_correct,
        V_c_after_correct,
        V_od_before_correct,
        V_od_after_correct,
        a_correct,
        b_correct,
    ) = plot_fig935(P_before[i], P_after[i], dV_c[i])

    student_plot = student_plot_fig935(
        P_before[i],
        P_after[i],
        dV_c[i],
        V_c_before,
        V_c_after,
        V_od_before,
        V_od_after,
        a,
        b,
    )

    # create table from all provided values
    df = pd.DataFrame(
        data={
            "Parameter": [
                "Prism before",
                "Prism after",
                "Channel reduction ΔVᶜ",
                "Vᶜ before",
                "Vᶜ after",
                "Demand of channels a",
                "Vᵒᵈ before",
                "Vᵒᵈ after",
                "Surplus of outer delta b",
                "Demand from outside a-b",
            ],
            "Correct [10⁶ m³]": np.int32(
                np.array(
                    [
                        P_before[i],
                        P_after[i],
                        dV_c[i],
                        V_c_before_correct,
                        V_c_after_correct,
                        a_correct,
                        V_od_before_correct,
                        V_od_after_correct,
                        b_correct,
                        a_correct - b_correct,
                    ]
                )
                / 10**6
            ),
            "Student [10⁶ m³]": np.int32(
                np.array(
                    [
                        P_before[i],
                        P_after[i],
                        dV_c[i],
                        V_c_before,
                        V_c_after,
                        a,
                        V_od_before,
                        V_od_after,
                        b,
                        a - b,
                    ]
                )
                / 10**6
            ),
        }
    )

    display(df)

    # create text to print which values are correct
    incorrect_params = "The following parameters are incorrect:"

    if not round(V_c_before / 10**6) == round(V_c_before_correct / 10**6):
        incorrect_params += "\n V_c_before"
    if not round(V_c_after / 10**6) == round(V_c_after_correct / 10**6):
        incorrect_params += "\n V_c_after"
    if not round(V_od_before / 10**6) == round(V_od_before_correct / 10**6):
        incorrect_params += "\n V_od_before"
    if not round(V_od_after / 10**6) == round(V_od_after_correct / 10**6):
        incorrect_params += "\n V_od_after"
    if not round(a / 10**6) == round(a_correct / 10**6):
        incorrect_params += "\n a"
    if not round(b / 10**6) == round(b_correct / 10**6):
        incorrect_params += "\n b"

    if not incorrect_params == "The following parameters are incorrect:":
        incorrect_params += (
            "\n\nHave you included the 10^6 in your definition of P and V?"
        )
        print(incorrect_params)
    else:
        print("Your answer is correct!")

    plot = (
        correct_plot.opts(
            title=f"Correct plot (closure {int(i+1)})",
            height=400,
            responsive=True,
            max_width=500,
        )
        + student_plot.opts(
            title=f"Student plot (closure {int(i+1)})",
            height=400,
            responsive=True,
            max_width=500,
        )
    ).opts(
        shared_axes=True,
        # xlim=(10**1, 5*10**3),
        # ylim=(2*10**0, 5*10**3)
    )

    return plot


def fig935_app():
    # create sliders
    P_before_slider = pn.widgets.EditableFloatSlider(
        name="Pᵇᵉᶠᵒʳᵉ [10⁶ m³]",
        start=1,
        end=1000,
        step=1,
        value=600,
        orientation="horizontal",
    )
    P_after_slider = pn.widgets.EditableFloatSlider(
        name="Pᵃᶠᵗᵉʳ [10⁶ m³]",
        start=1,
        end=1000,
        step=1,
        value=300,
        orientation="horizontal",
    )
    Delta_Vc_slider = pn.widgets.EditableFloatSlider(
        name="ΔVᶜ [10⁶ m³]",
        start=-1000,
        end=1000,
        step=1,
        value=300,
        orientation="horizontal",
    )

    # create table widget
    df = pd.DataFrame(columns=["Parameters", "Values [10⁶ m³]"], dtype="float")
    table_widget = pn.widgets.Tabulator(
        name="Values corresponding to intervention:", value=df
    )

    @pn.depends(
        P_before_slider.param.value,
        P_after_slider.param.value,
        Delta_Vc_slider.param.value,
    )
    def plot_fig935(P_before, P_after, dV_c):
        # define constants
        Cv = 65 * 10**-6
        Cod = 65.7 * 10**-4

        power_c = 1.5
        power_od = 1.23

        # convert values
        P_before, P_after, dV_c = P_before * 10**6, P_after * 10**6, dV_c * 10**6

        # compute table 9.6
        V_c_before = Cv * P_before**power_c
        V_c_after = Cv * P_after**power_c

        V_od_before = Cod * P_before**power_od
        V_od_after = Cod * P_after**power_od

        a = V_c_before - dV_c - V_c_after
        b = V_od_before - V_od_after

        # add values to table
        data = {
            "Parameters": [
                "Prism before",
                "Prism after",
                "Channel reduction ΔVᶜ",
                "Vᶜ before",
                "Vᶜ after",
                "Demand of channels a",
                "Vᵒᵈ before",
                "Vᵒᵈ after",
                "Surplus of outer delta b",
                "Demand from outside a-b",
            ],
            "Values [10⁶ m³]": np.int32(
                np.array(
                    [
                        P_before,
                        P_after,
                        dV_c,
                        V_c_before,
                        V_c_after,
                        a,
                        V_od_before,
                        V_od_after,
                        b,
                        (a - b),
                    ]
                )
                / 10**6
            ),
        }
        table_widget.value = pd.DataFrame(data=data)

        # set new minimum of P_before_slider based on the minimum Vc_before (which has to be larger than Delta_Vc)
        # in this reverse computation, Pmin can be complex, so we have to check for TypeError
        Pmin = (dV_c / Cv) ** (1 / power_c) / 10**6
        try:
            P_before_slider.start = max(np.int32(Pmin + 1), 1)
        except TypeError:
            P_before_slider.start = 1

        # set new maximum of Delta_Vc_slider based on Vc_before
        Delta_Vc_slider.end = min(np.int32(V_c_before / 10**6 - 1), 1000)

        # plotting variables
        P = np.logspace(2, 20, 100)
        Vc = Cv * P**power_c
        Vod = Cod * P**power_od

        # coordinates of points (Vc)
        Vc_old_eq = np.array((P_before, V_c_before))
        Vc_new = np.array((P_after, V_c_before - dV_c))
        Vc_new_eq = np.array((P_after, V_c_after))

        # coordinates of points (Vod)
        Vod_old_eq = np.array((P_before, V_od_before))
        Vod_new_eq = np.array((P_after, V_od_after))

        # plot volume curves
        curve_Vc = hv.Curve(zip(P / 10**6, Vc / 10**6), label="Vᶜ")
        curve_Vod = hv.Curve(zip(P / 10**6, Vod / 10**6), label="Vᵒᵈ")

        # empty line in legend
        empty_line_in_legend_curve = hv.Curve([(0, 0), (0, 0)], label=" ").opts(
            line_width=0
        )

        # plot points (Vc)
        Vc_old_eq_point = hv.Points([Vc_old_eq / 10**6]).opts(
            marker="x", color="black", size=10
        )
        Vc_new_point = hv.Points([Vc_new / 10**6]).opts(
            marker="circle", color="black", size=10
        )
        Vc_new_eq_point = hv.Points([Vc_new_eq / 10**6]).opts(
            marker="circle", color="white", line_color="black", size=10
        )

        # plot points (Vod)
        Vod_old_eq_point = hv.Points([Vod_old_eq / 10**6]).opts(
            marker="x", color="black", size=10
        )
        Vod_new_eq_point = hv.Points([Vod_new_eq / 10**6]).opts(
            marker="circle", color="white", line_color="black", size=10
        )

        # hlines (a)
        a1_line = hv.Curve(
            zip(
                (10**-2, Vc_new[0] / 10**6),
                (Vc_new[1] / 10**6, Vc_new[1] / 10**6),
            )
        ).opts(color="#fc4f30", line_dash="dashed")
        a2_line = hv.Curve(
            zip(
                (10**-2, Vc_new_eq[0] / 10**6),
                (Vc_new_eq[1] / 10**6, Vc_new_eq[1] / 10**6),
            )
        ).opts(color="#fc4f30", line_dash="dashed")

        # hlines (b)
        b1_line = hv.Curve(
            zip(
                (Vod_old_eq[0] / 10**6, 10**5),
                (Vod_old_eq[1] / 10**6, Vod_old_eq[1] / 10**6),
            )
        ).opts(color="#30a2da", line_dash="dashed")
        b2_line = hv.Curve(
            zip(
                (Vod_new_eq[0] / 10**6, 10**5),
                (Vod_new_eq[1] / 10**6, Vod_new_eq[1] / 10**6),
            )
        ).opts(color="#30a2da", line_dash="dashed")

        # elbow lines
        elbow_line1 = hv.Curve(
            zip(
                (Vc_new[0] / 10**6, Vc_old_eq[0] / 10**6),
                (Vc_old_eq[1] / 10**6, Vc_old_eq[1] / 10**6),
            ),
            label="ΔP",
        ).opts(color="#e5ae38")
        elbow_line2 = hv.Curve(
            zip(
                (Vc_new[0] / 10**6, Vc_new[0] / 10**6),
                (Vc_old_eq[1] / 10**6, Vc_new[1] / 10**6),
            ),
            label="ΔVᶜ",
        ).opts(color="#6d9060")

        # arrows
        arrow = hv.Curve(
            zip(
                [Vc_old_eq[0] / 10**6, Vc_new[0] / 10**6],
                [Vc_old_eq[1] / 10**6, Vc_new[1] / 10**6],
            )
        ).opts(color="black", line_dash="dashed")
        a_arrow = hv.Curve(
            zip(
                [50, 50],
                [
                    min((Vc_new[1], Vc_new_eq[1])) / 10**6,
                    max((Vc_new[1], Vc_new_eq[1])) / 10**6,
                ],
            )
        ).opts(color="black", line_dash="dashed")
        b_arrow = hv.Curve(
            zip(
                [1100, 1100],
                [
                    min((Vod_old_eq[1], Vod_new_eq[1])) / 10**6,
                    max((Vod_old_eq[1], Vod_new_eq[1])) / 10**6,
                ],
            )
        ).opts(color="black", line_dash="dashed")
        a_text = hv.Text(40, (Vc_new[1] + Vc_new_eq[1]) / 2 / 10**6, "a").opts(
            color="black"
        )
        b_text = hv.Text(900, (Vod_old_eq[1] + Vod_new_eq[1]) / 2 / 10**6, "b").opts(
            color="black"
        )

        # aggregate plot
        plot = (
            curve_Vod
            * curve_Vc
            * empty_line_in_legend_curve
            * Vc_old_eq_point
            * Vc_new_point
            * Vc_new_eq_point
            * Vod_old_eq_point
            * Vod_new_eq_point
            * a1_line
            * a2_line
            * b1_line
            * b2_line
            * elbow_line1
            * elbow_line2
            * arrow
            * a_arrow
            * b_arrow
            * a_text
            * b_text
        ).opts(
            show_grid=True,
            logx=True,
            logy=True,
            xlim=(10, 5000),
            ylim=(2, 5000),
            title="Effect of changes in prism and/or channel\nvolume on basin",
            xlabel="P [10⁶ m³]",
            ylabel="V [10⁶ m³]",
            legend_position="bottom_right",
            height=500,
            responsive=True,
            max_width=400,
        )

        return plot

    app = pn.Row(
        pn.Column(P_before_slider, P_after_slider, Delta_Vc_slider, table_widget),
        plot_fig935,
        sizing_mode="stretch_width",
    )

    return app
