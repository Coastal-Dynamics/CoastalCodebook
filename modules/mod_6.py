import pandas as pd
import numpy as np
import panel as pn
import holoviews as hv

from scipy.interpolate import CubicSpline


def interpolate_bathymetry(df_path, N_points=2000, rolling_param=150):
    # read data
    df = pd.read_csv(df_path, sep="; ", decimal=",", names=["x", "y"], engine="python")

    # define interpolation function
    f = CubicSpline(df.x.values, df.y.values)

    # generate finer x-grid
    x = np.linspace(min(df.x.values), max(df.x.values), N_points)

    # compute new y values
    y = f(x)

    # add padding
    x_start = x[: rolling_param // 2 - 1] - (
        x[: rolling_param // 2][-1] - x[: rolling_param // 2][0]
    )
    x_end = x[-rolling_param // 2 + 1 :] + (
        x[-rolling_param // 2 :][-1] - x[-rolling_param // 2 :][0]
    )

    y_start = y[0] * np.ones(x_start.shape)
    y_end = y[-1] * np.ones(x_end.shape)

    x = np.concatenate([x_start, x, x_end])
    y = np.concatenate([y_start, y, y_end])

    # create dataframe
    df_interpolated = pd.DataFrame({"x": x, "y": y})

    # smooth with rolling mean
    df_interpolated_rolling = df_interpolated
    df_interpolated_rolling = df_interpolated.rolling(rolling_param).mean()

    return df_interpolated_rolling


def cross_shore_transport_gradients(transport_component, x_array):
    """
    This function computes the cross-shore transport gradients for a given transport (component)
    and array containing x-coordinates for the given transport component.
    """

    def rolling_mean(arr, window_size):
        arr_start = np.ones(window_size // 2) * arr[0]
        arr_end = np.ones(window_size // 2) * arr[-1]

        arr = np.concatenate([arr_start, arr, arr_end])

        if window_size % 2 == 0:
            raise ValueError("Window size must be odd for symmetry.")

        half_window = window_size // 2
        kernel = np.ones(window_size) / window_size  # Uniform averaging kernel

        # Compute the rolling mean using convolution
        mean_values = np.convolve(arr, kernel, mode="same")

        # # Zero out values near the borders
        # mean_values[:half_window] = 0
        # mean_values[-half_window:] = 0

        mean_values = mean_values[window_size // 2 : -window_size // 2]

        return mean_values

    gradient_transport_component = -np.gradient(transport_component, x_array)

    # use numpy.gradient function, which employs a simple central differences algorithm
    gradient_transport_component = rolling_mean(gradient_transport_component, 21)

    return gradient_transport_component


# define function to return bathymetry
def bath(loc):
    if loc == "Sennen Cove, Cornwall (UK)":
        return uk_bath
    elif loc == "Scheldt flume (NL)":
        return nl_bath
    else:
        print("Choose either 'Sennen Cove, Cornwall (UK)' or 'Scheldt Flume (NL)'")


def show_transport(baths, tinker_functions):
    """
    Create app that shows transport (shape) functions.
    """
    # read Tinker functions
    (
        Tinker_mean,
        Tinker_osci,
        Tinker_surf_shoal,
        Tinker_onsh,
        Tinker_offs,
        Tinker_swash_surf,
        Tinker_total,
    ) = tinker_functions

    # define titles
    sush_title = pn.pane.Markdown("#### Surf/shoal zone", height=25)
    swsu_title = pn.pane.Markdown("#### Swash/surf zone", height=25)
    tota_title = pn.pane.Markdown("#### All zones", height=25)

    # define sliders
    hb_slider = pn.widgets.FloatSlider(
        name="Breaker depth [m]", start=0.05, end=3, step=0.05, value=0.8
    )
    wl_slider = pn.widgets.FloatSlider(
        name="Water Level [m] (relative to reference)",
        start=-2,
        end=2,
        step=0.05,
        value=0,
    )

    # define float input widget
    A_input = pn.widgets.FloatInput(name="A (Parabolic profile):", value=0.043)
    m_input = pn.widgets.FloatInput(name="m (Parabolic profile):", value=0.773)

    # define dropdown
    bath_dropdown = pn.widgets.Select(
        name="Bathymetry select:",
        options=list(baths.keys()) + ["Parabolic profile (A, m)"],
        value="Sennen Cove, Cornwall (UK)",
    )

    # define switches
    mean_switch = pn.widgets.Checkbox(
        name="Mean transport (surf / shoaling zone)", value=True
    )
    osci_switch = pn.widgets.Checkbox(
        name="Oscillatory transport (surf / shoaling zone)", value=True
    )
    sush_switch = pn.widgets.Checkbox(
        name="Total transport (surf / shoaling zone)", value=False
    )

    onsh_switch = pn.widgets.Checkbox(
        name="Onshore transport (swash / surf zone)", value=False
    )
    offs_switch = pn.widgets.Checkbox(
        name="Offshore transport (swash / surf zone)", value=False
    )
    swsu_switch = pn.widgets.Checkbox(
        name="Total transport (swash / surf zone)", value=False
    )

    tota_switch = pn.widgets.Checkbox(name="Total transport (all)", value=False)

    # make app depend on values of sliders, float input widget, dropdown, and switches
    @pn.depends(
        wl_slider.param.value,
        hb_slider.param.value,
        bath_dropdown.param.value,
        A_input.param.value,
        m_input.param.value,
        mean_switch.param.value,
        osci_switch.param.value,
        sush_switch.param.value,
        onsh_switch.param.value,
        offs_switch.param.value,
        swsu_switch.param.value,
        tota_switch.param.value,
    )

    # define plotting function
    def plot(
        wl,
        h_b,
        loc,
        A,
        m,
        include_mean,
        include_osci,
        include_sush,
        include_onsh,
        include_offs,
        include_swsu,
        include_tota,
    ):
        # select correct bathymetry
        if loc in list(baths.keys()):
            x_correction = {
                "Sennen Cove, Cornwall (UK)": 0,
                "Scheldt flume (NL)": 40,
            }

            df = baths[loc]
            x = df.x - x_correction[loc]
            y = df.y

        # if location is not in pre-set locations, use parabolic profile
        elif loc in ["Parabolic profile (A, m)"]:
            xmin = (-1 / -A) ** (1 / m)
            x = np.linspace(xmin, xmin + 1000, 5000)
            y = -A * x**m + 1  # shift profile 1m upwards, same as Figure 7.7 in book
            x -= xmin  # shift profile to start at x=0
        else:
            print(
                "Choose either 'Sennen Cove, Cornwall (UK)', 'Scheldt flume (NL)', or 'Parabolic profile (A, m)'"
            )

        # create array with depths
        h = np.maximum(np.zeros(y.shape), wl - y)

        # determine breaking point
        x_b = x[np.argmin(np.abs(h - h_b))]

        # plot bathymetry
        bath_plot = (
            hv.Curve((x, y), label="Bathymetry").opts(
                xlabel="x [m]", ylabel="z [m]", color="black"
            )
            * hv.HLine(wl, label="water level").opts(line_dash="dashed")
            * hv.VLine(x_b, label="location of breaking").opts(
                line_dash="dashed", color="grey"
            )
        ).opts(title="Bathymetry")

        # initialize transport curves
        curve_transport = hv.Curve(([], []))
        curve_gradient = hv.Curve(([], []))
        curve_h_hb = hv.Curve(([], []))

        # add each transport component to transport curves, depending on switch value
        if include_mean:
            mean_transport = Tinker_mean(h, h_b)
            mean_transport_gradient = cross_shore_transport_gradients(mean_transport, x)

            curve_transport *= hv.Curve(
                (x, mean_transport), label="mean transport"
            ).opts(color="crimson")
            curve_gradient *= hv.Curve(
                (x, mean_transport_gradient), label="mean transport"
            ).opts(color="crimson")
            curve_h_hb *= hv.Curve(
                (
                    np.linspace(0, 6, 500) / h_b,
                    Tinker_mean(np.linspace(0, 6, 500), h_b),
                ),
                label="mean transport",
            ).opts(color="crimson")

        if include_osci:
            osci_transport = Tinker_osci(h, h_b)
            osci_transport_gradient = cross_shore_transport_gradients(osci_transport, x)

            curve_transport *= hv.Curve(
                (x, osci_transport), label="oscillatory transport"
            ).opts(color="orange")
            curve_gradient *= hv.Curve(
                (x, osci_transport_gradient), label="oscillatory transport"
            ).opts(color="orange")
            curve_h_hb *= hv.Curve(
                (
                    np.linspace(0, 6, 500) / h_b,
                    Tinker_osci(np.linspace(0, 6, 500), h_b),
                ),
                label="oscillatory transport",
            ).opts(color="orange")

        if include_sush:
            sush_transport = Tinker_surf_shoal(h, h_b)
            sush_transport_gradient = cross_shore_transport_gradients(sush_transport, x)

            curve_transport *= hv.Curve(
                (x, sush_transport), label="total transport (surf/shoal)"
            ).opts(color="#aea04b")
            curve_gradient *= hv.Curve(
                (x, sush_transport_gradient), label="total transport (surf/shoal)"
            ).opts(color="#aea04b")
            curve_h_hb *= hv.Curve(
                (
                    np.linspace(0, 6, 500) / h_b,
                    Tinker_surf_shoal(np.linspace(0, 6, 500), h_b),
                ),
                label="total transport (surf/shoal)",
            ).opts(color="#aea04b")

        if include_onsh:
            onsh_transport = Tinker_onsh(h, h_b)
            onsh_transport_gradient = cross_shore_transport_gradients(onsh_transport, x)

            curve_transport *= hv.Curve(
                (x, onsh_transport), label="onshore transport"
            ).opts(color="green")
            curve_gradient *= hv.Curve(
                (x, onsh_transport_gradient), label="onshore transport"
            ).opts(color="green")
            curve_h_hb *= hv.Curve(
                (
                    np.linspace(0, 6, 500) / h_b,
                    Tinker_onsh(np.linspace(0, 6, 500), h_b),
                ),
                label="onshore transport",
            ).opts(color="green")

        if include_offs:
            offs_transport = Tinker_offs(h, h_b)
            offs_transport_gradient = cross_shore_transport_gradients(offs_transport, x)

            curve_transport *= hv.Curve(
                (x, offs_transport), label="offshore transport"
            ).opts(color="blue")
            curve_gradient *= hv.Curve(
                (x, offs_transport_gradient), label="offshore transport"
            ).opts(color="blue")
            curve_h_hb *= hv.Curve(
                (
                    np.linspace(0, 6, 500) / h_b,
                    Tinker_offs(np.linspace(0, 6, 500), h_b),
                ),
                label="offshore transport",
            ).opts(color="blue")

        if include_swsu:
            swsu_transport = Tinker_swash_surf(h, h_b)
            swsu_transport_gradient = cross_shore_transport_gradients(swsu_transport, x)

            curve_transport *= hv.Curve(
                (x, swsu_transport), label="total transport (swash/surf)"
            ).opts(color="#FF00FF")
            curve_gradient *= hv.Curve(
                (x, swsu_transport_gradient), label="total transport (swash/surf)"
            ).opts(color="#FF00FF")
            curve_h_hb *= hv.Curve(
                (
                    np.linspace(0, 6, 500) / h_b,
                    Tinker_swash_surf(np.linspace(0, 6, 500), h_b),
                ),
                label="total transport (swash/surf)",
            ).opts(color="#FF00FF")

        if include_tota:
            tota_transport = Tinker_total(h, h_b)
            tota_transport_gradient = cross_shore_transport_gradients(tota_transport, x)

            curve_transport *= hv.Curve(
                (x, tota_transport), label="total transport (all)"
            ).opts(color="grey")
            curve_gradient *= hv.Curve(
                (x, tota_transport_gradient), label="total transport (all)"
            ).opts(color="grey")
            curve_h_hb *= hv.Curve(
                (
                    np.linspace(0, 6, 500) / h_b,
                    Tinker_total(np.linspace(0, 6, 500), h_b),
                ),
                label="total transport (all)",
            ).opts(color="grey")

        # add axis lines
        curve_transport *= hv.HLine(0).opts(color="black")
        curve_gradient *= hv.HLine(0).opts(color="black")
        curve_h_hb *= hv.HLine(0).opts(color="black")

        # add axis labels and titles
        transport_plot = curve_transport.opts(
            xlabel="x [m]", ylabel="Q [kg/m/s]", title="Transport (x)"
        )
        gradient_plot = curve_gradient.opts(
            xlabel="x [m]", ylabel="dQ/dx [kg/m2/s]", title="Transport gradient (x)"
        )
        h_hb_plot = curve_h_hb.opts(
            xlabel="h / h_b [-]", ylabel="Q [kg/m/s]", title="Transport (h/h_b)"
        )

        # change x-ticks to be positive onshore
        x_locs = {
            "Sennen Cove, Cornwall (UK)": np.array([0, 20, 40, 60, 80, 100, 120]),
            "Scheldt flume (NL)": np.array([-10, -15, -20, -25, -30, -35, -40, -45]),
            "parabolic": np.array([0, 200, 400, 600, 800, 1000]),
        }
        if loc in x_locs.keys():
            x_loc = x_locs[loc]
        else:
            x_loc = x_locs["parabolic"]

        for plot in [bath_plot, transport_plot, gradient_plot]:
            plot.opts(xticks=[(i, j) for i, j in zip(x_loc, -x_loc)])

        # add all plots to same figure
        p = (
            (
                bath_plot.opts(
                    height=200,
                    width=1000,
                    show_grid=True,
                    active_tools=[],
                    toolbar=None,
                    legend_position="right",
                )
                + transport_plot.opts(
                    height=250,
                    width=1000,
                    show_grid=True,
                    active_tools=[],
                    toolbar=None,
                    legend_position="right",
                ).redim(y=hv.Dimension("Transport (x)", soft_range=(-0.1, 0.1)))
                + gradient_plot.opts(
                    height=250,
                    width=1000,
                    show_grid=True,
                    active_tools=[],
                    toolbar=None,
                    legend_position="right",
                ).redim(
                    y=hv.Dimension("Transport gradient (x)", soft_range=(-0.01, 0.01))
                )
                + h_hb_plot.opts(
                    height=250,
                    width=1000,
                    show_grid=True,
                    active_tools=[],
                    toolbar=None,
                    legend_position="right",
                ).redim(
                    x=hv.Dimension("h / h_b [-]", range=(0, 2)),
                    y=hv.Dimension("Transport (h /h_b)", soft_range=(-0.1, 0.1)),
                )
            )
            .opts(shared_axes=False)
            .cols(1)
        )

        return p

    # create app
    app = pn.Column(
        pn.Row(
            bath_dropdown,
            pn.Column(wl_slider, hb_slider),
            pn.Column(A_input, m_input),
            align="center",
        ),
        pn.Row(
            pn.Column(sush_title, mean_switch, osci_switch, sush_switch),
            pn.Column(swsu_title, onsh_switch, offs_switch, swsu_switch),
            pn.Column(tota_title, tota_switch),
            align="center",
        ),
        pn.Row(plot, align="center"),
    )

    return app


def h_hb_transport(tinker_functions):
    """
    Create app with h/hb plots.
    """
    # read Tinker functions
    (
        Tinker_mean,
        Tinker_osci,
        Tinker_surf_shoal,
        Tinker_onsh,
        Tinker_offs,
        Tinker_swash_surf,
        Tinker_total,
    ) = tinker_functions

    # add titles
    sush_title = pn.pane.Markdown("#### Surf/shoal zone", height=25)
    swsu_title = pn.pane.Markdown("#### Swash/surf zone", height=25)
    tota_title = pn.pane.Markdown("#### All zones", height=25)

    # add hb slider
    hb_slider = pn.widgets.RangeSlider(
        name="Range of hb [m] values to plot", start=0, end=2.5, value=(0, 2.5)
    )

    # add switches
    mean_switch = pn.widgets.Checkbox(
        name="Mean transport (surf / shoaling zone)", value=True
    )
    osci_switch = pn.widgets.Checkbox(
        name="Oscillatory transport (surf / shoaling zone)", value=True
    )
    sush_switch = pn.widgets.Checkbox(
        name="Total transport (surf / shoaling zone)", value=False
    )

    onsh_switch = pn.widgets.Checkbox(
        name="Onshore transport (swash / surf zone)", value=False
    )
    offs_switch = pn.widgets.Checkbox(
        name="Offshore transport (swash / surf zone)", value=False
    )
    swsu_switch = pn.widgets.Checkbox(
        name="Total transport (swash / surf zone)", value=False
    )

    tota_switch = pn.widgets.Checkbox(name="Total transport (all)")

    # make figure depend on sliders and switches
    @pn.depends(
        hb_slider.param.value,
        mean_switch.param.value,
        osci_switch.param.value,
        sush_switch.param.value,
        onsh_switch.param.value,
        offs_switch.param.value,
        swsu_switch.param.value,
        tota_switch.param.value,
    )

    # define plotting function
    def plot(
        h_b,
        include_mean,
        include_osci,
        include_sush,
        include_onsh,
        include_offs,
        include_swsu,
        include_tota,
    ):
        hb1 = h_b[0]
        hb4 = h_b[1]
        hb2 = (hb4 - hb1) * 1 / 3 + hb1
        hb3 = (hb4 - hb1) * 2 / 3 + hb1

        # initialize curves
        curve_transport1 = hv.Curve(([], []))
        curve_transport2 = hv.Curve(([], []))
        curve_transport3 = hv.Curve(([], []))
        curve_transport4 = hv.Curve(([], []))

        h = np.linspace(0, 5, 500)

        # create curves
        if include_mean:
            curve_transport1 *= hv.Curve(
                (h / hb1, Tinker_mean(h, hb1)), label="mean transport"
            ).opts(color="crimson")
            curve_transport2 *= hv.Curve(
                (h / hb2, Tinker_mean(h, hb2)), label="mean transport"
            ).opts(color="crimson")
            curve_transport3 *= hv.Curve(
                (h / hb3, Tinker_mean(h, hb3)), label="mean transport"
            ).opts(color="crimson")
            curve_transport4 *= hv.Curve(
                (h / hb4, Tinker_mean(h, hb4)), label="mean transport"
            ).opts(color="crimson")

        if include_osci:
            curve_transport1 *= hv.Curve(
                (h / hb1, Tinker_osci(h, hb1)), label="oscillatory transport"
            ).opts(color="orange")
            curve_transport2 *= hv.Curve(
                (h / hb2, Tinker_osci(h, hb2)), label="oscillatory transport"
            ).opts(color="orange")
            curve_transport3 *= hv.Curve(
                (h / hb3, Tinker_osci(h, hb3)), label="oscillatory transport"
            ).opts(color="orange")
            curve_transport4 *= hv.Curve(
                (h / hb4, Tinker_osci(h, hb4)), label="oscillatory transport"
            ).opts(color="orange")

        if include_sush:
            curve_transport1 *= hv.Curve(
                (h / hb1, Tinker_surf_shoal(h, hb1)),
                label="Total transport (surf/shoal)",
            ).opts(color="#aea04b")
            curve_transport2 *= hv.Curve(
                (h / hb2, Tinker_surf_shoal(h, hb2)),
                label="Total transport (surf/shoal)",
            ).opts(color="#aea04b")
            curve_transport3 *= hv.Curve(
                (h / hb3, Tinker_surf_shoal(h, hb3)),
                label="Total transport (surf/shoal)",
            ).opts(color="#aea04b")
            curve_transport4 *= hv.Curve(
                (h / hb4, Tinker_surf_shoal(h, hb4)),
                label="Total transport (surf/shoal)",
            ).opts(color="#aea04b")

        if include_onsh:
            curve_transport1 *= hv.Curve(
                (h / hb1, Tinker_onsh(h, hb1)), label="onshore transport"
            ).opts(color="green")
            curve_transport2 *= hv.Curve(
                (h / hb2, Tinker_onsh(h, hb2)), label="onshore transport"
            ).opts(color="green")
            curve_transport3 *= hv.Curve(
                (h / hb3, Tinker_onsh(h, hb3)), label="onshore transport"
            ).opts(color="green")
            curve_transport4 *= hv.Curve(
                (h / hb4, Tinker_onsh(h, hb4)), label="onshore transport"
            ).opts(color="green")

        if include_offs:
            curve_transport1 *= hv.Curve(
                (h / hb1, Tinker_offs(h, hb1)), label="offshore transport"
            ).opts(color="blue")
            curve_transport2 *= hv.Curve(
                (h / hb2, Tinker_offs(h, hb2)), label="offshore transport"
            ).opts(color="blue")
            curve_transport3 *= hv.Curve(
                (h / hb3, Tinker_offs(h, hb3)), label="offshore transport"
            ).opts(color="blue")
            curve_transport4 *= hv.Curve(
                (h / hb4, Tinker_offs(h, hb4)), label="offshore transport"
            ).opts(color="blue")

        if include_swsu:
            curve_transport1 *= hv.Curve(
                (h / hb1, Tinker_swash_surf(h, hb1)),
                label="Total transport (swash/surf)",
            ).opts(color="#FF00FF")
            curve_transport2 *= hv.Curve(
                (h / hb2, Tinker_swash_surf(h, hb2)),
                label="Total transport (swash/surf)",
            ).opts(color="#FF00FF")
            curve_transport3 *= hv.Curve(
                (h / hb3, Tinker_swash_surf(h, hb3)),
                label="Total transport (swash/surf)",
            ).opts(color="#FF00FF")
            curve_transport4 *= hv.Curve(
                (h / hb4, Tinker_swash_surf(h, hb4)),
                label="Total transport (swash/surf)",
            ).opts(color="#FF00FF")

        if include_tota:
            curve_transport1 *= hv.Curve(
                (h / hb1, Tinker_total(h, hb1)), label="Total transport (all)"
            ).opts(color="grey")
            curve_transport2 *= hv.Curve(
                (h / hb2, Tinker_total(h, hb2)), label="Total transport (all)"
            ).opts(color="grey")
            curve_transport3 *= hv.Curve(
                (h / hb3, Tinker_total(h, hb3)), label="Total transport (all)"
            ).opts(color="grey")
            curve_transport4 *= hv.Curve(
                (h / hb4, Tinker_total(h, hb4)), label="Total transport (all)"
            ).opts(color="grey")

        # create axis line
        curve_transport1 *= hv.HLine(0).opts(color="black")
        curve_transport2 *= hv.HLine(0).opts(color="black")
        curve_transport3 *= hv.HLine(0).opts(color="black")
        curve_transport4 *= hv.HLine(0).opts(color="black")

        # create plots
        transport_plot1 = curve_transport1.opts(
            xlabel="h / h_b [-]", ylabel="Q [kg/m/s]", title=f"Transport (hb={hb1:.1f})"
        )
        transport_plot2 = curve_transport2.opts(
            xlabel="h / h_b [-]", ylabel="Q [kg/m/s]", title=f"Transport (hb={hb2:.1f})"
        )
        transport_plot3 = curve_transport3.opts(
            xlabel="h / h_b [-]", ylabel="Q [kg/m/s]", title=f"Transport (hb={hb3:.1f})"
        )
        transport_plot4 = curve_transport4.opts(
            xlabel="h / h_b [-]", ylabel="Q [kg/m/s]", title=f"Transport (hb={hb4:.1f})"
        )

        # add all plots to same figure
        p = (
            transport_plot4.opts(
                height=250,
                width=1000,
                show_grid=True,
                active_tools=[],
                toolbar=None,
                legend_position="right",
            ).redim(
                x=hv.Dimension("h / h_b [-]", range=(0, 2)),
                y=hv.Dimension("Transport (h /h_b)", soft_range=(-0.1, 0.1)),
            )
            + transport_plot3.opts(
                height=250,
                width=1000,
                show_grid=True,
                active_tools=[],
                toolbar=None,
                legend_position="right",
            ).redim(
                x=hv.Dimension("h / h_b [-]", range=(0, 2)),
                y=hv.Dimension("Transport (h /h_b)", soft_range=(-0.1, 0.1)),
            )
            + transport_plot2.opts(
                height=250,
                width=1000,
                show_grid=True,
                active_tools=[],
                toolbar=None,
                legend_position="right",
            ).redim(
                x=hv.Dimension("h / h_b [-]", range=(0, 2)),
                y=hv.Dimension("Transport (h /h_b)", soft_range=(-0.1, 0.1)),
            )
            + transport_plot1.opts(
                height=250,
                width=1000,
                show_grid=True,
                active_tools=[],
                toolbar=None,
                legend_position="right",
            ).redim(
                x=hv.Dimension("h / h_b [-]", range=(0, 2)),
                y=hv.Dimension("Transport (h /h_b)", soft_range=(-0.1, 0.1)),
            )
        ).cols(1)

        return p

    # define app layout
    app = pn.Column(
        pn.Row(hb_slider, align="center"),
        pn.Row(
            pn.Column(sush_title, mean_switch, osci_switch, sush_switch),
            pn.Column(swsu_title, onsh_switch, offs_switch, swsu_switch),
            pn.Column(tota_title, tota_switch),
            align="center",
        ),
        pn.Row(plot, align="center"),
    )

    return app
