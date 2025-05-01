import math
from datetime import datetime, timedelta
import datetime as dt
import os
from warnings import filterwarnings

import pickle
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
import uptide
import panel as pn

# import ipywidgets as widgets

hv.extension("bokeh")


def tidal_scheveningen(dir_path, start_date, end_date):
    ## Download GESLA tide gauge data for Scheveningen
    fname = "Scheveningen_GESLA.pkl"
    tide_gauge = pd.read_pickle(os.path.join(dir_path, fname))

    ## Read FES2014 amplitude and phase data from pickle files already prepared for you and calculate the signal
    ## see the commented script at the bottom of the notebook if you want to know how to load FES2014 amplitude and phase data yourself
    tide = {}
    amplitudes = {}
    phases = {}
    comps = [
        "EPS2",
        "J1",
        "K1",
        "K2",
        "L2",
        "LAMBDA2",
        "M2",
        "M3",
        "M4",
        "M6",
        "M8",
        "MF",
        "MKS2",
        "MM",
        "MN4",
        "MS4",
        "MSF",
        "MSQM",
        "MTM",
        "MU2",
        "N2",
        "N4",
        "NU2",
        "O1",
        "P1",
        "Q1",
        "R2",
        "S1",
        "S2",
        "S4",
        "SA",
        "SSA",
        "T2",
    ]

    for comp in comps:
        fp = os.path.join(dir_path, "fes2014_amp_ph", ("02_" + comp.lower() + ".p"))
        tide[comp.lower()] = pd.read_pickle(fp)
        amplitudes[comp.lower()] = tide[comp.lower()]["amplitude"]["scheveningen"]
        phases[comp.lower()] = tide[comp.lower()]["phase"]["scheveningen"]

    # Choose the initial time for calculating the tidal signal (has to be between 1977-2017)
    initial_time = datetime(1977, 1, 1, 12, 0, 0)

    tidal_signal = uptide.Tides(
        comps
    )  # select which constituents to use, we will use all
    tidal_signal.set_initial_time(
        initial_time
    )  # set t=0 at 1 Jan 1977, UTC 12:00, arbitrary choice
    amp = [amplitudes[comp.lower()] for comp in comps]
    pha = [math.radians(phases[comp.lower()]) for comp in comps]  # phase (in radians!)

    t = np.arange(
        0, 365 * 40 * 24 * 3600, 900
    )  # seconds since initial time, 40 years since 1977, 15min frequency
    dates = np.array(
        [initial_time + timedelta(seconds=int(s)) for s in t]
    )  # so that we have datetime on x-axis
    eta = tidal_signal.from_amplitude_phase(amp, pha, t)  # calculate the signal
    eta_df = pd.DataFrame({"eta": eta}, index=dates)

    filtered_gauge = tide_gauge[start_date:end_date]
    filtered_tide = eta_df[start_date:end_date]

    ftide = hv.Curve(
        (filtered_tide.index, filtered_tide["eta"].values / 100), label="Tidal signal"
    )
    ftide.opts(color="blue", show_legend=True, line_width=0.7)

    fgauge = hv.Curve(
        (filtered_gauge.index, filtered_gauge.values), label="Observed sea level"
    )
    fgauge.opts(color="black", show_legend=True, line_width=0.7)

    tidal_signal = hv.Overlay(ftide * fgauge).opts(
        aspect=2,
        responsive=True,
        show_grid=True,
        xlabel="Date (month/day or month/year or year)",
        ylabel="Sea level [m]",
        title="Scheveningen",
        legend_position="right",
    )
    filterwarnings("ignore", category=FutureWarning)
    return tidal_signal, tide_gauge, eta_df


def plot_timeseries_with_interactive_controls_hv(tide_gauge, eta_df, scheveningen):
    # Define a list of checkboxes for component selection and put them in one row

    dates = np.array(
        [
            datetime(2000, 1, 1, 0, 0, 0) + timedelta(seconds=item * 3600)
            for item in range(24 * 365)  # 1 year
        ]
    )

    comps = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA"]

    checkboxes = pn.widgets.CheckBoxGroup(
        name="Tidal constituents", value=["M2", "S2"], options=comps, inline=True
    )

    slider = pn.widgets.DatetimeRangeSlider(
        name="Date Slider",
        start=dt.datetime(2000, 1, 1),
        end=dt.datetime(2001, 1, 1),
        value=(dt.datetime(2000, 5, 1), dt.datetime(2000, 6, 1)),
        step=60000 * 60 * 24,
    )

    t = pd.date_range(
        start=dt.datetime(2000, 1, 1), end=dt.datetime(2001, 1, 1), freq="15min"
    ).values[: len(scheveningen["k1"])]

    df = pd.DataFrame({"time": t})

    for comp in comps:
        lower_comp = comp.lower()
        df[lower_comp] = scheveningen[lower_comp]["eta"].values / 100

    @pn.depends(slider.param.value, checkboxes.param.value)
    def hv_plot_timeseries(date_range, selected_components):
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(
            date_range[1]
        )

        plot1 = hv.Curve(
            zip(pd.date_range(start=start_date, end=end_date, periods=1), [0, 0])
        )
        plot2 = hv.Curve(
            zip(pd.date_range(start=start_date, end=end_date, periods=1), [0, 0])
        )
        plot3 = hv.Curve(
            zip(pd.date_range(start=start_date, end=end_date, periods=1), [0, 0])
        )
        plot4 = hv.Curve(
            zip(pd.date_range(start=start_date, end=end_date, periods=1), [0, 0])
        )

        if len(selected_components) > 0:
            ### Plot 1 ###

            lower_case_components = [comp.lower() for comp in selected_components]

            df_plot = df[(df.time.values >= start_date) & (df.time.values < end_date)][
                ["time"] + lower_case_components
            ]

            for comp, col in zip(selected_components, list(df_plot.columns[1:])):
                plot1 *= hv.Curve(
                    (df_plot.time.values, df_plot[col].values), label=comp
                ).opts(
                    line_width=2,
                )

            ### Plot 2 ###

            df_plot["sum_selected_components"] = np.sum(
                df_plot.iloc[:, 1:].values.T, axis=0
            )

            sum_components_curve = hv.Curve(
                (df_plot.time.values, df_plot["sum_selected_components"]),
                label="Sum selected comp.",
            ).opts(line_width=2, color="blue")

            plot2 *= sum_components_curve

            ### Plot 3 ###

            tidal_sign_curve = hv.Curve(
                (
                    eta_df[start_date:end_date].index,
                    eta_df[start_date:end_date].values.flatten() / 100,
                ),
                label="Total tidal signal",
            ).opts(line_width=2, color="orange")

            plot3 *= sum_components_curve * tidal_sign_curve

            ### Plot 4 ###

            observed_curve = hv.Curve(
                (
                    tide_gauge[start_date:end_date].index,
                    tide_gauge[start_date:end_date].values.flatten(),
                ),
                label="Observed sea level",
            ).opts(color="darkred", line_width=2)

            plot4 *= sum_components_curve * tidal_sign_curve * observed_curve

        fixed_height = 200

        plot1.opts(
            ylabel="Sea level [m]",
            xlabel="Date [month/year or month/day]",
            title="Selected components",
            legend_position="right",
            legend_cols=2,
            responsive=True,
            frame_height=fixed_height,
        )

        plot2.opts(
            ylabel="Sea level [m]",
            xlabel="Date [month/year or month/day]",
            title="Sum selected comp.",
            show_legend=False,
            responsive=True,
            frame_height=fixed_height,
        )

        plot3.opts(
            ylabel="Sea level [m]",
            xlabel="Date [month/year or month/day]",
            title="Comparison with total tidal signal",
            legend_position="right",
            responsive=True,
            frame_height=fixed_height,
        )

        plot4.opts(
            ylabel="Sea level [m]",
            xlabel="Date [month/year or month/day]",
            title="Comparison with total tidal signal and observed sea level",
            legend_position="right",
            responsive=True,
            frame_height=fixed_height,
        )

        return (plot1 + plot2 + plot3 + plot4).cols(1)

    filterwarnings("ignore", category=FutureWarning)

    app = pn.Column(
        pn.Row(checkboxes, sizing_mode="stretch_width"),
        pn.Row(slider, sizing_mode="stretch_width"),
        pn.Row(hv_plot_timeseries, sizing_mode="stretch_both"),
        width_policy="max",
    )

    return app


# Function no longer used, now changed to plot_timeseries_with_interactive_controls_hv
# 20/2/25 Leave uncommented for now so that students can still run the old version of the notebook on the hub
def plot_timeseries_with_interactive_controls(tide_gauge, eta_df, scheveningen):
    # Define a list of checkboxes for component selection and put them in one row

    dates = np.array(
        [
            datetime(2000, 1, 1, 0, 0, 0) + timedelta(seconds=item * 3600)
            for item in range(24 * 365)  # 1 year
        ]
    )

    comps = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA"]
    checkboxes = [
        widgets.Checkbox(
            value=(comp == "M2"), description=comp, layout=widgets.Layout(width="auto")
        )
        for comp in comps
    ]
    checkbox_row = widgets.HBox(
        checkboxes, layout=widgets.Layout(display="flex", flex_flow="row wrap")
    )

    # Plot with interactive slider and checkboxes
    date_range_selector = widgets.SelectionRangeSlider(
        options=[(date.strftime("%d/%m %Hh"), date) for date in dates],
        index=(0, len(dates) - 1),
        description="Dates",
        orientation="horizontal",
        layout={"width": "700px"},
        continuous_update=False,
        readout=True,
    )

    def hv_plot_timeseries(date_range, **kwargs):
        start_date, end_date = date_range
        # Filter selected components
        selected_components = [comp for comp, value in kwargs.items() if value]

        plot1 = []
        # Plot selected components in first figure
        for comp in selected_components:
            curve = hv.Curve(
                scheveningen[comp.lower()][start_date:end_date] / 100, label=comp
            )
            curve.opts(line_width=0.5, show_legend=True)
            plot1.append(curve)

        # Create an "empty" dataframe for an empty figure so that the components can be plotted
        tide_df = pd.DataFrame(
            scheveningen[selected_components[0].lower()][start_date:end_date],
            columns=["gauge"],
        )
        tide_df["new"] = 0
        components = hv.Curve(
            (
                tide_df[start_date:end_date].index,
                tide_df[start_date:end_date]["new"].values,
            )
        ).opts(line_alpha=0)

        for i in range(len(plot1)):
            components *= plot1[i]
        components.opts(
            aspect=4,
            responsive=True,
            legend_position="right",
            ylabel="Sea level [m]",
            xlabel="Date [month/year or month/day]",
            legend_cols=2,
        )

        # Calculate and plot the sum on axes[1]
        sum_values = sum(
            scheveningen[comp.lower()][start_date:end_date]
            for comp in selected_components
        )

        sum_comp = hv.Curve(
            (sum_values.index, sum_values["eta"].values.flatten() / 100),
            label="Sum selected comp.",
        )
        sum_comp.opts(
            line_width=0.5,
            color="darkblue",
            show_legend=True,
            ylabel="Sea level [m]",
            aspect=4,
            responsive=True,
        )

        # Plot total tidal signal and the obtained sum on axes [2]
        tidal_sign = hv.Curve(
            (
                eta_df[start_date:end_date].index,
                eta_df[start_date:end_date].values.flatten() / 100,
            ),
            label="Total tidal signal",
        )
        tidal_sign.opts(
            line_width=0.5,
            show_legend=True,
            color="darkorange",
            aspect=4,
            responsive=True,
        )

        third_fig = (tidal_sign * sum_comp).opts(
            aspect=4,
            responsive=True,
            legend_position="right",
            xlabel="Date [month/year or month/day]",
        )

        observed_sl = hv.Curve(
            (
                tide_gauge[start_date:end_date].index,
                tide_gauge[start_date:end_date].values.flatten(),
            ),
            label="Observed sea level",
        )
        observed_sl.opts(color="black", show_legend=True, line_width=0.5)
        fourth_fig = (tidal_sign * sum_comp * observed_sl).opts(
            aspect=4,
            responsive=True,
            legend_position="right",
            xlabel="Date [month/years or month/days]",
        )

        timeseries = (
            hv.Layout(
                components.opts(title="Selected components")
                + sum_comp.opts(
                    aspect=4,
                    responsive=True,
                    show_legend=True,
                    xlabel="Date [month/year or month/day]",
                )
                + third_fig
                + fourth_fig
            )
            .cols(1)
            .opts(width=700)
        )
        return display(timeseries)

    filterwarnings("ignore", category=FutureWarning)

    figure1 = widgets.interactive(
        hv_plot_timeseries,
        date_range=date_range_selector,
        **{checkbox.description: checkbox for checkbox in checkboxes},
    )
    # Create a new container for arranging controls
    controls1 = widgets.VBox([checkbox_row, figure1.children[-1]])
    controls1.layout.height = "100%"
    display(controls1)


def tidal_constituents(dir_path):
    ## Load FES2014 amplitudes
    data_dir_path = os.path.join(dir_path, "fes2014_amp_ph")
    tide = {}
    amplitudes = {}

    # This time we will include more constituents:
    comps = [
        "eps2",
        "j1",
        "k1",
        "k2",
        "l2",
        "lambda2",
        "m2",
        "m3",
        "m4",
        "m6",
        "m8",
        "mf",
        "mm",
        "mn4",
        "ms4",
        "msf",
        "mtm",
        "mu2",
        "n2",
        "nu2",
        "o1",
        "p1",
        "q1",
        "r2",
        "s1",
        "s2",
        "s4",
        "sa",
        "ssa",
        "t2",
    ]

    for comp in comps:
        fp = os.path.join(data_dir_path, ("02_" + comp + ".p"))
        tide[comp] = pd.read_pickle(fp)
        amplitudes[comp] = tide[comp]["amplitude"]["scheveningen"]

    component_names = list(amplitudes.keys())
    component_names_upper = [
        comp.upper() for comp in component_names
    ]  # Convert to uppercase
    amplitude_values = [value / 100 for value in amplitudes.values()]

    periods = [
        13.13,
        23.09848146,
        23.93447213,
        11.96723606,
        12.19162085,
        12.22177348,
        12.4206012,
        8.280400802,
        6.210300601,
        4.140200401,
        3.105150301,
        327.8599387,
        661.3111655,
        6.269173724,
        6.103339275,
        354.3670666,
        219,
        12.8717576,
        12.65834751,
        12.62600509,
        25.81933871,
        24.06588766,
        26.868350,
        11.98359564,
        24,
        12,
        6,
        8766.15265,
        4383.076325,
        12.01644934,
    ]  # in [h]

    frequency = [1 / (period / 24) for period in periods]  # in [1/day]

    semidiurnal = ["m2", "s2", "n2", "k2"]
    diurnal = ["k1", "o1", "p1", "q1"]
    frequency_val = [float(f) for f in frequency]
    frequency_val.sort()
    ticks = np.arange(0, max(frequency) + 0.1, 0.5)

    ## Bar plot

    # Define a small offset to avoid log-scale issues
    epsilon = 1e-3  # Small value to ensure no amplitudes are zero

    loc_x = []
    names = []
    sorted_data = sorted(
        zip(frequency, amplitude_values, component_names), key=lambda x: x[0]
    )

    for fre, amp, name in sorted_data:
        adjusted_amp = amp + epsilon  # Add the offset
        # Plot semi-diurnal components in orange
        if name in semidiurnal:
            loc_x.append(
                hv.Segments(
                    [(fre, epsilon, fre, adjusted_amp)], label="Semi-diurnal"
                ).opts(color="orange", show_legend=True, line_width=3)
            )  # Vertical segment
            names.append(
                hv.Text(fre, adjusted_amp * 1.1, name.upper()).opts(color="orange")
            )  # Label text
        # Plot diurnal components in red
        elif name in diurnal:
            loc_x.append(
                hv.Segments([(fre, epsilon, fre, adjusted_amp)], label="Diurnal").opts(
                    color="red", show_legend=True, line_width=3
                )
            )  # Vertical segment
            names.append(
                hv.Text(fre, adjusted_amp * 1.1, name.upper()).opts(color="red")
            )  # Label text
        else:
            loc_x.append(
                hv.Segments([(fre, epsilon, fre, adjusted_amp)]).opts(
                    color="skyblue", line_width=3
                )
            )  # Vertical segment
            names.append(hv.Text(fre, adjusted_amp * 1.1, name.upper()))  # Label text

    # Combine all vertical lines and labels into an overlay and display
    plot = hv.Overlay(loc_x + names).opts(
        show_legend=True,
        aspect=2,
        responsive=True,
        xlabel="Frequency [1/day]",
        ylabel="Amplitude [m]",
        ylim=(10**-3, 1),
        logy=True,
        legend_position="right",
    )
    return plot
