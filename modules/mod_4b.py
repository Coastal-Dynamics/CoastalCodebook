import datetime as dt
from warnings import filterwarnings

import pickle
import pandas as pd
import holoviews as hv
import panel as pn
import numpy as np

from warnings import filterwarnings


def pickle_to_df(fp, components):
    """Loads a pickle file and converts it to a dataframe
    fp (Path): path to pickle file
    components (List): List of components(Str) to add to the dataframe
    """
    # load file
    with open(fp, "rb") as pickle_file:
        f = pickle.load(pickle_file)

    # initialize df with time column
    t = pd.date_range(
        start=dt.datetime(2000, 1, 1), end=dt.datetime(2001, 1, 1), freq="15min"
    ).values[: len(f["k1"])]

    df = pd.DataFrame({"time": t})

    sum_selectable_components = np.zeros(df.time.shape)

    for comp in components:
        lower_comp = comp.lower()
        df[comp] = f[lower_comp]["eta"].values / 100
        sum_selectable_components += df[comp]

    df["sum_selectable_components"] = sum_selectable_components

    full = np.zeros(df.time.shape)
    for key in f.keys():
        full += f[key]["eta"].values

    df["all_components"] = full / 100

    return df


# updated plotting function
def plot_2timeseries_with_interactive_controls_pn(comps, locations):
    # Drop-down with options (locations) for left and right plots
    dropdown1 = pn.widgets.Select(
        name="Location (left)",
        options=list(locations.keys()),
        value="Scheveningen",
    )
    dropdown2 = pn.widgets.Select(
        name="Location (right)",
        options=list(locations.keys()),
        value="Jakarta",
    )

    # Define a list of checkboxes for component selection and put them in one row
    checkboxes = pn.widgets.CheckBoxGroup(
        name="Tidal constituents",
        value=["M2", "S2"],
        options=comps,
        inline=True,
    )

    # date slider
    slider = pn.widgets.DatetimeRangeSlider(
        name="Date Slider",
        start=dt.datetime(2000, 1, 1),
        end=dt.datetime(2001, 1, 1),
        value=(dt.datetime(2000, 5, 1), dt.datetime(2000, 5, 15)),
        step=60000 * 60 * 24,
    )

    @pn.depends(
        dropdown1.param.value,
        dropdown2.param.value,
        slider.param.value,
        checkboxes.param.value,
    )
    def hv_plot_timeseries(loc1, loc2, date_range, selected_components):
        # set starting and end date
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(
            date_range[1]
        )

        # set correct df
        df1 = locations[loc1]
        df2 = locations[loc2]

        # mask out correct timeframe
        df1 = df1[(df1.time.values >= start_date) & (df1.time.values <= end_date)]
        df2 = df2[(df2.time.values >= start_date) & (df2.time.values <= end_date)]

        # initialize holoviews plots
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
        plot5 = hv.Curve(
            zip(pd.date_range(start=start_date, end=end_date, periods=1), [0, 0])
        )
        plot6 = hv.Curve(
            zip(pd.date_range(start=start_date, end=end_date, periods=1), [0, 0])
        )

        # plot 1 & 2
        sum_selected_components1 = np.zeros(df1.time.values.shape)
        sum_selected_components2 = np.zeros(df2.time.values.shape)

        # Plot selected components
        for comp in selected_components:
            values1 = df1[comp].values
            values2 = df2[comp].values

            curve1 = hv.Curve((df1.time.values, values1), label=comp).opts(
                line_width=1.5
            )

            curve2 = hv.Curve((df2.time.values, values2), label=comp).opts(
                line_width=1.5
            )

            plot1 *= curve1
            plot2 *= curve2

            sum_selected_components1 += values1
            sum_selected_components2 += values2

        # plot 3 & 4
        curve3 = hv.Curve(
            (df1.time.values, df1.sum_selectable_components.values),
            label="All selectable",
        ).opts(
            line_width=1.5,
            color="Darkblue",
        ) * hv.Curve(
            (df1.time.values, sum_selected_components1), label="Selected"
        ).opts(
            line_width=1.5, color="Green"
        )

        curve4 = hv.Curve(
            (df2.time.values, df2.sum_selectable_components.values),
            label="All selectable",
        ).opts(
            line_width=1.5,
            color="Darkblue",
        ) * hv.Curve(
            (df2.time.values, sum_selected_components2), label="Selected"
        ).opts(
            line_width=1.5,
            color="Green",
        )

        plot3 *= curve3
        plot4 *= curve4

        # plot 5 & 6
        curve5 = hv.Curve(
            (df1.time.values, df1.all_components.values), label="All FES"
        ).opts(
            line_width=1.5,
            color="Grey",
        ) * hv.Curve(
            (df1.time.values, df1.sum_selectable_components.values),
            label="All selectable",
        ).opts(
            line_width=1.5,
            color="Darkblue",
        )

        curve6 = hv.Curve(
            (df2.time.values, df2.all_components.values), label="All FES"
        ).opts(
            line_width=1.5,
            color="Grey",
        ) * hv.Curve(
            (df2.time.values, df2.sum_selectable_components.values),
            label="All selectable",
        ).opts(
            line_width=1.5,
            color="Darkblue",
        )

        plot5 *= curve5
        plot6 *= curve6

        # plotting options
        fixed_height = 200

        plot1.opts(
            title=f"Tidal constituents at {loc1}",
            show_legend=True,
            xlabel="Time",
            ylabel="Sea level [m]",
            responsive=True,
            legend_position="right",
            legend_cols=2,
            frame_height=fixed_height,
        )
        plot2.opts(
            title=f"Tidal constituents at {loc2}",
            show_legend=True,
            xlabel="Time",
            ylabel="Sea level [m]",
            responsive=True,
            legend_position="right",
            legend_cols=2,
            frame_height=fixed_height,
        )
        plot3.opts(
            title=f"Sum of constituents at {loc1}",
            show_legend=True,
            xlabel="Time",
            ylabel="Sea level [m]",
            responsive=True,
            legend_position="right",
            frame_height=fixed_height,
        )
        plot4.opts(
            title=f"Sum of constituents at {loc2}",
            show_legend=True,
            xlabel="Time",
            ylabel="Sea level [m]",
            responsive=True,
            legend_position="right",
            frame_height=fixed_height,
        )
        plot5.opts(
            title=f"Sum of constituents at {loc1}",
            show_legend=True,
            xlabel="Time",
            ylabel="Sea level [m]",
            responsive=True,
            legend_position="right",
            frame_height=fixed_height,
        )
        plot6.opts(
            title=f"Sum of constituents at {loc2}",
            show_legend=True,
            xlabel="Time",
            ylabel="Sea level [m]",
            responsive=True,
            legend_position="right",
            frame_height=fixed_height,
        )

        full_comp = [
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

        plot = (
            (plot1 + plot2 + plot3 + plot4 + plot5 + plot6)
            .cols(2)
            .opts(shared_axes=False)
        )

        return plot

    app = pn.Column(
        pn.Row(dropdown1, dropdown2, sizing_mode="stretch_width"),
        pn.Row(slider, sizing_mode="stretch_width"),
        pn.Row(checkboxes, sizing_mode="stretch_width"),
        pn.Row(hv_plot_timeseries, sizing_mode="stretch_width"),
        width_policy="max",
    )
    return app


# def plot_2timeseries_with_interactive_controls(comps, dates, tide, locs):
#     all_comp = comps

#     # Define a list of checkboxes for component selection and put them in one row
#     checkboxes = [
#         widgets.Checkbox(
#             value=(comp in ["M2", "S2"]),
#             description=comp,
#             layout=widgets.Layout(width="auto"),
#         )
#         for comp in comps
#     ]
#     checkbox_row = widgets.HBox(
#         checkboxes, layout=widgets.Layout(display="flex", flex_flow="row wrap")
#     )

#     # Plot with interactive slider and checkboxes
#     date_range_selector = widgets.SelectionRangeSlider(
#         options=[(date.strftime("%d/%m %Hh"), date) for date in dates],
#         index=(0, len(dates) - 1),
#         description="Dates",
#         orientation="horizontal",
#         layout={"width": "700px"},
#         continuous_update=False,
#         readout=True,
#     )

#     def hv_plot_timeseries(date_range, **kwargs):
#         start_date, end_date = date_range

#         # Filter selected components
#         selected_components = [comp for comp, value in kwargs.items() if value]

#         # holoviews
#         comp1 = []
#         comp2 = []

#         # Plot selected components
#         for comp in selected_components:
#             curve1 = hv.Curve(
#                 tide[locs[0]][comp.lower()][start_date:end_date], label=comp
#             )
#             curve1.opts(line_width=1.0, show_legend=True, title=locs[0])
#             comp1.append(curve1)

#             curve2 = hv.Curve(
#                 tide[locs[1]][comp.lower()][start_date:end_date], label=comp
#             )
#             curve2.opts(line_width=1.0, show_legend=True, title=locs[1])
#             comp2.append(curve2)

#         plot1 = hv.Overlay(comp1)
#         plot2 = hv.Overlay(comp2)

#         plot1.opts(
#             title=locs[0],
#             show_legend=True,
#             xlabel="Time",
#             ylabel="Sea level [m]",
#             responsive=True,
#             aspect=2,
#             legend_position="right",
#             legend_cols=2,
#         )
#         plot2.opts(
#             title=locs[1],
#             show_legend=True,
#             xlabel="Time",
#             ylabel="Sea level [m]",
#             responsive=True,
#             aspect=2,
#             legend_position="right",
#             legend_cols=2,
#         )

#         # Calculate and plot the sum of selected components
#         sum_values = [0] * len(locs)
#         for i, loc in enumerate(locs):
#             sum_values[i] = sum(
#                 tide[loc][comp.lower()][start_date:end_date]
#                 for comp in selected_components
#             )

#         sumselected1 = hv.Curve(
#             (sum_values[0].index, sum_values[0].values.flatten()),
#             label="Checked",
#         )
#         sumselected1.opts(
#             color="darkblue",
#             line_width=1.0,
#             xlabel="Time",
#             ylabel="Sea level [m]",
#             show_legend=True,
#             responsive=True,
#             aspect=2,
#         )

#         sumselected2 = hv.Curve(
#             (sum_values[1].index, sum_values[1].values.flatten()),
#             label="Checked",
#         )
#         sumselected2.opts(
#             color="darkblue",
#             line_width=1.0,
#             xlabel="Time",
#             ylabel="Sea level [m]",
#             show_legend=True,
#             responsive=True,
#             aspect=2,
#         )

#         full_comp = [
#             "EPS2",
#             "J1",
#             "K1",
#             "K2",
#             "L2",
#             "LAMBDA2",
#             "M2",
#             "M3",
#             "M4",
#             "M6",
#             "M8",
#             "MF",
#             "MKS2",
#             "MM",
#             "MN4",
#             "MS4",
#             "MSF",
#             "MSQM",
#             "MTM",
#             "MU2",
#             "N2",
#             "N4",
#             "NU2",
#             "O1",
#             "P1",
#             "Q1",
#             "R2",
#             "S1",
#             "S2",
#             "S4",
#             "SA",
#             "SSA",
#             "T2",
#         ]

#         # Calculate and plot the sum of aLL components
#         sum_all = [0] * len(locs)
#         for i, loc in enumerate(locs):
#             sum_all[i] = sum(
#                 tide[loc][comp.lower()][start_date:end_date] for comp in all_comp
#             )
#         sum_full = [0] * len(locs)
#         for i, loc in enumerate(locs):
#             sum_full[i] = sum(
#                 tide[loc][comp.lower()][start_date:end_date] for comp in full_comp
#             )

#         tot_signal1 = hv.Curve(
#             (sum_all[0].index, sum_all[0].values.flatten()),
#             label="All",
#         )
#         tot_signal1.opts(
#             color="lightblue",
#             line_width=1.0,
#             show_legend=True,
#             xlabel="Time",
#             ylabel="Sea level [m]",
#             responsive=True,
#             aspect=2,
#         )
#         real_signal1 = hv.Curve(
#             (sum_full[0].index, sum_full[0].values.flatten()),
#             label="All FES",
#         )
#         real_signal1.opts(
#             color="grey",
#             alpha=0.75,
#             line_width=0.5,
#             show_legend=True,
#             xlabel="Time",
#             ylabel="Sea level [m]",
#             responsive=True,
#             aspect=2,
#         )

#         tot_signal2 = hv.Curve(
#             (sum_all[1].index, sum_all[1].values.flatten()),
#             label="All",
#         )
#         tot_signal2.opts(
#             color="lightblue",
#             line_width=1.0,
#             show_legend=True,
#             xlabel="Time",
#             ylabel="Sea level [m]",
#             responsive=True,
#             aspect=2,
#         )
#         real_signal2 = hv.Curve(
#             (sum_full[1].index, sum_full[1].values.flatten()),
#             label="All FES",
#         )
#         real_signal2.opts(
#             color="grey",
#             alpha=0.75,
#             line_width=0.5,
#             show_legend=True,
#             xlabel="Time",
#             ylabel="Sea level [m]",
#             responsive=True,
#             aspect=2,
#         )

#         curve3 = hv.Overlay(tot_signal1 * sumselected1).opts(
#             show_legend=True,
#             legend_position="right",
#             title="Checked vs. all checked",
#         )
#         curve4 = hv.Overlay(tot_signal2 * sumselected2).opts(
#             show_legend=True,
#             legend_position="right",
#             title="Checked vs. all checked",
#         )

#         curve5 = hv.Overlay(tot_signal1 * real_signal1).opts(
#             show_legend=True,
#             legend_position="right",
#             title="All checked vs. all FES comp.",
#         )
#         curve6 = hv.Overlay(tot_signal2 * real_signal2).opts(
#             show_legend=True,
#             legend_position="right",
#             title="All checked vs. all FES comp.",
#         )

#         plot = (
#             hv.Layout(plot1 + plot2 + curve3 + curve4 + curve5 + curve6)
#             .cols(2)
#             .opts(shared_axes=True)
#         )
#         return display(plot)

#     filterwarnings("ignore", category=UserWarning)
#     filterwarnings("ignore", category=FutureWarning)

#     # Create an interactive widget with checkboxes
#     figure = widgets.interactive(
#         hv_plot_timeseries,
#         date_range=date_range_selector,
#         **{checkbox.description: checkbox for checkbox in checkboxes},
#     )

#     # Create a new container for arranging controls
#     controls = widgets.VBox([checkbox_row, figure.children[-1]])
#     controls.layout.height = "100%"
#     display(controls)
