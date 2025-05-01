import math
import os
from pathlib import Path
import sys
from warnings import filterwarnings


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import ipywidgets as widgets
from IPython.display import display
import holoviews as hv
from holoviews import opts
import pandas as pd
import cartopy.crs as ccrs
import pickle
import uptide
from datetime import datetime, timedelta
import numpy as np

hv.extension("bokeh")


def tidal_characters(F_data):
    # Define the categories and corresponding colors
    categories = {
        "Diurnal": {"min": 3, "max": float("inf"), "color": "#ad4123"},
        "Mixed, mainly diurnal": {"min": 1.5, "max": 2.9, "color": "#e7541e"},
        "Mixed, mainly semidiurnal": {"min": 0.25, "max": 1.49, "color": "#f4a030"},
        "Semidiurnal": {"min": 0, "max": 0.249, "color": "#f8ce97"},
    }

    # Create a figure and axis
    fig, ax = plt.subplots(
        figsize=(15, 13),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0.0)},
    )
    ax.set_global()

    # Plot the scatter points with specific colors for each category
    legend_patches = []
    for category, values in categories.items():
        subset_data = F_data[
            (F_data["F"] >= values["min"]) & (F_data["F"] <= values["max"])
        ]
        if not subset_data.empty:
            scatter = ax.scatter(
                subset_data.index.get_level_values("lon").values,
                subset_data.index.get_level_values("lat").values,
                s=1,
                color=values["color"],
                label=category,
                transform=ccrs.PlateCarree(),
            )
            legend_patches.append(
                Patch(color=scatter.get_facecolor()[0], label=category)
            )

    # Add markers for specific locations - here you can edit the code if you are wondering for a specific location
    locs = {
        "Scheveningen": [4.25, 52.125],  # lon, lat
        "Galveston": [-94.6875, 29.25],
        "Jakarta": [106.8125, -6.0625],
        "Valparaiso": [-71.625, -33],
    }
    for loc, coordinates in locs.items():
        lon, lat = coordinates
        ax.scatter(
            lon, lat, color="black", s=10, transform=ccrs.PlateCarree(), zorder=4
        )
        ax.text(
            lon - 25,
            lat + 3,
            loc,
            color="black",
            fontsize=12,
            fontweight="bold",
            transform=ccrs.PlateCarree(),
        )

    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        fontsize=12,
    )
    ax.coastlines(resolution="110m", color="black", linewidth=0.5)

    # fig.savefig(Path("./database/3_tides_nonlinearity/global_contour.png"))

    return plt.show()


def FES_tidal_signal(DATA_DIR, start_date, end_date):
    locs = ["Galveston", "Jakarta", "Scheveningen", "Valparaiso"]

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

    for i, loc in enumerate(locs):
        for comp in comps:
            fp = os.path.join(DATA_DIR, "fes2014_amp_ph", ("02_" + comp.lower() + ".p"))
            tide[comp.lower()] = pd.read_pickle(fp)
            amplitudes[comp.lower()] = tide[comp.lower()]["amplitude"][f"{loc.lower()}"]
            phases[comp.lower()] = tide[comp.lower()]["phase"][f"{loc.lower()}"]

        initial_time = datetime(2000, 1, 1, 12, 0, 0)

        tidal_signal = uptide.Tides(
            comps
        )  # select which constituents to use, we will use all
        tidal_signal.set_initial_time(initial_time)

        amp = [amplitudes[comp.lower()] for comp in comps]
        pha = [
            math.radians(phases[comp.lower()]) for comp in comps
        ]  # phase (in radians!)
        t = np.arange(
            0, 365 * 1 * 24 * 3600, 900  # DDD * Y * hh * ssss
        )  # seconds since initial time, 40 years since 1977, 15min frequency
        dates = np.array(
            [initial_time + timedelta(seconds=int(s)) for s in t]
        )  # so that we have datetime on x-axis
        eta = tidal_signal.from_amplitude_phase(amp, pha, t)  # calculate the signal
        eta_df = pd.DataFrame({"eta": eta}, index=dates)

        filtered_tide = eta_df[start_date:end_date]
        ftide = hv.Curve(
            (
                eta_df[start_date:end_date].index,
                eta_df[start_date:end_date]["eta"].values / 100,
            ),
            #            label=f"{loc} FES signal",
            label="total FES",
        )
        ftide.opts(
            title=f"Location {i+1}",
            color="grey",
            show_legend=True,
            aspect=2,
            responsive=True,
            line_width=0.7,
            ylabel="Elevation [m]",
            xlabel="Time",
        )
        if loc == "Galveston":
            galveston = ftide
        if loc == "Jakarta":
            jakarta = ftide
        if loc == "Scheveningen":
            scheveningen = ftide
        if loc == "Valparaiso":
            valparaiso = ftide
    total_signal = hv.Layout(galveston + jakarta + scheveningen + valparaiso).cols(2)

    return total_signal, scheveningen, galveston, jakarta, valparaiso


data_dir = Path("../database/2_wind_waves_tides/")

scheveningen_fp = os.path.join(data_dir, "tide_scheveningen.p")
galveston_fp = os.path.join(data_dir, "tide_galveston.p")
jakarta_fp = os.path.join(data_dir, "tide_jakarta.p")
valparaiso_fp = os.path.join(data_dir, "tide_valparaiso.p")

with open(scheveningen_fp, "rb") as pickle_file:
    scheveningen_f = pickle.load(pickle_file)
with open(galveston_fp, "rb") as pickle_file:
    galveston_f = pickle.load(pickle_file)
with open(jakarta_fp, "rb") as pickle_file:
    jakarta_f = pickle.load(pickle_file)
with open(valparaiso_fp, "rb") as pickle_file:
    valparaiso_f = pickle.load(pickle_file)

tide = {
    "Scheveningen": scheveningen_f,
    "Valparaiso": valparaiso_f,
    "Jakarta": jakarta_f,
    "Galveston": galveston_f,
}

dates_range = np.array(
    [
        datetime(2000, 1, 1, 0, 0, 0) + timedelta(seconds=item * 3600)
        for item in range(24 * 365)  # 1 year
    ]
)

comps = [
    "M2",
    "S2",
    "N2",
    "K2",  # semi-diurnal
    "K1",
    "O1",
    "P1",
    "Q1",  # diurnal
    "MF",
    "MM",
    "SSA",  # long period
    "M4",
    "M6",
    "S4",
    "MN4",
]

locas = ["Scheveningen", "Valparaiso", "Jakarta", "Galveston"]

scheveningen = []
valparaiso = []
jakarta = []
galveston = []

for comp in comps:
    shev = tide["Scheveningen"][comp.lower()][dates_range[0] : dates_range[-1]] / 100
    valp = tide["Valparaiso"][comp.lower()][dates_range[0] : dates_range[-1]] / 100
    jaka = tide["Jakarta"][comp.lower()][dates_range[0] : dates_range[-1]] / 100
    galv = tide["Galveston"][comp.lower()][dates_range[0] : dates_range[-1]] / 100

    scheveningen.append(shev)
    valparaiso.append(valp)
    jakarta.append(jaka)
    galveston.append(galv)


def plot_4timeseries_with_interactive_controls(
    locs, comps, start_date, end_date, FES1, FES2, FES3, FES4
):
    days = end_date - start_date
    dates = np.array(
        [start_date + timedelta(seconds=item * 3600) for item in range(24 * days.days)]
    )

    # locs = ["Scheveningen", "Valparaiso", "Jakarta", "Galveston"]

    # Define a list of checkboxes for component selection and put them in one row
    checkboxes = [
        widgets.Checkbox(
            value=(comp in ["M2", "S2", "K1", "O1"]),
            description=comp,
            layout=widgets.Layout(width="auto"),
        )
        for comp in comps
    ]
    # ["N2", "K2", "K1", "O1", "P1", "Q1"]
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
        Scheveningen = FES1
        Valparaiso = FES4
        Jakarta = FES3
        Galveston = FES2

        # Filter selected components
        selected_components = [comp for comp, value in kwargs.items() if value]
        components = [i for i in range(len(comps)) if comps[i] in selected_components]

        # Plot the first location with the selected components
        location1 = globals()[locs[0].lower()]
        curves1 = [
            hv.Curve(location1[comp][start_date:end_date], label=comps[comp]).opts(
                line_width=0.5, show_legend=True
            )
            for comp in components
        ]
        figure1 = hv.Overlay(curves1)
        figure1.opts(
            title=locs[0],
            xlabel="Time",
            ylabel="Sea level [m]",
            legend_cols=2,
            show_legend=True,
            aspect=2,
            responsive=True,
            legend_position="right",
        )

        # Plot the second location with the selected components
        location2 = globals()[locs[1].lower()]
        curves2 = [
            hv.Curve(location2[comp][start_date:end_date], label=comps[comp]).opts(
                line_width=0.5, show_legend=True
            )
            for comp in components
        ]
        figure2 = hv.Overlay(curves2)
        figure2.opts(
            title=locs[1],
            xlabel="Time",
            ylabel="Sea level [m]",
            legend_cols=2,
            show_legend=True,
            aspect=2,
            responsive=True,
            legend_position="right",
        )

        # Calculate and plot the sum
        sum_loc1 = sum([location1[comp][start_date:end_date] for comp in components])
        sum_loc2 = sum([location2[comp][start_date:end_date] for comp in components])
        curves3 = hv.Curve(sum_loc1, label="Sum sel.")
        curves3.opts(
            color="red",
            line_width=0.5,
            xlabel="Time",
            ylabel="Sea level [m]",
            show_legend=True,
            aspect=2,
            responsive=True,
        )
        fes_loc1 = locals()[locs[0]]
        figure3 = hv.Overlay(fes_loc1 * curves3)
        figure3.opts(
            title="FES: sum of selected components and total signal",
            xlabel="Time",
            ylabel="Sea level [m]",
            show_legend=True,
            aspect=2,
            responsive=True,
            legend_position="right",
        )

        curves4 = hv.Curve(sum_loc2, label="Sum sel.")
        curves4.opts(
            color="red",
            line_width=0.5,
            xlabel="Time",
            ylabel="Sea level [m]",
            show_legend=True,
            aspect=2,
            responsive=True,
        )
        fes_loc2 = locals()[locs[1]]
        figure4 = hv.Overlay(fes_loc2 * curves4)
        figure4.opts(
            title="FES: sum of selected components and total signal",
            xlabel="Time",
            ylabel="Sea level [m]",
            show_legend=True,
            aspect=2,
            responsive=True,
            legend_position="right",
        )

        final = (
            hv.Layout(figure1 + figure2 + figure3 + figure4)
            .cols(2)
            .opts(width=1000, height=3000)
        )
        filterwarnings("ignore", category=FutureWarning)
        return display(final)

    filterwarnings("ignore", category=FutureWarning)
    filterwarnings("ignore", category=UserWarning)

    # Create an interactive widget with checkboxes
    figure = widgets.interactive(
        hv_plot_timeseries,
        date_range=date_range_selector,
        **{checkbox.description: checkbox for checkbox in checkboxes},
    )

    # Create a new container for arranging controls
    controls = widgets.VBox([checkbox_row, figure.children[-1]])
    controls.layout.height = "100%"
    display(controls)
