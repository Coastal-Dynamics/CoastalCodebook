from pathlib import Path

import panel as pn
import holoviews as hv
import hvplot.pandas
import numpy as np


def plot_brazilian_coast(gdf):
    """
    Show a map of the Brazilian coast, with data points.
    """

    # below we build the widget
    title_bar = pn.Row(
        pn.pane.Markdown(
            "## Brazilian Coast",
            styles={"color": "black"},
            sizing_mode="fixed",
            margin=(10, 5, 10, 15),
        ),
        align="center",
    )

    # dropdown menu of coasts
    options = {
        "Pará": ["Princesa", "Atalaia", "Ajuruteua"],
        "Maranhão": ["São Luís (Calhau)"],
        "Rio de Janeiro": ["Marambaia"],
        "Santa Catarina": [
            "Campo Bom",
            "Laguna",
            "Enseada de Pinheira",
            "Praia do Moçambique",
            "Tijucas",
            "Balneário Camboriú",
            "Do Ubatuba",
            "Barra Velha",
        ],
    }
    coasts_dropdown = pn.widgets.Select(
        name="Coast select (grouped by state)", groups=options, value="Campo Bom"
    )

    @pn.depends(coasts_dropdown.param.value)
    def plot_coast(name, plot_size=0.04):
        beach = gdf[gdf["Label"] == name].copy()
        beach = beach.astype({"Omega [-] ": str})
        lat, lon = beach[["Latitude", "Longitude"]].values.flatten()
        lat, lon = np.float64(lat), np.float64(lon)

        points = gdf.hvplot.points(
            geo=True,
            tiles="ESRI",
            ylabel="Latitude [deg]",
            xlabel="Longitude [deg]",
            xlim=(lon - plot_size / 2, lon + plot_size / 2),
            ylim=(lat - plot_size / 2, lat + plot_size / 2),
            tools=["tap"],
            hover_cols=[
                "Label",
                "State",
                "Region",
                "Longitude",
                "Latitude",
                "Hs [m]",
                "T [s]",
                "MTR [m]",
                "RTR [-]",
                "Beach slope [degrees]",
                "D [mm]",
                "w_s [m/s]",
                "Hb [m]",
                "Omega [-] ",
            ],
            c="State",
            cmap="Accent",
            line_color="black",
            size=300,
        )

        plot = (points).opts(width=1200, height=800, tools=["wheel_zoom"])

        return plot

    app = pn.Column(
        pn.Row(title_bar, align="center"),
        pn.Row(coasts_dropdown, align="center"),
        pn.Row(plot_coast, align="center"),
    )

    return app


def fig413(gdf):
    # Make the plot with background
    fp = Path("../images/5_coastal_impact_beach_states/5_fig413_bg.jpg")

    bg = hv.RGB.load_image(fp, bounds=(0, 0, 2.5, 6)).opts(alpha=0.5)

    # create the points
    points = gdf.hvplot.points(
        x="Hs [m]",
        y="MTR [m]",
        by="Label",
        size=100,
        cmap="Accent",
        line_color="black",
        hover_cols=["Label", "State", "Hs [m]", "MTR [m]", "RTR [-]"],
        # hover_cols=list(df.columns), # uncomment to show all columns
    )

    fig = (bg * points).opts(
        width=700,
        height=600,
        show_grid=True,
        active_tools=[],
        toolbar=None,
        xlabel="mean wave height [m]",
        ylabel="mean tidal range [m]",
        xlim=(0, 2.5),
        ylim=(0, 6),
        show_legend=True,
    )

    return fig


def fig710(df):
    title_para = pn.pane.Markdown("**Pará**", align=("start", "end"))
    title_saoluis = pn.pane.Markdown("**Maranhão**", align=("start", "end"))
    title_rio = pn.pane.Markdown("**Rio de Janeiro**", align=("start", "end"))
    title_santacat = pn.pane.Markdown("**Santa Catarina**", align=("start", "end"))

    beach_names_para = list(df[df["State"] == "Pará"].Label.values)
    beach_names_saoluis = list(df[df["State"] == "Maranhão"].Label.values)
    beach_names_rio = list(df[df["State"] == "Rio de Janeiro"].Label.values)
    beach_names_santacat = list(df[df["State"] == "Santa Catarina"].Label.values)
    beach_names_santacat1 = beach_names_santacat[: len(beach_names_santacat) // 2]
    beach_names_santacat2 = beach_names_santacat[len(beach_names_santacat) // 2 :]

    checkboxes_para = pn.widgets.CheckBoxGroup(
        value=beach_names_para,
        options=beach_names_para,
        inline=False,
        align=("start", "start"),
    )
    # checkboxes_para = pn.widgets.CheckBoxGroup(name="Select beaches to include in plot", inline=True, value=beach_names_para, options=beach_names_para)
    checkboxes_saoluis = pn.widgets.CheckBoxGroup(
        name="Select beaches to include in plot",
        value=beach_names_saoluis,
        options=beach_names_saoluis,
        inline=False,
        align=("start", "start"),
    )
    checkboxes_rio = pn.widgets.CheckBoxGroup(
        name="Select beaches to include in plot",
        value=beach_names_rio,
        options=beach_names_rio,
        inline=False,
        align=("start", "start"),
    )
    checkboxes_santacat1 = pn.widgets.CheckBoxGroup(
        name="Select beaches to include in plot",
        value=beach_names_santacat1,
        options=beach_names_santacat1,
        inline=False,
        align=("start", "start"),
    )
    checkboxes_santacat2 = pn.widgets.CheckBoxGroup(
        name="Select beaches to include in plot",
        value=beach_names_santacat2,
        options=beach_names_santacat2,
        inline=False,
        align=("start", "start"),
    )

    @pn.depends(
        checkboxes_para.param.value,
        checkboxes_saoluis.param.value,
        checkboxes_rio.param.value,
        checkboxes_santacat1.param.value,
        checkboxes_santacat2.param.value,
    )
    def create_app(cb1, cb2, cb3, cb4, cb5):
        cbs = list(cb1) + list(cb2) + list(cb3) + list(cb4) + list(cb5)

        df_mod = df[df["Label"].isin(cbs)]

        TOOLTIPS = [
            ("Label", "@Label"),
            ("State", "@State"),
            ("Region", "@Region"),
            ("Beach slope [degrees]", "@Beach slope [degrees]"),
            ("D [mm]", "@D [mm]"),
            ("RTR [-]", "@RTR [-]"),
            ("Omega [-]", "@Omega [-] "),
            ("--------", "--------"),
        ]

        # points for other states
        points = df_mod.hvplot.points(
            x="Beach slope [degrees]",
            y="D [mm]",
            size=100,
            c="Label",
            cmap="Accent",
            hover_cols=[
                "State",
                "Label",
                "Region",
                "Beach slope [degrees]",
                "D [mm]",
                "RTR [-]",
                "Omega [-] ",
            ],
            line_color="black",
            responsive=True,
            hover_tooltips=TOOLTIPS,
        )

        # plot horizontal and vertical lines
        hlines = hv.HLine(0.25).opts(color="lightgrey") * hv.HLine(0.5).opts(
            color="lightgrey"
        )
        vlines = (
            hv.VLine(3.5).opts(color="lightgrey")
            * hv.VLine(8.5).opts(color="lightgrey")
        ).opts(border_line_color="lightgrey")

        # plot beach state labels
        classify_labels = (
            hv.Text(1.75, 0.95, "dissipative", fontsize=10)
            * hv.Text(6, 0.95, "intermediate", fontsize=10)
            * hv.Text(13, 0.95, "reflective", fontsize=10)
        )

        # plot grain size labels
        grain_labels = (
            hv.Text(15.8, 0.75, "coarse sand", fontsize=10, halign="right")
            * hv.Text(15.8, 0.375, "medium sand", fontsize=10, halign="right")
            * hv.Text(15.8, 0.125, "fine sand", fontsize=10, halign="right")
        )

        # combine all figure components
        fig = (points * hlines * vlines * classify_labels * grain_labels).opts(
            xlabel="Slope [degrees]",
            ylabel="Mean grain size [mm]",
            ylim=(0, 1),
            xlim=(0, 16),
            tools=["pan", "wheel_zoom"],
            toolbar=None,
            frame_height=300,
        )

        return fig

    app = pn.Column(
        pn.Row(
            pn.Column(width=50),
            pn.Column(title_para, checkboxes_para),
            pn.Column(width=20),
            pn.Column(title_saoluis, checkboxes_saoluis),
            pn.Column(width=20),
            pn.Column(title_rio, checkboxes_rio),
            pn.Column(width=20),
            pn.Column(
                title_santacat,
                pn.Row(
                    checkboxes_santacat1,
                    checkboxes_santacat2,
                    sizing_mode="stretch_width",
                ),
            ),
            sizing_mode="stretch_width",
        ),
        pn.Row(create_app, sizing_mode="stretch_width"),
        width_policy="max",
    )

    return app
