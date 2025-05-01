import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts
from bokeh.models import Arrow, NormalHead
import pandas as pd

hv.extension("bokeh")


def tide_diagram(angle_sun, angle_moon):
    def ellipse_points(x_radius, y_radius, angle, num_points=100):
        # Parametric angles
        angles = np.linspace(0, 2 * np.pi, num_points)

        # Parametric equation of the ellipse
        x = x_radius * np.cos(angles)
        y = y_radius * np.sin(angles)

        # Rotation matrix
        cos_theta = np.cos(angle * np.pi / 180)
        sin_theta = np.sin(angle * np.pi / 180)

        # Apply the rotation matrix to each point
        x_rotated = cos_theta * x - sin_theta * y
        y_rotated = sin_theta * x + cos_theta * y

        return np.column_stack([x_rotated, y_rotated])

    dm = 2

    e_points = ellipse_points(dm, dm, angle_moon)
    earth = hv.Polygons([{"x": e_points[:, 0], "y": e_points[:, 1]}])
    earth.opts(opts.Polygons(fill_color="green", alpha=1, line_color="green"))
    text = hv.Text(0, 0, "Earth")

    if (angle_moon - angle_sun) % 180:
        title = "Angle moon: " + str(angle_moon) + ", Angle sun: " + str(angle_sun)
        tot_points = ellipse_points(dm + 0.8, dm + 0.45, angle_moon)
        w = 525
        h = 450
    else:
        title = "Angle moon: " + str(angle_moon) + ", Angle sun: " + str(angle_sun)
        tot_points = ellipse_points(dm + 1.05, dm + 0.2, angle_moon)
        w = 625
        h = 400

    total_tide = hv.Polygons(
        [{"x": tot_points[:, 0], "y": tot_points[:, 1]}], label="Total_tide"
    )
    total_tide = total_tide.opts(
        opts.Polygons(
            fill_color="royalblue", alpha=1, line_color="royalblue", show_legend=True
        )
    )

    l_points = ellipse_points(dm + 0.75, dm + 0.15, angle_moon)
    lunar_tide = hv.Polygons(
        [{"x": l_points[:, 0], "y": l_points[:, 1]}], label="Lunar tide"
    )
    lunar_tide = lunar_tide.opts(
        opts.Polygons(
            line_alpha=1,
            line_color="black",
            line_dash="dashed",
            fill_alpha=0,
            show_legend=True,
        )
    )

    s_points = ellipse_points(dm + 0.3, dm + 0.05, angle_sun)
    solar_tide = hv.Polygons(
        [{"x": s_points[:, 0], "y": s_points[:, 1]}], label="Solar tide"
    )
    solar_tide = solar_tide.opts(
        opts.Polygons(
            line_alpha=1,
            line_color="orange",
            line_dash="dashed",
            fill_alpha=0,
            show_legend=True,
        )
    )

    mplot = total_tide * earth * text * lunar_tide * solar_tide
    return mplot.opts(
        width=w,
        height=h,
        title=title,
        show_legend=True,
        legend_position="right",
        xaxis="bare",
        yaxis="bare",
    )


def plot_grav_pull():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # Adding outer and inner circles
    circ_out = mpatches.Circle((0, 0), 0.3, fill=True, color="grey", linewidth=2)
    circ_in = mpatches.Circle((0, 0), 0.275, fill=True, color="lightgrey")
    circ_c = mpatches.Circle((0, 0), 0.005, fill=True, color="black")
    axs[0].add_patch(circ_out)
    axs[0].add_patch(circ_in)
    axs[1].add_patch(mpatches.Circle((0, 0), 0.3, fill=True, color="grey", linewidth=2))
    axs[1].add_patch(mpatches.Circle((0, 0), 0.275, fill=True, color="lightgrey"))
    axs[1].add_patch(circ_c)

    # Coordinates and force components. The first two values are the coordinates of the position.
    # The 3rd and 4th value are the x/y components of the GP at each location (not to scale)
    positions = {
        "X": [0, 0, 0.2, 0],
        "A": [0, 0.3, 0.2, -0.025],
        "B": [
            0.3 * np.cos(45 * np.pi / 180),
            0.3 * np.cos(45 * np.pi / 180),
            0.25,
            -0.015,
        ],
        "C": [0.3, 0, 0.25, 0],
        "D": [
            0.3 * np.cos(45 * np.pi / 180),
            -0.3 * np.cos(45 * np.pi / 180),
            0.25,
            0.015,
        ],
        "E": [0, -0.3, 0.2, 0.025],
        "F": [
            -0.3 * np.cos(45 * np.pi / 180),
            -0.3 * np.cos(45 * np.pi / 180),
            0.15,
            0.01,
        ],
        "G": [-0.3, 0, 0.15, 0],
        "H": [
            -0.3 * np.cos(45 * np.pi / 180),
            0.3 * np.cos(45 * np.pi / 180),
            0.15,
            -0.01,
        ],
    }

    # Loop through each position
    for pos in positions:
        # Center of the Earth
        if pos == "X":
            axs[0].add_patch(mpatches.Arrow(*positions[pos], width=0.025, color="k"))

        # Other positions
        else:
            # Gravitational pull (GP)
            axs[0].add_patch(
                mpatches.Arrow(
                    *positions[pos][0:2],
                    0.2,
                    0,
                    width=0.025,
                    edgecolor="k",
                    facecolor="None",
                )
            )  # GP at center of Earth
            axs[0].add_patch(
                mpatches.Arrow(*positions[pos], width=0.025, color="C3")
            )  # GP at each individual location

            # Differential pull
            diffpull = list(
                2 * (np.array(positions[pos][2:]) - np.array(positions["X"][2:]))
            )  # Subtract th GP at the center from the other locations
            axs[1].add_patch(
                mpatches.Arrow(*positions[pos][0:2], *diffpull, width=0.025, color="C2")
            )

    # Adding center and text
    titles = ["Gravitational pull", "Differential pull"]
    for i in range(len(axs)):
        for pos in positions:
            coords = [x + 0.025 for x in positions[pos][0:2]]
            axs[i].text(*coords, pos, fontsize=12, ha="center", va="center")
        axs[i].set_aspect("equal")
        axs[i].axis("on")
        axs[i].set_xlim(-0.6, 0.6)
        axs[i].set_ylim(-0.6, 0.6)
        axs[i].set_title(titles[i])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
