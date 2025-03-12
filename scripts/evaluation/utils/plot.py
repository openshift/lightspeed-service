"""Plot score for evaluation."""

import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS


def plot_score(results_df, score_name, out_file_path=None):
    """Plot score."""
    _, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel(score_name)
    ax.set_xlim(0, 1)

    ax.axvline(x=0.25, linewidth=2, color="red")
    ax.axvline(x=0.5, linewidth=2, color="orange")
    ax.axvline(x=0.75, linewidth=2, color="green")

    ax.axvspan(0, 0.25, facecolor="gainsboro")
    ax.axvspan(0.25, 0.5, facecolor="mistyrose")
    ax.axvspan(0.5, 0.75, facecolor="lightyellow")
    ax.axvspan(0.75, 1.0, facecolor="lightgreen")

    ax.grid(True)

    labels = results_df.columns
    bplot = ax.boxplot(
        results_df.fillna(0),
        patch_artist=True,
        sym=".",
        widths=0.5,
        # tick_labels=labels,
        labels=labels,
        vert=False,
    )
    colors = list(BASE_COLORS.keys())[: len(labels)]
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    plt.yticks(rotation=45)

    if out_file_path:
        plt.savefig(out_file_path)
