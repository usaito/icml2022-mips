import matplotlib.pyplot as plt
import seaborn as sns


registered_colors = {
    "MIPS": "tab:gray",
    "MIPS (true)": "tab:orange",
    "MIPS (w/ SLOPE)": "tab:green",
    "IPS": "tab:red",
    "DR": "tab:blue",
    "DM": "tab:purple",
    "SwitchDR": "tab:brown",
    r"DR-$\lambda$": "tab:olive",
    "DRos": "tab:pink",
}


def plot_line(
    result_df,
    log_path,
    embed_selection,
    x,
    xlabel,
    xticklabels,
) -> None:
    plt.style.use("ggplot")
    query_list = [
        "(est == 'IPS' or est == 'DR' or est == 'DM' or est == 'MIPS' or est == 'MIPS (true)')",
        "(est != 'MIPS (slope)')",
    ]
    legend_list = [
        ["IPS", "DR", "DM", "MIPS", "MIPS (true)"],
        [
            "IPS",
            "DR",
            "DM",
            "SwitchDR",
            "DRos",
            r"DR-$\lambda$",
            "MIPS",
            "MIPS (true)",
        ],
    ]
    suffix_list = ["main", "all"]
    if embed_selection is True:
        query_list = [
            "(est == 'MIPS (true)' or est == 'MIPS (slope)')",
        ]
        legend_list = [
            ["MIPS (true)", "MIPS (w/ SLOPE)"],
        ]
        suffix_list = ["slope"]

    for query, legend, dir_ in zip(query_list, legend_list, suffix_list):
        line_path = log_path / "fig" / dir_
        line_path.mkdir(exist_ok=True, parents=True)
        palette = [registered_colors[est] for est in legend]

        ### MSE ###
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=5,
            marker="o",
            markersize=8,
            markers=True,
            x=x,
            y="se",
            hue="est",
            ax=ax,
            palette=palette,
            data=result_df.query(query),
        )
        # title and legend
        ax.legend(
            legend,
            fontsize=25,
        )
        # yaxis
        ax.set_yscale("log")
        ax.set_ylabel("mean squared error (MSE)", fontsize=25)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # xaxis
        if x in ["n_action", "n_val_data"]:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(line_path / "mse.png")

        ### MSE ###
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=5,
            legend=False,
            marker="o",
            markersize=8,
            markers=True,
            x=x,
            y="se",
            hue="est",
            ax=ax,
            palette=palette,
            data=result_df.query(query),
        )
        # yaxis
        ax.set_yscale("log")
        ax.set_ylabel("mean squared error (MSE)", fontsize=25)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # xaxis
        if x in ["n_action", "n_val_data"]:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(line_path / "mse_no_legend.png")

        ### MSE ###
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=5,
            marker="o",
            markersize=8,
            markers=True,
            x=x,
            y="se",
            hue="est",
            palette=palette,
            ax=ax,
            data=result_df.query(query),
        )
        # title and legend
        ax.legend(
            legend,
            fontsize=25,
        )
        # yaxis
        ax.set_ylabel("mean squared error (MSE)", fontsize=25)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # xaxis
        if x in ["n_action", "n_val_data"]:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(line_path / "mse_no_log.png")

        ### MSE ###
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=5,
            marker="o",
            markersize=8,
            markers=True,
            x=x,
            y="se",
            hue="est",
            palette=palette,
            ax=ax,
            legend=False,
            data=result_df.query(query),
        )
        # yaxis
        ax.set_ylabel("mean squared error (MSE)", fontsize=25)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # xaxis
        if x in ["n_action", "n_val_data"]:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(line_path / "mse_no_log_no_legend.png")

        ### Bias ###
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=6,
            marker="o",
            markersize=8,
            markers=True,
            x=x,
            y="bias",
            hue="est",
            palette=palette,
            ax=ax,
            ci=None,
            data=result_df.query(query),
        )
        # title and legend
        ax.legend(
            legend,
            fontsize=25,
        )
        # yaxis
        ax.set_ylabel("squared bias", fontsize=25)
        ax.tick_params(axis="y", labelsize=14)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # xaxis
        if x in ["n_action", "n_val_data"]:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(line_path / "bias_no_log.png")

        ### Variance ###
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=6,
            marker="o",
            markersize=8,
            markers=True,
            x=x,
            y="variance",
            hue="est",
            palette=palette,
            ax=ax,
            ci=None,
            data=result_df.query(query),
        )
        # title and legend
        ax.legend(
            legend,
            fontsize=25,
        )
        # yaxis
        ax.set_yscale("log")
        ax.set_ylabel("variance", fontsize=25)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # xaxis
        if x in ["n_action", "n_val_data"]:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(line_path / "variance.png")

        ### Variance ###
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=6,
            legend=False,
            marker="o",
            markersize=8,
            markers=True,
            x=x,
            y="variance",
            hue="est",
            palette=palette,
            ax=ax,
            ci=None,
            data=result_df.query(query),
        )
        # yaxis
        ax.set_yscale("log")
        ax.set_ylabel("variance", fontsize=25)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # xaxis
        if x in ["n_action", "n_val_data"]:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(line_path / "variance_no_legend.png")

        ### Variance ###
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=6,
            marker="o",
            markersize=8,
            markers=True,
            x=x,
            y="variance",
            hue="est",
            palette=palette,
            ax=ax,
            ci=None,
            data=result_df.query(query),
        )
        # title and legend
        ax.legend(
            legend,
            fontsize=25,
        )
        # yaxis
        ax.set_ylabel("variance", fontsize=25)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # xaxis
        if x in ["n_action", "n_val_data"]:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(line_path / "variance_no_log.png")

        ### Variance ###
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=6,
            marker="o",
            markersize=8,
            markers=True,
            x=x,
            y="variance",
            hue="est",
            palette=palette,
            ax=ax,
            ci=None,
            legend=False,
            data=result_df.query(query),
        )
        # yaxis
        ax.set_ylabel("variance", fontsize=25)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # xaxis
        if x in ["n_action", "n_val_data"]:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(line_path / "variance_no_log_no_legend.png")
