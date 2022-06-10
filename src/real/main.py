from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from time import time
from typing import Optional
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
from obp.dataset import OpenBanditDataset
from obp.ope import RegressionModel
from obp.policy import BernoulliTS
from obp.policy import Random
from obp.types import BanditFeedback
from omegaconf import DictConfig
from ope import run_ope
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = getLogger(__name__)


registered_colors = {
    "MIPS (w/o SLOPE)": "tab:gray",
    "MIPS (w/ SLOPE)": "tab:green",
    "IPS": "tab:red",
    "DR": "tab:blue",
    "DM": "tab:purple",
    "SwitchDR": "tab:brown",
    "MRDR": "tab:cyan",
    r"DR-$\lambda$": "tab:olive",
    "DRos": "tab:pink",
}


@dataclass
class ModifiedOpenBanditDataset(OpenBanditDataset):
    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return int(self.action.max() + 1)

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset."""
        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        pos = DataFrame(self.position)
        self.action_context = (
            self.item_context.drop(columns=["item_id", "item_feature_0"], axis=1)
            .apply(LabelEncoder().fit_transform)
            .values
        )
        self.action_context = self.action_context[self.action]
        self.action_context = np.c_[self.action_context, pos]

        self.action = self.position * self.n_actions + self.action
        self.position = np.zeros_like(self.position)
        self.pscore /= 3

    def sample_bootstrap_bandit_feedback(
        self,
        sample_size: Optional[int] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ) -> BanditFeedback:

        if is_timeseries_split:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )[0]
        else:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )
        n_rounds = bandit_feedback["n_rounds"]
        if sample_size is None:
            sample_size = bandit_feedback["n_rounds"]
        else:
            check_scalar(
                sample_size,
                name="sample_size",
                target_type=(int),
                min_val=0,
                max_val=n_rounds,
            )
        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(
            np.arange(n_rounds), size=sample_size, replace=True
        )
        for key_ in [
            "action",
            "position",
            "reward",
            "pscore",
            "context",
            "action_context",
        ]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        bandit_feedback["n_rounds"] = sample_size
        return bandit_feedback


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    logger.info(f"The current working directory is {Path().cwd()}")
    start_time = time()

    # log path
    log_path = Path("./all")
    df_path = log_path / "df"
    df_path.mkdir(exist_ok=True, parents=True)
    fig_path = log_path / "fig"
    fig_path.mkdir(exist_ok=True, parents=True)

    # configurations
    sample_size = cfg.setting.sample_size
    random_state = cfg.setting.random_state
    obd_path = Path().cwd().parents[1] / "open_bandit_dataset"

    # define policies
    policy_ur = Random(
        n_actions=80,
        len_list=3,
        random_state=random_state,
    )
    policy_ts = BernoulliTS(
        n_actions=80,
        len_list=3,
        random_state=random_state,
        is_zozotown_prior=True,
        campaign="all",
    )

    # calc ground-truth policy value (on-policy)
    policy_value = ModifiedOpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy="bts", campaign="all", data_path=obd_path
    )

    # define a dataset class
    dataset = ModifiedOpenBanditDataset(
        behavior_policy="random",
        data_path=obd_path,
        campaign="all",
    )

    elapsed_prev = 0.0
    squared_error_list = []
    relative_squared_error_list = []
    for t in np.arange(cfg.setting.n_seeds):
        pi_b = policy_ur.compute_batch_action_dist(n_rounds=sample_size)
        pi_e = policy_ts.compute_batch_action_dist(n_rounds=sample_size)
        pi_e = pi_e.reshape(sample_size, 240, 1) / 3

        val_bandit_data = dataset.sample_bootstrap_bandit_feedback(
            sample_size=sample_size,
            random_state=t,
        )
        val_bandit_data["pi_b"] = pi_b.reshape(sample_size, 240, 1) / 3

        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=RandomForestClassifier(
                n_estimators=10, max_samples=0.8, random_state=12345
            ),
        )
        estimated_rewards = regression_model.fit_predict(
            context=val_bandit_data["context"],  # context; x
            action=val_bandit_data["action"],  # action; a
            reward=val_bandit_data["reward"],  # reward; r
            n_folds=2,
            random_state=12345,
        )

        squared_errors, relative_squared_errors = run_ope(
            val_bandit_data=val_bandit_data,
            action_dist_val=pi_e,
            estimated_rewards=estimated_rewards,
            estimated_rewards_mrdr=estimated_rewards,
            policy_value=policy_value,
        )
        squared_error_list.append(squared_errors)
        relative_squared_error_list.append(relative_squared_errors)

        elapsed = np.round((time() - start_time) / 60, 2)
        diff = np.round(elapsed - elapsed_prev, 2)
        logger.info(f"t={t}: {elapsed}min (diff {diff}min)")
        elapsed_prev = elapsed

    # aggregate all results
    result_df = (
        DataFrame(DataFrame(squared_error_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "se"})
    )
    result_df.reset_index(inplace=True, drop=True)
    result_df.to_csv(df_path / "result_df.csv")

    rel_result_df = (
        DataFrame(DataFrame(relative_squared_error_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "se"})
    )
    rel_result_df.reset_index(inplace=True, drop=True)
    rel_result_df.to_csv(df_path / "rel_result_df.csv")

    # plot CDFs
    estimators = result_df.est.unique().tolist()
    palette = [registered_colors[est] for est in estimators[::-1]]

    ### CDF of relative SE ###
    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    sns.ecdfplot(
        linewidth=4,
        palette=palette,
        data=rel_result_df,
        x="se",
        hue="est",
        hue_order=estimators[::-1],
        ax=ax,
    )
    # title and legend
    ax.legend(estimators, loc="upper left", fontsize=22)
    # yaxis
    ax.set_ylabel("probability", fontsize=25)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    # xaxis
    ax.set_xscale("log")
    ax.set_xlabel("relative squared errors w.r.t. IPS", fontsize=25)
    ax.tick_params(axis="x", labelsize=18)
    ax.xaxis.set_label_coords(0.5, -0.1)
    plt.savefig(fig_path / "relative_cdf.png")

    ### CDF of relative SE zoom ###
    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    sns.ecdfplot(
        linewidth=4,
        palette=palette,
        data=rel_result_df,
        x="se",
        hue="est",
        hue_order=estimators[::-1],
        ax=ax,
    )
    # title and legend
    ax.legend(estimators, loc="upper left", fontsize=22)
    # yaxis
    ax.set_ylabel("probability", fontsize=25)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    # xaxis
    ax.set_xscale("log")
    ax.set_xlim(0.09, 10)
    ax.set_xlabel("relative squared errors w.r.t. IPS", fontsize=25)
    ax.tick_params(axis="x", labelsize=18)
    ax.xaxis.set_label_coords(0.5, -0.1)
    plt.savefig(fig_path / "relative_cdf_zoom.png")


if __name__ == "__main__":
    main()
