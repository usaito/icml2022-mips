from logging import getLogger
from pathlib import Path
from time import time
import warnings

import hydra
import numpy as np
from obp.dataset import linear_reward_function
from obp.dataset import SyntheticBanditDatasetWithActionEmbeds
from obp.ope import RegressionModel
from omegaconf import DictConfig
from ope import run_ope
import pandas as pd
from pandas import DataFrame
from plots import plot_line
from policy import gen_eps_greedy
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    logger.info(f"The current working directory is {Path().cwd()}")
    start_time = time()

    # log path
    log_path = Path("./varying_n_def_actions")
    df_path = log_path / "df"
    df_path.mkdir(exist_ok=True, parents=True)
    random_state = cfg.setting.random_state

    elapsed_prev = 0.0
    result_df_list = []
    for n_def_action in cfg.setting.n_def_actions_list:
        estimated_policy_value_list = []
        ## define a dataset class
        dataset = SyntheticBanditDatasetWithActionEmbeds(
            n_actions=cfg.setting.n_actions,
            dim_context=cfg.setting.dim_context,
            beta=cfg.setting.beta,
            reward_type="continuous",
            n_cat_per_dim=cfg.setting.n_cat_per_dim,
            latent_param_mat_dim=cfg.setting.latent_param_mat_dim,
            n_cat_dim=cfg.setting.n_cat_dim,
            n_unobserved_cat_dim=cfg.setting.n_unobserved_cat_dim,
            n_irrelevant_cat_dim=cfg.setting.n_irrelevant_cat_dim,
            n_deficient_actions=int(cfg.setting.n_actions * n_def_action),
            reward_function=linear_reward_function,
            reward_std=cfg.setting.reward_std,
            random_state=random_state,
        )
        ### test bandit data is used to approximate the ground-truth policy value
        test_bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds=cfg.setting.n_test_data
        )
        action_dist_test = gen_eps_greedy(
            expected_reward=test_bandit_data["expected_reward"],
            is_optimal=cfg.setting.is_optimal,
            eps=cfg.setting.eps,
        )
        policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=test_bandit_data["expected_reward"],
            action_dist=action_dist_test,
        )

        for _ in range(cfg.setting.n_seeds):
            ## generate validation data
            val_bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=cfg.setting.n_val_data,
            )

            ## make decisions on validation data
            action_dist_val = gen_eps_greedy(
                expected_reward=val_bandit_data["expected_reward"],
                is_optimal=cfg.setting.is_optimal,
                eps=cfg.setting.eps,
            )

            ## OPE using validation data
            reg_model = RegressionModel(
                n_actions=dataset.n_actions,
                action_context=val_bandit_data["action_context"],
                base_model=RandomForestRegressor(
                    n_estimators=10,
                    max_samples=0.8,
                    random_state=random_state + _,
                ),
            )
            estimated_rewards = reg_model.fit_predict(
                context=val_bandit_data["context"],  # context; x
                action=val_bandit_data["action"],  # action; a
                reward=val_bandit_data["reward"],  # reward; r
                n_folds=2,
                random_state=random_state + _,
            )
            estimated_policy_values = run_ope(
                val_bandit_data=val_bandit_data,
                action_dist_val=action_dist_val,
                estimated_rewards=estimated_rewards,
            )
            estimated_policy_value_list.append(estimated_policy_values)

        ## summarize results
        result_df = (
            DataFrame(DataFrame(estimated_policy_value_list).stack())
            .reset_index(1)
            .rename(columns={"level_1": "est", 0: "value"})
        )
        result_df["n_def_actions"] = int(cfg.setting.n_actions * n_def_action)
        result_df["se"] = (result_df.value - policy_value) ** 2
        result_df["bias"] = 0
        result_df["variance"] = 0
        sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
        for est_ in sample_mean["est"]:
            estimates = result_df.loc[result_df["est"] == est_, "value"].values
            mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
            mean_estimates = np.ones_like(estimates) * mean_estimates
            result_df.loc[result_df["est"] == est_, "bias"] = (
                policy_value - mean_estimates
            ) ** 2
            result_df.loc[result_df["est"] == est_, "variance"] = (
                estimates - mean_estimates
            ) ** 2
        result_df_list.append(result_df)

        elapsed = np.round((time() - start_time) / 60, 2)
        diff = np.round(elapsed - elapsed_prev, 2)
        logger.info(f"n_def_action={n_def_action}: {elapsed}min (diff {diff}min)")
        elapsed_prev = elapsed

    # aggregate all results
    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv(df_path / "result_df.csv")

    plot_line(
        result_df=result_df,
        log_path=log_path,
        embed_selection=cfg.setting.embed_selection,
        x="n_def_actions",
        xlabel=r"number of deficient actions",
        xticklabels=cfg.setting.n_def_actions_list,
    )


if __name__ == "__main__":
    main()
