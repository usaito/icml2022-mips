import itertools
from typing import Dict
from typing import Optional

import numpy as np
from obp.ope import BaseOffPolicyEstimator
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import DoublyRobustWithShrinkageTuning as DRos
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import OffPolicyEvaluation
from obp.ope import SubGaussianDoublyRobustTuning as SGDR
from obp.ope import SwitchDoublyRobustTuning as SwitchDR
from obp.utils import check_array
from scipy import stats
from sklearn.naive_bayes import CategoricalNB


class MIPS(BaseOffPolicyEstimator):
    def _estimate_round_rewards(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_emb: np.ndarray,
        action_dist_b: np.ndarray,
        action_dist_e: np.ndarray,
        position: Optional[np.ndarray] = None,
        n_actions: Optional[int] = None,
        delta: float = 0.05,
        with_cnf: bool = False,
        **kwargs,
    ) -> np.ndarray:
        n = reward.shape[0]
        w_x_e = self._estimate_w_x_e(
            context=context,
            action=action,
            action_emb=action_emb,
            pi_e=action_dist_e[:, :, 0],
            pi_b=action_dist_b[:, :, 0],
            n_actions=n_actions,
        )

        if with_cnf:
            r_hat = reward * w_x_e
            cnf = np.sqrt(np.var(r_hat) / (n - 1))
            cnf *= stats.t.ppf(1.0 - (delta / 2), n - 1)

            return r_hat.mean(), cnf

        return reward * w_x_e

    def _estimate_w_x_e(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_emb: np.ndarray,
        pi_b: np.ndarray,
        pi_e: np.ndarray,
        n_actions: int,
    ) -> np.ndarray:

        n = action.shape[0]
        realized_actions = np.unique(action)
        w_x_a = pi_e / pi_b
        w_x_a = np.where(w_x_a < np.inf, w_x_a, 0)
        p_a_e_model = CategoricalNB()
        p_a_e_model.fit(action_emb, action)
        p_a_e = np.zeros((n, n_actions))
        p_a_e[:, realized_actions] = p_a_e_model.predict_proba(action_emb)
        w_x_e = (w_x_a * p_a_e).sum(1)

        return w_x_e

    def estimate_policy_value(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_emb: np.ndarray,
        action_dist_b: np.ndarray,
        action_dist_e: np.ndarray,
        n_actions: int,
        position: Optional[np.ndarray] = None,
        min_emb_dim: int = 1,
        feature_pruning: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action_emb, name="action_emb", expected_dim=2)
        check_array(array=action_dist_b, name="action_dist_b", expected_dim=3)
        check_array(array=action_dist_e, name="action_dist_e", expected_dim=3)

        if feature_pruning == "exact":
            return self._estimate_with_exact_pruning(
                context=context,
                reward=reward,
                action=action,
                action_emb=action_emb,
                action_dist_b=action_dist_b,
                action_dist_e=action_dist_e,
                n_actions=n_actions,
                position=position,
                min_emb_dim=min_emb_dim,
            )

        else:
            return self._estimate_round_rewards(
                context=context,
                reward=reward,
                action=action,
                action_emb=action_emb,
                action_dist_b=action_dist_b,
                action_dist_e=action_dist_e,
                n_actions=n_actions,
                position=position,
            ).mean()

    def _estimate_with_exact_pruning(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_emb: np.ndarray,
        action_dist_b: np.ndarray,
        action_dist_e: np.ndarray,
        n_actions: int,
        position: Optional[np.ndarray] = None,
        min_emb_dim: int = 1,
    ) -> float:

        n_emb_dim = action_emb.shape[1]
        min_emb_dim = np.int32(np.minimum(n_emb_dim, min_emb_dim))
        theta_list, cnf_list = [], []
        feat_list, C = np.arange(n_emb_dim), np.sqrt(6) - 1
        for i in np.arange(n_emb_dim, min_emb_dim - 1, -1):
            comb_list = list(itertools.combinations(feat_list, i))
            theta_list_, cnf_list_ = [], []
            for comb in comb_list:
                theta, cnf = self._estimate_round_rewards(
                    context=context,
                    reward=reward,
                    action=action,
                    action_emb=action_emb[:, comb],
                    action_dist_b=action_dist_b,
                    action_dist_e=action_dist_e,
                    n_actions=n_actions,
                    with_cnf=True,
                )
                if len(theta_list) > 0:
                    theta_list_.append(theta), cnf_list_.append(cnf)
                else:
                    theta_list.append(theta), cnf_list.append(cnf)
                    continue

            idx_list = np.argsort(cnf_list_)[::-1]
            for idx in idx_list:
                theta_i, cnf_i = theta_list_[idx], cnf_list_[idx]
                theta_j, cnf_j = np.array(theta_list), np.array(cnf_list)
                if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
                    theta_list.append(theta_i), cnf_list.append(cnf_i)
                else:
                    return theta_j[-1]

        return theta_j[-1]

    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure."""
        return NotImplementedError


def run_ope(
    val_bandit_data: Dict,
    action_dist_val: np.ndarray,
    estimated_rewards: np.ndarray,
    estimated_rewards_mrdr: np.ndarray,
    policy_value: float,
) -> np.ndarray:

    lambdas = [10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]
    lambdas_sg = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
    ope = OffPolicyEvaluation(
        bandit_feedback=val_bandit_data,
        ope_estimators=[
            IPS(estimator_name="IPS"),
            DR(estimator_name="DR"),
            DM(estimator_name="DM"),
            SwitchDR(lambdas=lambdas, tuning_method="slope", estimator_name="SwitchDR"),
            DR(estimator_name="MRDR"),
            DRos(lambdas=lambdas, tuning_method="slope", estimator_name="DRos"),
            SGDR(
                lambdas=lambdas_sg,
                tuning_method="slope",
                estimator_name=r"DR-$\lambda$",
            ),
        ],
    )
    estimated_rewards_dict = {
        "DR": estimated_rewards,
        "DM": estimated_rewards,
        "SwitchDR": estimated_rewards,
        "MRDR": estimated_rewards_mrdr,
        "DRos": estimated_rewards,
        r"DR-$\lambda$": estimated_rewards,
    }
    squared_errors = ope.evaluate_performance_of_estimators(
        ground_truth_policy_value=policy_value,
        action_dist=action_dist_val,
        estimated_rewards_by_reg_model=estimated_rewards_dict,
        metric="se",
    )
    mips_estimate = MIPS().estimate_policy_value(
        context=val_bandit_data["context"],
        reward=val_bandit_data["reward"],
        action=val_bandit_data["action"],
        action_emb=val_bandit_data["action_context"],
        action_dist_b=val_bandit_data["pi_b"],
        action_dist_e=action_dist_val,
        n_actions=val_bandit_data["n_actions"],
        feature_pruning="no",
    )
    squared_errors["MIPS (w/o SLOPE)"] = (policy_value - mips_estimate) ** 2
    mips_estimate_slope = MIPS().estimate_policy_value(
        context=val_bandit_data["context"],
        reward=val_bandit_data["reward"],
        action=val_bandit_data["action"],
        action_emb=val_bandit_data["action_context"],
        action_dist_b=val_bandit_data["pi_b"],
        action_dist_e=action_dist_val,
        n_actions=val_bandit_data["n_actions"],
        feature_pruning="exact",
    )
    squared_errors["MIPS (w/ SLOPE)"] = (policy_value - mips_estimate_slope) ** 2

    relative_squared_errors = {}
    baseline = squared_errors["IPS"]
    for key, value in squared_errors.items():
        relative_squared_errors[key] = value / baseline

    return squared_errors, relative_squared_errors
