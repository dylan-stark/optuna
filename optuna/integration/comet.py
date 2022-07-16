import os

from comet_ml import get_config

from optuna import logging
from optuna._imports import try_import


_logger = logging.get_logger(__name__)

with try_import() as _imports:
    from comet_ml import Experiment


class CometCallback:
    """A callback to for logging study trials in Comet."""

    def __init__(self) -> None:
        """Initializes callback."""
        self.is_enabled = self._check_if_should_run()

        if self.is_enabled:
            self.metric_name = os.environ.get("COMET_OPTUNA_METRIC_NAME", "value")

    def __call__(self, study, trial) -> None:
        """Logs study and trial info to Comet if it is installed."""
        if self.is_enabled:
            log_study_trial(study, trial, metric_name=self.metric_name)

    def _check_if_should_run(self) -> bool:
        """Checks if we have everything we need to automatically run."""
        have_comet = _imports.is_successful()
        have_comet_api_key = not get_config("comet.api_key") is None

        if have_comet and not have_comet_api_key:
            _logger.warning(
                "Found Comet but no an API key. "
                "Set `COMET_API_KEY` to enable automatic logging."
            )

        return have_comet and have_comet_api_key


def log_study_trial(study, trial, metric_name: str = "value") -> None:
    """Logs a single trial for the given study to Comet."""

    def add_flattened_with_index(source: dict, prefix: str, target: dict) -> dict:
        """Adds flattened dictionary with indexed set of (name, value) pairs."""
        for i, key in enumerate(source.keys()):
            target[f"{prefix}_{i}_name"] = key
            target[f"{prefix}_{i}_value"] = source[key]
        return target

    def add_flattened(source: dict, prefix: str, target: dict) -> dict:
        """Adds flattened dictionary with (name, value) pairs."""
        for key, value in source.items():
            target[f"{prefix}_{key}"] = value
        return target

    experiment = Experiment(project_name=study.study_name)
    experiment.set_name(f"trail_{trial.number}")

    study_info = {
        "study_best_trial_datetime_complete": study.best_trial.datetime_complete,
        "study_best_trial_datetime_start": study.best_trial.datetime_start,
        "study_best_trial_duration": study.best_trial.duration,
        "study_best_trial_number": study.best_trial.number,
        "study_best_trial_state": study.best_trial.state,
        f"study_best_trial_{metric_name}": study.best_trial.value,
        # TODO: Add support for values from best trial
        # TODO: Add support for best trials
        f"study_best_{metric_name}": study.best_value,
        "study_direction": study.direction,
        # TODO: Add support for directions
        "study_pruner": study.pruner,
        "study_sampler": study.sampler,
        "study_name": study.study_name,
    }
    study_info = add_flattened_with_index(study.best_params, "study_best_params", study_info)
    study_info = add_flattened_with_index(study.best_trial.distributions, "study_best_trial_distributions", study_info)
    study_info = add_flattened_with_index(study.best_trial.params, "study_best_trial_params", study_info)
    study_info = add_flattened(study.best_trial.system_attrs, "study_best_trial_system_attrs", study_info)
    study_info = add_flattened(study.best_trial.user_attrs, "study_best_trial_user_attrs", study_info)
    study_info = add_flattened(study.system_attrs, "study_system_attrs", study_info)
    study_info = add_flattened(study.user_attrs, "study_user_attrs", study_info)

    trial_info = {
        "trial_datetime_complete": trial.datetime_complete,
        "trial_datetime_start": trial.datetime_start,
        "trial_duration": trial.duration,
        "trial_number": trial.number,
        "trial_state": trial.state,
        f"trial_{metric_name}": trial.value,
    }
    trial_info = add_flattened_with_index(trial.distributions, "trial_distributions", trial_info)
    trial_info = add_flattened_with_index(trial.params, "trial_params", trial_info)
    trial_info = add_flattened(trial.system_attrs, "trial_system_attrs", trial_info)
    trial_info = add_flattened(trial.user_attrs, "trial_user_attrs", trial_info)

    experiment.log_parameters(trial.params)
    experiment.log_metrics(
        dic={
            metric_name: trial.value,
        },
        step=0)
    experiment.log_others(study_info)
    experiment.log_others(trial_info)
    experiment.end()
