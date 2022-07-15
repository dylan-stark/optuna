from optuna._experimental import experimental_class
from optuna._imports import try_import


with try_import() as _imports:
    from comet_ml import Experiment


class CometCallback:
    """A callback to for logging study trials in Comet."""

    def __call__(self, study, trial) -> None:
        """Logs study and trial info to Comet if it is installed."""
        if _imports.is_successful():
            log_study_trial(study, trial)


def log_study_trial(study, trial) -> None:
    """Logs a single trial for the given study to Comet."""
    project_name = f"sklearn_simple_comet_sdk_{study.study_name}"
    experiment = Experiment(project_name=project_name)
    experiment.set_name(f"trail_{trial.number}")

    experiment.log_parameters(trial.params)
    experiment.log_metrics(
        dic={
            "value": trial.value,
        },
        step=0)
    experiment.log_others({
        "trial_number": trial.number,
        "trial_datetime_start": trial.datetime_start,
        "trial_system_attrs": trial.system_attrs,
        "trial_user_attrs": trial.user_attrs,
        "study_direction": study.direction,
    })
    experiment.end()
