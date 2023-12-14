#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.operations.operator import Operator
from mephisto.tools.scripts import task_script, build_custom_bundle
from mephisto.data_model.qualification import QUAL_NOT_EXIST, QUAL_EXISTS
from mephisto.utils.qualifications import make_qualification_dict
from mephisto.abstractions.blueprints.abstract.static_task.static_blueprint import (
    SharedStaticTaskState,
)
from omegaconf import DictConfig

NUM_TASKS = 350
PILOT_QUALIFICATION = 'long-caps-ready'
ALLOWLIST_QUALIFICATION = 'long-caps-approved'

@task_script(default_config_file="qualify.yaml")
def main(operator: Operator, cfg: DictConfig) -> None:
    shared_state = SharedStaticTaskState(
        static_task_data=[{}] * NUM_TASKS,
    )
    shared_state.qualifications = [
        make_qualification_dict(
            PILOT_QUALIFICATION,
            QUAL_NOT_EXIST,
            None,
        ),
        make_qualification_dict(
            ALLOWLIST_QUALIFICATION,
            QUAL_NOT_EXIST,
            None,
        ),
    ]
    shared_state.mturk_specific_qualifications = [
        {
            "QualificationTypeId": "00000000000000000040",
            "Comparator": "GreaterThanOrEqualTo",
            "IntegerValues": [10000],
            "ActionsGuarded": "DiscoverPreviewAndAccept",
        },
        {
            "QualificationTypeId": "000000000000000000L0",
            "Comparator": "GreaterThanOrEqualTo",
            "IntegerValues": [99],
            "ActionsGuarded": "DiscoverPreviewAndAccept",
        },
    ]
    task_dir = cfg.task_dir

    build_custom_bundle(
        task_dir,
        force_rebuild=cfg.mephisto.task.force_rebuild,
        post_install_script=cfg.mephisto.task.post_install_script,
    )

    operator.launch_task_run(cfg.mephisto, shared_state)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()
