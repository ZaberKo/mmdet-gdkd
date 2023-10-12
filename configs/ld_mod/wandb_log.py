_base_ = '../_base_/wandb_log.py'

_base_.wandb_backend.define_metric_cfg.append(dict(name="ld/loss*", step_metric='iter'))