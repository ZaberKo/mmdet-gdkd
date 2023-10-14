_base_ = ['./ld_dkd_r18-gflv1-r101_fpn_1x_coco.py']

model = dict(
    bbox_head=dict(
        loss_ld_avg_mode='mean',
    )
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['ld', 'dkd', 'r18-gflv1-r101', 'fpn', '1x', 'coco','ld-mean']
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')