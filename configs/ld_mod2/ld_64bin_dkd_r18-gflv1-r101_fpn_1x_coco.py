_base_ = './ld_32bin_dkd_r18-gflv1-r101_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(
        splits=64
    )
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['ld', '64bin', 'dkd', 'r18-gflv1-r101', 'fpn', '1x', 'coco']
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
