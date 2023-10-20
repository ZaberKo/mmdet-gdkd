_base_ = './gfl_32bin_r50_fpn_ms-2x_coco.py'

model = dict(
    bbox_head=dict(
        splits=64)
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['gfl', '64bin', 'r50', 'fpn', '1x', 'coco']
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')