_base_ = './ld_32bin_kd_r18-gflv1-r101_fpn_1x_coco.py'

custom_imports = dict(imports=['distiller'], allow_failed_imports=False)

teacher_ckpt = './models_home/checkpoints/gfl_32bin_r101_fpn_ms-2x_coco.pth'  # noqa

model = dict(
    bbox_head=dict(
        splits=64
    )
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['ld', '64bin', 'kd', 'r18-gflv1-r101', 'fpn', '1x', 'coco']
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
