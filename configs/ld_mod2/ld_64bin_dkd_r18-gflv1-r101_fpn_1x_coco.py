_base_ = './ld_32bin_dkd_r18-gflv1-r101_fpn_1x_coco.py'

teacher_ckpt = './models_home/checkpoints/gfl_64bin_r101_fpn_ms-2x_coco.pth'  # noqa

model = dict(
    teacher_config='configs/gfl_mod/gfl_64bin_r101_fpn_ms-2x_coco.py',
    teacher_ckpt=teacher_ckpt,
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
