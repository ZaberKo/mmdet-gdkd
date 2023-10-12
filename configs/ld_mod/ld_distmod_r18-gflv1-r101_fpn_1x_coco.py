_base_ = ['../ld/ld_r18-gflv1-r101_fpn_1x_coco.py', './wandb_log.py']

custom_imports = dict(imports=['distiller'], allow_failed_imports=False)

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'  # noqa
model = dict(
    type='KnowledgeDistillationSingleStageDetector',
    teacher_config='configs/gfl/gfl_r101_fpn_ms-2x_coco.py',
    teacher_ckpt=teacher_ckpt,
    bbox_head=dict(
        type='LDHeadMod',
        loss_ld=dict(
            type='KnowledgeDistillationDISTModLoss', 
            loss_weight=0.25,
            beta=2.0,
            T=4),
        reg_max=16)
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['ld', 'dist', 'r18-gflv1-r101', 'fpn', '1x', 'coco']
    )
)


vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')