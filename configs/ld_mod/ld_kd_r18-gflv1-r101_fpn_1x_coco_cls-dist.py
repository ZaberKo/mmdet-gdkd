_base_ = ['./ld_kd_r18-gflv1-r101_fpn_1x_coco.py']

model = dict(
    bbox_head=dict(
        loss_ld=dict(
            type='KnowledgeDistillationKDLoss',
            loss_weight=0.5,
            T=4),
        loss_cls_kd=dict(
            type='KnowledgeDistillationDISTLoss',
            loss_weight=0.5,
            beta=1.0,
            gamma=1.0,
            T=2)
    )
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['ld', 'kd', 'r18-gflv1-r101', 'fpn', '1x', 'coco', 'cls-dist']
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
