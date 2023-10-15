_base_ = ['./ld_gdkd_r18-gflv1-r101_fpn_1x_coco.py']

model = dict(
    bbox_head=dict(
        loss_ld=dict(
            type='KnowledgeDistillationGDKDLoss', 
            loss_weight=0.5,
            w0=1.0,
            w1=1.0,
            w2=2.0,
            T=4),
        loss_cls_kd=dict(
            type='KnowledgeDistillationDISTModLoss',
            loss_weight=1.0,
            beta=1.0,
            T=1)
    )
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['ld', 'gdkd', 'r18-gflv1-r101', 'fpn', '1x', 'coco', 'cls-dist']
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')