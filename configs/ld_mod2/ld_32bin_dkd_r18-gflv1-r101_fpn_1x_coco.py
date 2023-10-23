_base_ = './ld_32bin_kd_r18-gflv1-r101_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(
        type='LDHeadMod',
        loss_ld=dict(
            type='KnowledgeDistillationDKDLoss',
            loss_weight=0.5,
            alpha=1.0,
            beta=2.0,
            T=1)
    )
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['ld', '32bin', 'dkd', 'r18-gflv1-r101', 'fpn', '1x', 'coco']
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
