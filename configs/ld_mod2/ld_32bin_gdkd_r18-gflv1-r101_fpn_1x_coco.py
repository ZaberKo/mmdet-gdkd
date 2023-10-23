_base_ = './ld_32bin_kd_r18-gflv1-r101_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(
        type='LDHeadMod2',
        loss_ld=dict(
            type='KnowledgeDistillationGDKDLoss',
            loss_weight=0.5,
            w0=1.0,
            w1=1.0,
            w2=2.0,
            k=5,
            T=1)
    )
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['ld', '32bin', 'gdkd', 'r18-gflv1-r101', 'fpn', '1x', 'coco']
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
