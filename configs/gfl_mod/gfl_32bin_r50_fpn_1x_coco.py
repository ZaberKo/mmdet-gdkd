_base_ = ['../gfl/gfl_r50_fpn_1x_coco.py', '../_base_/wandb_log.py']

custom_imports = dict(imports=['distiller'], allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='GFLHeadMod',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLossMod', loss_weight=0.25),
        range_max=16,
        splits=32,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
)

_base_.wandb_backend.init_kwargs.update(
    dict(
        name='{{fileBasenameNoExtension}}',
        group='{{fileBasenameNoExtension}}_group',
        tags=['gfl', '32bin', 'r50', 'fpn', '1x', 'coco']
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    _base_.wandb_backend
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
