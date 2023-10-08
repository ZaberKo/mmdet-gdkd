_base_ = '../ld/ld_r18-gflv1-r101_fpn_1x_coco.py'

custom_imports = dict(imports=['distiller.loss'], allow_failed_imports=False)

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'  # noqa
model = dict(
    type='KnowledgeDistillationSingleStageDetector',
    teacher_config='configs/gfl/gfl_r101_fpn_ms-2x_coco.py',
    teacher_ckpt=teacher_ckpt,
    bbox_head=dict(
        type='LDHead',
        loss_ld=dict(
            type='KnowledgeDistillationDISTLoss', 
            loss_weight=0.25, 
            T=10),
        reg_max=16)
)
