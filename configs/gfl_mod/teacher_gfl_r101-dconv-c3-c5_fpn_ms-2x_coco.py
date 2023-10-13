_base_ = '../gfl/gfl_r101-dconv-c3-c5_fpn_ms-2x_coco.py'

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth'

custom_imports = dict(imports=['distiller.debug_pkg'], allow_failed_imports=False)

model=dict(
    type = "GFLDebug",
    bbox_head=dict(type="GFLHeadDebug")
)