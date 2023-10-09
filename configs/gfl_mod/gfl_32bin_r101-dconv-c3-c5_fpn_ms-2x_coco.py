_base_ = '../gfl/gfl_r101-dconv-c3-c5_fpn_ms-2x_coco.py'
model = dict(
    bbox_head=dict(
        type='GFLHead',
        reg_max=32)
)
