_base_ = '../gfl/gfl_r50_fpn_1x_coco.py'
model = dict(
    bbox_head=dict(
        type='GFLHead',
        reg_max=32)
)
