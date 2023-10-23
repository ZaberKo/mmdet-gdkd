_base_ = './gfl_32bin_r101_fpn_ms-2x_coco.py'

load_from = 'models_home/checkpoints/gfl_32bin_r101_fpn_ms-2x_coco.pth'

custom_imports = dict(
    imports=['distiller.debug_pkg'], allow_failed_imports=False)

model=dict(
    bbox_head=dict(type="GFLHeadModDebug")
)

vis_backends = [
    dict(type='LocalVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')