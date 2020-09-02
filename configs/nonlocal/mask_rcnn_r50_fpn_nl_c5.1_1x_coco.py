_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='NonLocal2d'),
            stages=(False, False, False, True),
            position='after_conv3')
    ]))
