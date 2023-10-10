wandb_backend = dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         project='mmdet-distill',
                         name='{{fileBasenameNoExtension}}',
                         group='{{fileBasenameNoExtension}}_group',
                         tags=[]
                     ),
                     define_metric_cfg=[
                         dict(name="lr", step_metric='iter'),
                         dict(name="loss*", step_metric='iter'),
                         dict(name="coco/*", step_metric='epoch'),
                     ]
                     )
