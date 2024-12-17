auto_scale_lr = dict(base_batch_size=16)
backend_args = None
data_root = '/data/disk1_ssd/ftl/ExDark/ExDa4k'
dataset_type = 'VOCDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = 'work_dirs/detr_r18-exdark/epoch_315.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
model = dict(
    backbone=dict(
        depth=18,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        embed_dims=256,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            bg_cls_weight=0.1,
            class_weight=1.0,
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=80,
        type='DETRHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1,
                num_fcs=2),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8)),
        num_layers=6,
        return_intermediate=True),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1,
                num_fcs=2),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            512,
        ],
        kernel_size=1,
        norm_cfg=None,
        num_outs=1,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=100,
    positional_encoding=dict(normalize=True, num_feats=128),
    test_cfg=dict(max_per_img=100),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=1.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DETR')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(decay_mult=1.0, lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=500,
        gamma=0.1,
        milestones=[
            334,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='VOC2007/ImageSets/Main/test.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2007'),
        data_root='/data/disk1_ssd/ftl/ExDark/ExDa4k',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                416,
                416,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='VOCDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(eval_mode='11points', metric='mAP', type='VOCMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        416,
        416,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=500, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=64,
    dataset=dict(
        dataset=dict(
            datasets=[
                dict(
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    backend_args=None,
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    data_root='/data/disk1_ssd/ftl/ExDark/ExDa4k',
                    filter_cfg=dict(
                        bbox_min_size=32, filter_empty_gt=True, min_size=32),
                    pipeline=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            keep_ratio=True, scale=(
                                416,
                                416,
                            ), type='Resize'),
                        dict(prob=0.5, type='RandomFlip'),
                        dict(type='PackDetInputs'),
                    ],
                    type='VOCDataset'),
                dict(
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    backend_args=None,
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    data_root='/data/disk1_ssd/ftl/ExDark/ExDa4k',
                    filter_cfg=dict(
                        bbox_min_size=32, filter_empty_gt=True, min_size=32),
                    pipeline=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            keep_ratio=True, scale=(
                                416,
                                416,
                            ), type='Resize'),
                        dict(prob=0.5, type='RandomFlip'),
                        dict(type='PackDetInputs'),
                    ],
                    type='VOCDataset'),
            ],
            ignore_keys=[
                'dataset_type',
            ],
            type='ConcatDataset'),
        times=1,
        type='RepeatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        416,
        416,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='VOC2007/ImageSets/Main/test.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2007/'),
        data_root='/data/disk1_ssd/ftl/ExDark/ExDa4k',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                416,
                416,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='VOCDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(eval_mode='11points', metric='mAP', type='VOCMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/detr_r18-exdark'
