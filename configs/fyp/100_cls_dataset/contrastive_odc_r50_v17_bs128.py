_base_ = '../../base.py'
# model settings
# num_classes = 1000

# NOTE
# set to 200? ODC has classes = 10000 while imagenet only has 1000 classes
# num_classes = 20
num_classes = 100
train_bs = 64

model = dict(
    type='ContrastiveODC_V17',
    pretrained=None,
    with_sobel=False,
    num_neg_centroids=8,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        with_cp=True),
    neck=dict(
        type='NonLinearNeckSimCLR',
        in_channels=2048,
        hid_channels=2048,
        out_channels=256,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveODCHead_V17',
        alpha=0.2,
        beta=1,
        with_avg_pool=False,
        in_channels=256,
        num_classes=num_classes),
    memory_bank=dict(
        type='ODCMemory',
        # length=5052,
        length=63916,
        feat_dim=256,
        momentum=0.5,
        num_classes=num_classes,
        min_cluster=32,
        debug=False))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/imagenet/meta/subdataset/train_labeled_50percent_10interval_no_label.txt'
data_train_root = 'data/imagenet/train'
dataset_type = 'ContrastiveODCDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0,
                kernel_size=23)
        ],
        p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

extract_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=train_bs,  # 64*8
    sampling_replace=True,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline))

# additional hooks
custom_hooks = [
    dict(
        type='DeepClusterHook',
        extractor=dict(
            imgs_per_gpu=128,
            workers_per_gpu=4,
            dataset=dict(
                type=dataset_type,
                for_extractor=True,
                data_source=dict(
                    list_file=data_train_list,
                    root=data_train_root,
                    **data_source_cfg),
                pipeline=extract_pipeline)),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=-1),  # no pca
        unif_sampling=False,
        reweight=True,
        reweight_pow=0.5,
        init_memory=True,
        initial=True,  # call initially
        interval=9999999999),  # initial only
    dict(
        type='ODCHook',
        centroids_update_interval=10,  # iter
        deal_with_small_clusters_interval=1,
        evaluate_interval=50,
        reweight=True,
        reweight_pow=0.5)
]

# optimizer
optimizer = dict(type='LARS', lr=0.2, weight_decay=0.000001, momentum=0.9,
                 paramwise_options={
                     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                     'bias': dict(weight_decay=0., lars_exclude=True)})

# optimizer = dict(
#     type='SGD', lr=0.005, momentum=0.9, weight_decay=0.00001,
#     nesterov=False,
#     paramwise_options={'\Ahead.': dict(momentum=0.)})

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)

checkpoint_config = dict(interval=20)

# runtime settings
total_epochs = 200

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
