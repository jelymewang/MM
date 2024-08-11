class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data_B/renjie/ViPT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data_B/renjie/ViPT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data_B/renjie/ViPT/pretrained_networks'
        self.got10k_val_dir = '/data_B/renjie/ViPT/data/got10k/val'
        self.lasot_lmdb_dir = '/data_B/renjie/ViPT/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data_B/renjie/ViPT/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data_B/renjie/ViPT/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data_B/renjie/ViPT/data/coco_lmdb'
        self.coco_dir = '/data_B/renjie/ViPT/data/coco'
        self.lasot_dir = '/data_B/renjie/ViPT/data/lasot'
        self.got10k_dir = '/data_B/renjie/ViPT/data/got10k/train'
        self.trackingnet_dir = '/data_B/renjie/ViPT/data/trackingnet'
        self.depthtrack_dir = '/data_G/yifan1/dataset/rgbd/depthtrack/train'
        self.lasher_dir = '/data_B/renjie/ViPT/data/lasher/trainingset'
        self.visevent_dir = '/data_B/renjie/ViPT/data/visevent/train'
