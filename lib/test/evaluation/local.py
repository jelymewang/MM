from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data_B/renjie/ViPT/data/got10k_lmdb'
    settings.got10k_path = '/data_B/renjie/ViPT/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data_B/renjie/ViPT/data/itb'
    settings.lasot_extension_subset_path_path = '/data_B/renjie/ViPT/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data_B/renjie/ViPT/data/lasot_lmdb'
    settings.lasot_path = '/data_B/renjie/ViPT/data/lasot'
    settings.network_path = '/data_B/renjie/ViPT/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data_B/renjie/ViPT/data/nfs'
    settings.otb_path = '/data_B/renjie/ViPT/data/otb'
    settings.prj_dir = '/data_B/renjie/ViPT'
    settings.result_plot_path = '/data_B/renjie/ViPT/output/test/result_plots'
    settings.results_path = '/data_B/renjie/ViPT/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data_B/renjie/ViPT/output'
    settings.segmentation_path = '/data_B/renjie/ViPT/output/test/segmentation_results'
    settings.tc128_path = '/data_B/renjie/ViPT/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data_B/renjie/ViPT/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data_B/renjie/ViPT/data/trackingnet'
    settings.uav_path = '/data_B/renjie/ViPT/data/uav'
    settings.vot18_path = '/data_B/renjie/ViPT/data/vot2018'
    settings.vot22_path = '/data_B/renjie/ViPT/data/vot2022'
    settings.vot_path = '/data_B/renjie/ViPT/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

