from backbone import resnet3D
from head import transformer

def generate_model(cfg):
    assert cfg.NAME in [
        'tempr4','resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if cfg.NAME == 'resnet':
        model = resnet3D.generate_model(model_depth=cfg.MODEL_DEPTH,
                                      n_classes=cfg.N_CLASSES,
                                      n_input_channels=cfg.N_INPUT_CHANNELS,
                                      shortcut_type=cfg.RESNET_SHORTCUT,
                                      conv1_t_size=cfg.CONV1_T_SIZE,
                                      conv1_t_stride=cfg.CONV1_T_STRIDE,
                                      no_max_pool=cfg.NO_MAX_POOL,
                                      widen_factor=cfg.RESNET_WIDEN_FACTOR,
                                      pretrained_path = cfg.PRETRAINED_MODEL)
    elif cfg.NAME == 'tempr4':
        model = transformer.Tempr4( num_freq_bands = cfg.NUM_FREQ_BANDS,
                                    depth = cfg.DEPTH,
                                    max_freq = cfg.MAX_FREQ,
                                    input_channels = cfg.INPUT_CHANNELS,
                                    input_axis = cfg.INPUT_AXIS,
                                    num_latents = cfg.NUM_LATENTS,
                                    latent_dim = cfg.LATENT_DIM,
                                    cross_heads = cfg.CROSS_HEADS,
                                    latent_heads = cfg.LATENT_HEADS,
                                    cross_dim_head = cfg.CROSS_DIM_HEAD,
                                    latent_dim_head = cfg.LATENT_DIM_HEAD,
                                    num_classes = cfg.NUM_CLASSES,
                                    attn_dropout = cfg.ATTN_DROPOUT,
                                    ff_dropout = cfg.FF_DROPOUT,
                                    weight_tie_layers = cfg.WEIGHT_TIE_LAYERS,
                                    fourier_encode_data = cfg.FOURIER_ENCODE_DATA,
                                    self_per_cross_attn = cfg.SELF_PER_CROSS_ATTN,
                                    final_classifier_head = cfg.FINAL_CLASSIFIER_HEAD)
        

    return model