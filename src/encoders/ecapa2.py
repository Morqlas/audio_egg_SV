# This is a helper function file to define the building blocks of the ECAPA2 model. It is not meant to be run on its own, but rather imported into the ECAPA2.py file.

import torch
import torchaudio
import torch.nn as nn
from src.encoders.blocks import Log, Normalize, SEBlock2d, SEBlock, Res2NetConv1d, AttentiveStatPoolingContextChannelEfficientVAD, Net

from collections import OrderedDict
from typing import Dict, Optional

## Builders from state_dict (file) shapes ##

def get_pre_processing(hop_length=160, window_size=400, n_fft=511):
    return nn.Sequential(
        torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=window_size,
            hop_length=hop_length,
            power=2.0,
            center=True,
            window_fn=torch.hamming_window
        ),
        Log(),
        Normalize()
    )

def infer_same_padding_2d(kernel_size):
    kh, kw = kernel_size
    return (kh // 2, kw // 2)

def infer_same_padding_1d(kernel_size):
    return kernel_size // 2

def build_conv2d_exact(sd, key, stride=None, padding=None, dilation=None):
    w = sd[f"{key}.weight"]
    out_ch, in_ch, kh, kw = w.shape

    if stride is None:
        stride = (1, 1)
    if padding is None:
        padding = (kh // 2, kw // 2)
    if dilation is None:
        dilation = (1, 1)

    return nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=(kh, kw),
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=f"{key}.bias" in sd,
    )

def build_conv1d_from_weight(weight: torch.Tensor, stride=1, bias=True) -> nn.Conv1d:
    out_ch, in_ch, k = weight.shape
    return nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=stride, padding=infer_same_padding_1d(k), bias=bias)

def build_linear_from_weight(weight: torch.Tensor, bias=True) -> nn.Linear:
    out_f, in_f = weight.shape
    return nn.Linear(in_features=in_f, out_features=out_f, bias=bias)

def build_bn1d_from_weight(weight: torch.Tensor) -> nn.BatchNorm1d:
    num_features = weight.shape[0]
    return nn.BatchNorm1d(num_features=num_features)

def build_bn2d_from_weight(weight: torch.Tensor) -> nn.BatchNorm2d:
    num_features = weight.shape[0]
    return nn.BatchNorm2d(num_features=num_features)

def build_lfe_block(sd, prefix, first_stride=(1, 1)):
    conv0 = build_conv2d_exact(sd, f"{prefix}.0", stride=first_stride, padding=(1, 1), dilation=(1, 1))
    bn2 = nn.BatchNorm2d(sd[f"{prefix}.2.weight"].shape[0])

    conv3 = build_conv2d_exact(sd, f"{prefix}.3", stride=(1, 1), padding=(1, 1), dilation=(1, 1))
    bn5 = nn.BatchNorm2d(sd[f"{prefix}.5.weight"].shape[0])

    conv6 = build_conv2d_exact(sd, f"{prefix}.6", stride=(1, 1), padding=(1, 1), dilation=(1, 1))
    bn8 = nn.BatchNorm2d(sd[f"{prefix}.8.weight"].shape[0])

    se_in = sd[f"{prefix}.9.fc.3.weight"].shape[0]
    se_red = sd[f"{prefix}.9.fc.0.weight"].shape[0]
    se = SEBlock2d(se_in, se_red)

    return nn.Sequential(
        conv0,
        nn.ReLU(),
        bn2,
        conv3,
        nn.ReLU(),
        bn5,
        conv6,
        nn.ReLU(),
        bn8,
        se,
    )

def build_downsample_block(sd: Dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    # In ECAPA2 this is typically stride (2,1)

    conv_w = sd[f"{prefix}.0.weight"]
    out_ch, in_ch, kh, kw = conv_w.shape
    conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(kh, kw), stride=(2,1), padding=infer_same_padding_2d((kh, kw)), bias=f"{prefix}.0.bias" in sd)

    bn = build_bn2d_from_weight(sd[f"{prefix}.1.weight"])
    return nn.Sequential(conv, bn)

def build_tdnn_1(sd: Dict[str, torch.Tensor]) -> nn.Sequential:
    return nn.Sequential(
        build_conv1d_from_weight(sd["tdnn_1.0.weight"], bias="tdnn_1.0.bias" in sd),
        nn.ReLU(),
        build_bn1d_from_weight(sd["tdnn_1.2.weight"])
    )   

def build_res2net_conv1d(sd: Dict[str, torch.Tensor], prefix: str) -> Res2NetConv1d:
    conv_filters = []
    batch_norms = []

    i = 0
    while f"{prefix}.conv_filters.{i}.weight" in sd:
        w = sd[f"{prefix}.conv_filters.{i}.weight"]
        conv_filters.append(
            nn.Conv1d(
                in_channels=w.shape[1],
                out_channels=w.shape[0],
                kernel_size=w.shape[2],
                padding=2,
                dilation=2,
                padding_mode='reflect',  # THIS was the bug
                bias=f"{prefix}.conv_filters.{i}.bias" in sd,
            )
        )
        batch_norms.append(
            build_bn1d_from_weight(sd[f"{prefix}.batch_norms.{i}.weight"])
        )
        i += 1

    return Res2NetConv1d(nn.ModuleList(conv_filters), nn.ModuleList(batch_norms))

def build_tdnn_2(sd):
    se_in = sd["tdnn_2.7.fc.3.weight"].shape[0]
    se_red = sd["tdnn_2.7.fc.0.weight"].shape[0]

    return nn.Sequential(
        nn.Conv1d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=True),  # tdnn_2.0
        nn.ReLU(),
        nn.BatchNorm1d(1024),                                                    # tdnn_2.2
        build_res2net_conv1d(sd, "tdnn_2.3"),                                    # tdnn_2.3
        nn.Conv1d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=True),    # tdnn_2.4
        nn.ReLU(),
        nn.BatchNorm1d(1024),                                                    # tdnn_2.6
        SEBlock(1024, se_red),                                                   # tdnn_2.7
    )

def build_dense_1(sd: Dict[str, torch.Tensor]) -> nn.Sequential:
    return nn.Sequential(
        build_conv1d_from_weight(sd["dense_1.0.weight"], bias="dense_1.0.bias" in sd),
        nn.ReLU()
    )

def build_pooling_1(sd: Dict[str, torch.Tensor]) -> nn.Sequential:
    # attention.0 input is in_channels*3, output is hidden_ch
    hidden_ch = sd["pooling_1.0.attention.0.weight"].shape[0]
    in_ch_times_3 = sd["pooling_1.0.attention.0.weight"].shape[1]
    in_ch = in_ch_times_3 // 3
    pooling = AttentiveStatPoolingContextChannelEfficientVAD(in_ch, hidden_ch)
    bn = build_bn1d_from_weight(sd["pooling_1.1.weight"])
    
    return nn.Sequential(pooling, bn)

def build_dense_2(sd: Dict[str, torch.Tensor]) -> nn.Sequential:
    return nn.Sequential(
        build_linear_from_weight(sd["dense_2.0.weight"], bias="dense_2.0.bias" in sd)
    )

###########################################################################################


### Building the model from the state dictionary ###

def build_ecapa2_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    hop_length: Optional[int] = 10,
    scripted_model: torch.jit.ScriptModule = None
) -> Net:
    
    # ---- pre-processing

    pre_processing = get_pre_processing()

    # ---- LFE blocks
    conv_names = [
        "conv_1", "conv_2", "conv_17", "conv_3",
        "conv_4", "conv_5", "conv_18", "conv_6",
        "conv_7", "conv_8", "conv_19", "conv_9",
        "conv_10", "conv_11", "conv_20", "conv_12",
        "conv_13", "conv_14", "conv_15", "conv_16",
    ]

    strided_blocks = {"conv_3", "conv_6", "conv_9", "conv_12"}

    conv_blocks = OrderedDict()
    for name in conv_names:
        first_stride = (2, 1) if name in strided_blocks else (1, 1)
        conv_blocks[name] = build_lfe_block(state_dict, name, first_stride=first_stride)

        downsample_names = ["downsample_1", "downsample_2", "downsample_3", "downsample_4"]
        downsample_blocks = OrderedDict((name, build_downsample_block(state_dict, name)) for name in downsample_names)

    # ---- GFE / pooling / embedding
    tdnn_1 = build_tdnn_1(state_dict)
    tdnn_2 = build_tdnn_2(state_dict)
    dense_1 = build_dense_1(state_dict)
    pooling_1 = build_pooling_1(state_dict)
    dense_2 = build_dense_2(state_dict)

    model = Net(
        pre_processing=pre_processing,
        conv_blocks=conv_blocks,
        downsample_blocks=downsample_blocks,
        tdnn_1=tdnn_1,
        tdnn_2=tdnn_2,
        dense_1=dense_1,
        pooling_1=pooling_1,
        dense_2=dense_2,
    )

# Loading the state dict into the model (with non-strict loading to allow for any missing keys that are not being used)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("WARNING - missing keys:", missing)
    if unexpected:
        print("WARNING - unexpected keys:", unexpected)

    return model
    