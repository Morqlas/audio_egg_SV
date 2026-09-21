# This is a file to store the helper classes for the ECAPA2 model. It is not meant to be run on its own, but rather imported into the ECAPA2.py file.
import torch
import torch.nn as nn
import torch.nn.functional as F

#from collections import OrderedDict
from typing import Dict, List, Optional

## PRE-PROCESSING LOG AND NORMALIZATION BLOCKS ##

class Log(nn.Module):
    def forward(self, x):
        return torch.log(torch.add(x, 1e-6))

class Normalize(nn.Module):
    def forward(self, x):
        return torch.sub(x, torch.mean(x, dim=-1, keepdim=True))

## SQUEEZE-EXCITATION BLOCKS ##

class SEBlock2d(nn.Module):

    def __init__(self, channels: int, reduction_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction_channels),    #fc.0
            nn.ReLU(),                                  #fc.1
            nn.BatchNorm1d(reduction_channels),         #fc.2
            nn.Linear(reduction_channels, channels),    #fc.3
            nn.Sigmoid()                                #fc.4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.avg_pool(x).view((b,c)) # [Batch, Channels]    
        s = self.fc(s) # [Batch, Channels, 1, 1]
        return x * s.view((b,c,1,1))  # Scale the input by the SE weights
        

class SEBlock(nn.Module):
    # 1D version of SEBlock, used in the Res2Net blocks
    def __init__(self, channels: int, reduction_channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction_channels),    #fc.0
            nn.ReLU(),                                  #fc.1
            nn.BatchNorm1d(reduction_channels),         #fc.2
            nn.Linear(reduction_channels, channels),    #fc.3
            nn.Sigmoid()                                #fc.4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Channels, Time]
        s = x.mean(dim=2) # [Batch, Channels] 
        s = self.fc(s).unsqueeze(2) # [Batch, Channels, 1] 
        return x * s
    
## RES2NET CONV1D BLOCK ##

class Res2NetConv1d(nn.Module):
    def __init__(self, conv_filters: nn.ModuleList, batch_norms: nn.ModuleList):
        super().__init__()
        self.conv_filters = conv_filters
        self.batch_norms = batch_norms
        self.n_groups = len(conv_filters) + 1  # Number of groups is one more than the number of conv layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.n_groups, dim=1)
        
        outputs = [chunks[0]]

        # First branch: no addition, relu THEN bn
        out = F.relu(self.conv_filters[0](chunks[1]), inplace=False)
        out = self.batch_norms[0](out)
        outputs.append(out)

        # Remaining branches
        for i in range(1, len(self.conv_filters)):
            input_chunk = chunks[i + 1] + outputs[i]
            out = F.relu(self.conv_filters[i](input_chunk), inplace=False)
            out = self.batch_norms[i](out)
            outputs.append(out)

        return torch.cat(outputs, dim=1)
    
## ATTENTIVE STATISTICAL POOLING CONTEXT CHANNEL EFFICIENT VAD ##

class AttentiveStatPoolingContextChannelEfficientVAD(nn.Module):
    def __init__(self, in_channels: int, attention_hidden_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels * 3, attention_hidden_channels, kernel_size=1),  # x3 for context
            nn.ReLU(),
            nn.BatchNorm1d(attention_hidden_channels),
            nn.Conv1d(attention_hidden_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        mean = torch.mean(x, dim=2, keepdim=True).expand_as(x)
        std = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=True) + 1e-5).expand_as(x)
        
        # context-aware input: concatenate x, mean, std
        context = torch.cat([x, mean, std], dim=1)  # [B, 3C, T]
        
        attention_weights_logits = self.attention(context)
        attention_weights = torch.softmax(attention_weights_logits, dim=2)
        
        x_scaled = attention_weights * x
        mu = torch.sum(x_scaled, dim=2)
        
        sigma = torch.sqrt(
            torch.abs(torch.sum(x_scaled * x, dim=2) - mu ** 2) + 1e-5
        )
        
        return torch.cat([mu, sigma], dim=1)
    
## ECAPA2 MODEL CLASS ##

class Net(nn.Module):
    """
    Plain reconstruction of the TorchScript ECAPA2 model.
    """
    def __init__(self,
                 pre_processing: nn.Sequential,
                 conv_blocks: Dict[str, nn.Module],
                 downsample_blocks: Dict[str, nn.Module],
                 tdnn_1: nn.Sequential,
                 tdnn_2: nn.Sequential,
                 dense_1: nn.Sequential,
                 pooling_1: nn.Sequential,
                 dense_2: nn.Sequential,
                 valid_labels: Optional[List[str]] = None):

        super().__init__()
        
        self.pre_processing = pre_processing

        # Register conv blocks with exact names
        for name, module in conv_blocks.items():
            setattr(self, name, module)

        # Register downsample blocks with exact names
        for name, module in downsample_blocks.items():
            setattr(self, name, module)

        self.tdnn_1 = tdnn_1
        self.tdnn_2 = tdnn_2
        self.dense_1 = dense_1
        self.pooling_1 = pooling_1
        self.dense_2 = dense_2

        if valid_labels is None:
            valid_labels = ["embedding", "gfe_1", "gfe_2", "pool", "attention"]
        self.valid_labels = valid_labels

    def utterance_level_feature_extraction(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_2(x)

    def forward(self, x: torch.Tensor, labels: str = "embedding") -> torch.Tensor:
        label_list = labels.split("|")
        if not all(item in self.valid_labels for item in label_list):
            raise AssertionError("Invalid label given")

        outputs = []

        # Preprocessing

        x = torch.squeeze(self.pre_processing(x), 1) # [Batch, Frequency, Time] (just 1 channel -> squeezed)
        x = x.to(self.conv_1[0].weight.dtype)
        x = self.conv_1(torch.unsqueeze(x, 1)) # Channel unsqueezed

        # Local Feature Encoder stages

        x = self.conv_2(x) + x
        x = self.conv_17(x) + x
        y = self.conv_3(x)
        x = y + self.downsample_1(x)

        x = self.conv_4(x) + x
        x = self.conv_5(x) + x
        x = self.conv_18(x) + x
        y = self.conv_6(x)
        x = y + self.downsample_2(x)

        x = self.conv_7(x) + x
        x = self.conv_8(x)
        x = self.conv_19(x) + x
        y = self.conv_9(x)
        x = y + self.downsample_3(x)

        x = self.conv_10(x) + x
        x = self.conv_11(x) + x
        x = self.conv_20(x) + x
        y = self.conv_12(x)
        x = y + self.downsample_4(x)

        x = self.conv_13(x) + x
        x = self.conv_14(x) + x
        x = self.conv_15(x) + x
        x = self.conv_16(x) + x

        # Flatten 2D -> 1D 
        # [Batches, Channels, Freq, Time] -> [Batches, Channels * Freq, Time]

        b, c, f, t = x.shape

        x = x.reshape(b, c*f, t)

        # Global Feature Extractor part

        x = self.tdnn_1[0](x)

        if "gfe_1" in label_list:
            outputs.append(torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1))
            label_list.remove("gfe_1")

        if len(label_list) == 0:
            return torch.cat(outputs, dim=1)

        x = self.tdnn_1[1](x)
        x = self.tdnn_1[2](x)

        x = self.tdnn_2(x) + x

        if "gfe_2" in label_list:
            outputs.append(torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1))
            label_list.remove("gfe_2")

        if len(label_list) == 0:
            return torch.cat(outputs, dim=1)

        x = self.dense_1[0](x)

        if "pool" in label_list:
            outputs.append(torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1))
            label_list.remove("pool")

        if len(label_list) == 0:
            return torch.cat(outputs, dim=1)

        x = self.dense_1[1](x)
        pool = self.pooling_1(x)

        if "attention" in label_list:
            outputs.append(pool)
            label_list.remove("attention")

        if len(label_list) == 0:
            return torch.cat(outputs, dim=1)

        embedding = self.utterance_level_feature_extraction(pool)
        outputs.append(embedding)
        return torch.cat(outputs, dim=1)
    
import math

import torch
from torch import nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    def __init__(self, embed_size, num_classes, scale=30, margin=0.2, easy_margin=False, **kwargs):
        r"""
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).
        
        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is from https://github.com/ronghuaiyang/arcface-pytorch, which takes
        54.804054962005466 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080Ti.
        """
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
        one_hot.scatter_(1, ground_truth.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cos_theta)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.scale

        loss = self.ce(output, ground_truth)
        return loss


            





            


            

