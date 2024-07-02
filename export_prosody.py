#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict

import math
import torch
import argparse

import utils
import commons
import attentions
from torch import nn
from models import DurationPredictor, ResidualCouplingBlock, Generator
from text.symbols import symbols
import os


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        self.emb_bert = nn.Linear(256, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths,bert):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        b,t,h = x.size()
        if bert is not None:
            b = self.emb_bert(bert[:,:t,:])
            x = x + b
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class VITS_Encoder(nn.Module):

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=False,
        **kwargs
    ):

        super().__init__()
        self.n_speakers = n_speakers
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def infer(self, x, x_lengths, bert,sid=None, noise_scale=1, length_scale=1):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths,bert)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        return z_p, y_mask


class VITS_Decoder(nn.Module):

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=False,
        **kwargs
    ):

        super().__init__()
        self.n_speakers = n_speakers
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def remove_weight_norm(self):
        self.flow.remove_weight_norm()

    def infer(self, z_p, y_mask, sid=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask), g=g)
        return o.squeeze(dim=1)


class OnnxModel_Encoder(torch.nn.Module):
    def __init__(self, model: VITS_Encoder):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, bert, noise_scale=1, length_scale=1):
        return self.model.infer(
            x=x,
            x_lengths=x_lengths,
            bert=bert,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )


class OnnxModel_Decoder(torch.nn.Module):
    def __init__(self, model: VITS_Decoder):
        super().__init__()
        self.model = model

    def forward(self, z_p, y_mask):
        return self.model.infer(
            z_p=z_p,
            y_mask=y_mask,
        )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Inference code for bert vits models')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    config_file = args.config
    checkpoint = args.model
    outdir = args.outdir

    hps = utils.get_hparams_from_file(config_file)
    print(hps)

    opset_version = 13

    # Encoder
    #########################################################################
    net_g = VITS_Encoder(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )

    _ = net_g.eval()
    _ = utils.load_model(checkpoint, net_g)

    x = torch.randint(low=0, high=100, size=(50,), dtype=torch.int64)
    x = x.unsqueeze(0)

    x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([1], dtype=torch.float32)
    length_scale = torch.tensor([1], dtype=torch.float32)

    bert_vec = torch.rand(size=(1,50,256),dtype=torch.float32)

    encoder = OnnxModel_Encoder(net_g)

    filename = os.path.join(outdir, "bert-vits-encoder.onnx")

    torch.onnx.export(
        encoder,
        (x, x_length, bert_vec,noise_scale, length_scale),
        filename,
        opset_version=opset_version,
        input_names=[
            "x",
            "x_length",
            "bert_vec",
            "noise_scale",
            "length_scale",
        ],
        output_names=["z_p", "y_mask"],
        dynamic_axes={
            "x": {0:"batch",1: "L"},
            "x_length":{0:"batch"},
            "bert_vec":{0:"batch",1:"L"},
            "z_p": {0:"batch",2: "L"},
            "y_mask": {0:"batch",2: "L"},
        },
    )

    # Decoder
    #########################################################################
    net_g = VITS_Decoder(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )

    _ = net_g.eval()
    _ = utils.load_model(checkpoint, net_g)
    net_g.remove_weight_norm()

    z_p = torch.rand(size=(1, hps.model.inter_channels, 200), dtype=torch.float32)
    y_mask = torch.randint(low=0, high=1, size=(1, 1, 200), dtype=torch.float32)

    decoder = OnnxModel_Decoder(net_g)

    filename = os.path.join(outdir,"vits-decoder-batch.onnx")

    torch.onnx.export(
        decoder,
        (z_p, y_mask),
        filename,
        opset_version=opset_version,
        input_names=[
            "z_p",
            "y_mask",
        ],
        output_names=["y"],
        dynamic_axes={
            "y": {0:"batch",1:"L"},
            "z_p": {0:"batch",2: "L"},
            "y_mask": {0:"batch",2: "L"},
        },
    )


if __name__ == "__main__":
    main()
