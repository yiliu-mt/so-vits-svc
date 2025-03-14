import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import modules.attentions as attentions
import modules.commons as commons
import modules.modules as modules

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import utils
from modules.commons import init_weights, get_padding
from modules.speaker_classifier import SpeakerClassifier
from utils import f0_to_coarse

class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        total_logdet = 0
        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += logdet
        else:
            for flow in reversed(self.flows):
                x, logdet = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += logdet
        return x, total_logdet


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        # print(x.shape,x_lengths.shape)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class TextEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_layers,
                 gin_channels=0,
                 filter_channels=None,
                 n_heads=None,
                 p_dropout=None,
                 use_f0=True):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        if use_f0:
            self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, x, x_mask, f0=None, noice_scale=1):
        x = x + (self.f0_emb(f0).transpose(1, 2) if f0 is not None else 0)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask
        return z, m, logs, x_mask, x


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            # embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed

class F0Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
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
                 gin_channels,
                 ssl_dim,
                 n_speakers,
                 sampling_rate=44100,
                 vol_embedding=False,
                 vocoder_name="nsf-hifigan",
                 bidirectional_flow_weight=0,
                 speaker_grl_weight=0,
                 use_f0=True,
                 ppg_std=0,
                 vae_std=0,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.bidirectional_flow_weight = bidirectional_flow_weight
        self.use_f0 = use_f0
        self.ppg_std = ppg_std
        self.vae_std = vae_std
        self.speaker_grl_weight = speaker_grl_weight

        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        if self.ssl_dim == 1:
            # The input content feature is index
            # we assume that the content index is less than 1000
            self.emb_c = nn.Embedding(1000, filter_channels)
            self.pre = nn.Conv1d(filter_channels, hidden_channels, kernel_size=5, padding=2)
        else:
            self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            use_f0=use_f0,
        )
        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "use_f0": use_f0,
        }

        if "decoder" in kwargs:
            vocoder_name = kwargs['decoder']

        if vocoder_name == "nsf-hifigan" or vocoder_name == "nsf_decoder":
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)
        elif vocoder_name == "vits-hifigan" or vocoder_name == "vits_decoder":
            from vdecoder.hifigan.vits_models import Generator
            self.dec = Generator(h=hps)
        elif vocoder_name == "nsf-snake-hifigan":
            from vdecoder.hifiganwithsnake.models import Generator
            self.dec = Generator(h=hps)
        else:
            print("[?] Unkown vocoder: use default(nsf-hifigan)")
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)

        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
        if self.use_f0:
            self.f0_decoder = F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels
            )
            # TODO (yi.liu): Do we need it in speech conversion
            self.emb_uv = nn.Embedding(2, hidden_channels)

        if vol_embedding:
           self.emb_vol = nn.Linear(1, hidden_channels)

        self.character_mix = False

        if self.speaker_grl_weight > 0:
            self.speaker_classifier = SpeakerClassifier(
                hidden_channels,
                n_speakers
            )

    def EnableCharacterMix(self, n_speakers_map, device):
        self.speaker_map = torch.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
        for i in range(n_speakers_map):
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[i]]).to(device))
        self.speaker_map = self.speaker_map.unsqueeze(0).to(device)
        self.character_mix = True

    def forward(self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None, vol = None):
        if self.ssl_dim == 1:
            # embedding if the content feature is index
            c = self.emb_c(c.long()).squeeze(1).transpose(1,2)

        c = c + (torch.randn_like(c) * self.ppg_std if self.ppg_std > 0 else 0)
        g = self.emb_g(g).transpose(1,2)


        # vol proj
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol!=None and self.vol_embedding else 0

        # ssl prenet
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask + (self.emb_uv(uv.long()).transpose(1,2) if self.use_f0 else 0) + vol

        # f0 predict
        if self.use_f0:
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        else:
            lf0, norm_lf0, pred_lf0 = None, None, None

        # encoder
        z_p, m_p, logs_p, _, text_info = self.enc_p(
            x, x_mask, f0=(f0_to_coarse(f0) if self.use_f0 else None))

        # speaker discriminator
        if self.speaker_grl_weight > 0:
            text_emb = torch.sum(text_info, dim=-1) / c_lengths.unsqueeze(1)
            spk_preds = self.speaker_classifier(text_emb)
        else:
            spk_preds = None

        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(z_q, f0, spec_lengths, self.segment_size)
        # nsf decoder
        o = self.dec(
            z_slice + (torch.randn_like(z_slice) * self.vae_std if self.vae_std > 0 else 0),
            g=g,
            f0=(pitch_slice if self.use_f0 else None)
        )

        # flow
        # forward flow
        z_f, logdet_f = self.flow(z_q, spec_mask, g=g)

        if self.bidirectional_flow_weight > 0:
            # backward flow
            z_r, logdet_r = self.flow(z_p, spec_mask, g=g, reverse=True)
        else:
            z_r, logdet_r = None, None

        return o, ids_slice, spec_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), pred_lf0, norm_lf0, lf0, spk_preds

    def infer(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468, predict_f0=False, vol = None, f0_tran=0):

        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        if self.ssl_dim == 1:
            # embedding if the content feature is index
            c = self.emb_c(c.long()).squeeze(1).transpose(1,2)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if self.character_mix and len(g) > 1:   # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)
        
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        # vol proj
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol!=None and self.vol_embedding else 0
           
        x = self.pre(c) * x_mask + (self.emb_uv(uv.long()).transpose(1,2) if self.use_f0 else 0) + vol
        
        if self.use_f0 and predict_f0:

            uv_sum = torch.sum(uv, dim=1, keepdim=True)
            uv_sum[uv_sum == 0] = 9999
            print("before")
            print(torch.sum(f0 * uv, dim=1, keepdim=True) / uv_sum)
            print(torch.sqrt(torch.sum((f0 * uv) ** 2, dim=1, keepdim=True) / uv_sum - (torch.sum(f0 * uv, dim=1, keepdim=True) / uv_sum) ** 2))

            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
            
            print("after")
            print(torch.sum(f0 * uv, dim=1, keepdim=True) / uv_sum)
            print(torch.sqrt(torch.sum((f0 * uv) ** 2, dim=1, keepdim=True) / uv_sum - (torch.sum(f0 * uv, dim=1, keepdim=True) / uv_sum) ** 2))

            # f0 transform
            f0 = f0 * 2 ** (f0_tran / 12)
            print("transform")
            print(torch.sum(f0 * uv, dim=1, keepdim=True) / uv_sum)
            print(torch.sqrt(torch.sum((f0 * uv) ** 2, dim=1, keepdim=True) / uv_sum - (torch.sum(f0 * uv, dim=1, keepdim=True) / uv_sum) ** 2))
        
        z_p, m_p, logs_p, c_mask, _ = self.enc_p(
            x,
            x_mask,
            f0=(f0_to_coarse(f0) if self.use_f0 else None),
            noice_scale=noice_scale
        )
        z, _ = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=(f0 if self.use_f0 else None))
        return o,f0
