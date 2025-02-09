import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class AutoformerConfig:
    seq_len: int                # input sequence length
    pred_len: int               # forecast horizon
    e_layers: int               # no. of encoder layers
    d_layers: int               # no. of decoder layers
    d_model: int                # dimension of model's hidden states and embeddings
    n_heads: int                # no. of attention heads
    d_ff: Optional[int]         # dimension of feed-forward network in transformer blocks
    kernel_size: Optional[int] = 25       # kernel size for moving average in series decomposition

    # ADDED:
    enc_in: Optional[int] = 7   # encoder input size
    dec_in: Optional[int] = 7   # decoder input size

    # c: Auto-correlation intensity factor
    # Controls the number of time delay steps (k = c * log(L))
    # Typically set between 1-3
    factor: float = 1


class AutoCorrelation(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.n_heads
        self.c = config.factor
        
        # Projections for Q/K/V
        # self.query_proj = nn.Linear(config.d_model, config.d_model)
        # self.key_proj = nn.Linear(config.d_model, config.d_model)
        # self.value_proj = nn.Linear(config.d_model, config.d_model)
        # self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.query_proj = nn.Linear(config.d_model, config.d_model)
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    # def time_delay_agg(self, values: torch.Tensor, corr: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
    #     # Time delay aggregation
    #     batch_size, L, H, d_k = values.shape
    #     output = torch.zeros_like(values)
        
    #     for i, delay in enumerate(delays):
    #         # Roll the values by delay steps
    #         rolled = torch.roll(values, shifts=int(delay), dims=1)
    #         # Weight by correlation score
    #         output += rolled * corr[:, i:i+1, :, None]
            
    #     return output

    # def time_delay_agg(self, values, delays, indices):
    #     # values: [B, L, D]
    #     # delays: [B, k]
    #     rolled_list = []
    #     for b in range(values.size(0)):
    #         delay_b = delays[b]  # shape [k]
    #         # For simplicity, choose one delay (e.g., the largest) or average them
    #         # Or loop over k if you want multiple shifted versions
    #         shift = int(delay_b[0].item())  
    #         rolled_list.append(torch.roll(values[b], shifts=shift, dims=0).unsqueeze(0))
    #     return torch.cat(rolled_list, dim=0)

    def time_delay_agg(self, values, delays, indices):
        # values: [B, L, H, d_k]
        output = torch.zeros_like(values)
        B, L, H, d_k = values.shape
        
        for b in range(B):
            for h in range(H):
                for c in range(d_k):
                    shift = int(delays[b, 0, h, c].item())  # get one shift per head/channel
                    rolled = torch.roll(values[b, :, h, c], shifts=shift, dims=0)
                    output[b, :, h, c] = rolled
        return output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, L, _ = q.shape
        
        # Project and reshape for multi-head
        q = self.query_proj(q).view(batch_size, L, self.n_heads, self.d_k)
        k = self.key_proj(k).view(batch_size, L, self.n_heads, self.d_k)
        v = self.value_proj(v).view(batch_size, L, self.n_heads, self.d_k)

        # Compute autocorrelation for each head
        fft_q = torch.fft.rfft(q, dim=1)
        fft_k = torch.fft.rfft(k, dim=1)
        
        # Cross correlation in frequency domain
        corr = fft_q * torch.conj(fft_k)
        corr = torch.fft.irfft(corr, dim=1)
        
        # Select top-k delays
        num_delays = int(self.c * math.log(L))
        delays, indices = torch.topk(corr[:, 1:], num_delays, dim=1)  # Exclude 0 lag
        delays = F.softmax(delays, dim=-1)
        
        # Time delay aggregation
        out = self.time_delay_agg(v, delays, indices + 1)
        
        # Reshape and project output
        out = out.reshape(batch_size, L, self.d_model)
        return self.out_proj(out)


class MovingAvg(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.kernel_size = config.kernel_size
        self.padding_size = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # padding logic taken from original implementation
        padding = torch.cat([
            x[:, :1, :].repeat(1, self.padding_size, 1),  # front padding
            x,
            x[:, -1:, :].repeat(1, self.padding_size, 1)  # end padding
        ], dim=1)
        
        # compute moving average using avg_pool1d
        return F.avg_pool1d(
            padding.permute(0, 2, 1), 
            kernel_size=self.kernel_size, 
            stride=1
        ).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(config)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.auto_correlation = AutoCorrelation(config)
        self.series_decomp = SeriesDecomp(config)
        self.d_ff = 4 * config.d_model if config.d_ff is None else config.d_ff # d_ff value logic taken from original implementation
        
        # Two-layer feed forward network
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, config.d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Auto-correlation with residual
        auto_out = self.auto_correlation(x, x, x)
        s1, _ = self.series_decomp(auto_out + x)
        
        # Feed forward with residual
        ff_out = self.ff(s1)
        s2, _ = self.series_decomp(ff_out + s1)
        
        return s2


class AutoformerEncoder(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super(AutoformerEncoder, self).__init__()
        self.encoders = nn.ModuleList([AutoformerEncoderLayer(config) for l in range(config.e_layers)])
    
    def forward(self, x: torch.Tensor):
        for encoder in self.encoders:
            x = encoder(x)
        return x


class AutoformerDecoderLayer(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.auto_correlation = AutoCorrelation(config)
        self.cross_correlation = AutoCorrelation(config)
        self.series_decomp = SeriesDecomp(config)
        self.d_ff = 4 * config.d_model if config.d_ff is None else config.d_ff
        
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, config.d_model)
        )
        
        # Trend projection layers
        self.trend_proj1 = nn.Linear(config.d_model, config.d_model)
        self.trend_proj2 = nn.Linear(config.d_model, config.d_model)
        self.trend_proj3 = nn.Linear(config.d_model, config.d_model)
        
    def forward(self, x_seasonal: torch.Tensor, x_trend: torch.Tensor, enc_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention
        auto_out = self.auto_correlation(x_seasonal, x_seasonal, x_seasonal)
        s1, t1 = self.series_decomp(auto_out + x_seasonal)
        trend = x_trend + self.trend_proj1(t1)
        
        # Cross attention
        cross_out = self.cross_correlation(s1, enc_out, enc_out)
        s2, t2 = self.series_decomp(cross_out + s1)
        trend = trend + self.trend_proj2(t2)
        
        # Feed forward
        ff_out = self.ff(s2)
        s3, t3 = self.series_decomp(ff_out + s2)
        trend = trend + self.trend_proj3(t3)
        
        return s3, trend


class AutoformerDecoder(nn.Module):
    def __init__(self, config:AutoformerConfig):
        super().__init__()
        self.config = config
        self.decoders = nn.ModuleList([AutoformerDecoderLayer(config) for l in range(config.d_layers)])
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.series_decomp = SeriesDecomp(config)

    def forward(self, x: torch.Tensor, enc_out:torch.Tensor) -> torch.Tensor:
        # Initialize seasonal and trend components
        I, d = x.shape
        x_ens, x_ent = self.series_decomp(x[I//2:])
        x_des = torch.cat([x_ens, torch.zeros(self.config.pred_len, d)])
        x_det = torch.cat([x_ent, x.mean(dim=0).repeat(self.pred_len, 1)])
        
        # Progressive refinement through decoder layers
        for decoder in self.decoders:
            x_des, x_det = decoder(x_des, x_det, enc_out)
            
        # Final prediction combines both components
        return self.projection(x_des) + x_det
        



class Model(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.enc_in, config.d_model)
        self.dec_embedding = nn.Linear(config.enc_in, config.d_model)
        self.encoder = AutoformerEncoder(config)
        self.decoder = AutoformerDecoder(config).decoders # TODO: replace
        self.output_proj = nn.Linear(config.d_model, 1)
        self.series_decomp = SeriesDecomp(config)
        
    def forward(self, x: torch.Tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None) -> torch.Tensor: # TODO: take care of the unused vars.
        # Split input sequence
        enc_in = x[:, :self.config.seq_len, :]
        
        # Encoder
        enc_out = self.encoder(self.embedding(enc_in))
            
        # Initialize decoder inputs
        I = self.config.seq_len
        dec_init = x[:, I//2:I, :]
        seasonal, trend = self.series_decomp(dec_init)
        
        # Pad seasonal and trend components
        pad_seasonal = torch.zeros(
            (x.size(0), self.config.pred_len, x.size(-1)), 
            device=x.device
        )
        pad_trend = x[:, -1:, :].repeat(1, self.config.pred_len, 1)
        
        seasonal = torch.cat([seasonal, pad_seasonal], dim=1)
        trend = torch.cat([trend, pad_trend], dim=1)
        
        # Decoder
        for dec_layer in self.decoder:
            seasonal, trend = dec_layer(seasonal, trend, enc_out)
            
        # Final prediction
        seasonal = self.output_proj(seasonal)
        
        out = seasonal + trend

        return out[:, -self.pred_len:, :]  # [B, L, D]
