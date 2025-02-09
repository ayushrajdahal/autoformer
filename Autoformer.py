import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
from layers.Embed import TokenEmbedding, PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding
import math


@dataclass
class AutoformerConfig:
    kernel_size: int        # kernel size for moving average in series decomposition
    seq_len: int            # input sequence length
    label_len: int          # label length
    pred_len: int           # forecast horizon
    e_layers: int         # no. of encoder layers
    d_layers: int         # no. of decoder layers
    d_model: int            # dimension of model's hidden states and embeddings
    n_heads: int            # no. of attention heads
    d_ff: int               # dimension of feed-forward network in transformer blocks
    target:str             # target time series to forecast
    
    # c: Auto-correlation intensity factor
    # Controls the number of time delay steps (k = c * log(L))
    # Typically set between 1-3
    factor: float
    enc_in: int            # no. of encoder input features
    dec_in: int            # no. of decoder input features
    c_out: int             # output size

    # OPTIONAL HYPERPARAMETERS
    root_path: str = "./dataset/ETT-small/",   # path to dataset
    data_path: str = "ETTh1.csv",
    model_id: str = "ETTh1_96_24",
    model: str = "Autoformer",
    data: str = "ETTh1",
    features: str = "M",
    des: str = "Exp",
    itr: int = 1,
    is_training:int=1,
    patience:int=7,
    device:str="cuda" if torch.cuda.is_available() else "cpu",
    learning_rate:float=0.0001,
    embed:str="timeF",
    batch_size:int=32,
    freq:str="h",


class AutoCorrelation(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.factor = config.factor
        self.scale = None
        self.mask_flag = True
        self.output_attention = config.output_attention
        self.dropout = nn.Dropout(config.dropout)
        
        # Projections
        self.query_proj = nn.Linear(config.d_model, config.d_model)
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def time_delay_agg_training(self, values, corr):
        # Implementation from original code
        B, H, E, L = values.shape
        # Top k correlation selection
        top_k = int(self.factor * math.log(L))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # Softmax correlation weights
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Aggregation
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), dims=-1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1))
        return delays_agg
        
    def time_delay_agg_inference(self, values, corr):
        # Similar to original implementation
        B, H, E, L = values.shape
        # Initialize index
        init_index = torch.arange(L).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, H, E, 1).to(values.device)
        
        # Top k correlation selection
        top_k = int(self.factor * math.log(L))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        
        # Softmax correlation weights
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return delays_agg

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        
        # Handle variable sequence lengths
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L-S), :]).float() 
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
            
        # Period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # Time delay aggregation
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr)
            
        V = V.permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None

# Data Embedding w/o positional encoding: taken from the original implementation
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


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
        self.encoders = nn.ModuleList([AutoformerEncoderLayer(config) for _ in range(config.e_layers)])
    
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
        self.output_proj = nn.Linear(config.d_model, config.c_out)
        self.series_decomp = SeriesDecomp(config)

    def forward(self, x: torch.Tensor, enc_out:torch.Tensor) -> torch.Tensor:
        # Initialize seasonal and trend components
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

        # x_ens, x_ent = self.series_decomp(x[I//2:])
        # x_des = torch.cat([x_ens, torch.zeros(self.config.pred_len, d)])
        # x_det = torch.cat([x_ent, x.mean(dim=0).repeat(self.pred_len, 1)])
        
        # Progressive refinement through decoder layers
        for decoder in self.decoders:
            seasonal, trend = decoder(seasonal, trend, enc_out)
            
        # Final prediction combines both components
        return self.projection(seasonal) + trend


class Model(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.config = config
        self.enc_embedding = DataEmbedding_wo_pos(config.enc_in, config.d_model, config.embed, config.freq, config.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(config.dec_in, config.d_model, config.embed, config.freq, config.dropout)
        self.encoder = AutoformerEncoder(config)
        self.decoder = AutoformerDecoder(config) # TODO: replace
        self.output_proj = nn.Linear(config.d_model, 1)
        self.series_decomp = SeriesDecomp(config)
        
    def forward(self, x: torch.Tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None) -> torch.Tensor: # TODO: take care of the unused vars.
        # Split input sequence
        enc_in = x[:, :self.config.seq_len, :]
        
        # Encoder
        enc_out = self.encoder(self.enc_embedding(enc_in))
        
        # Decoder
        dec_out = self.decoder(self.dec_embedding(x_dec), enc_out)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
