import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

class CorrelatedAttentionBlock(nn.Module):
    """
    입력  x: (B, C, T, F)   # C=embed_dim, T=시간 프레임, F=주파수 bin
    출력 out: (B, C, T, F)

    - (t,f) 위치별 채널-선형사상으로 Q,K,V 생성 → 헤드 분할
    - 시간축(T) L2 정규화 후 채널-채널 어텐션(열 softmax)
    - 채널별 지연(learnable):
        * 'sinc'     : windowed-sinc FIR (Depth-wise Conv1d)
        * 'lagrange4': 3차 Lagrange 4-tap FIR (Depth-wise Conv1d)
        * 'grid'     : grid_sample (좌표 수정 + AMP 안전 캐스팅)
    - 블렌딩(blend_impl):
        * 'scalar' : 스칼라 β 혼합 (기본)
        * 'vector' : β_(h,c) 채널별 혼합
        * 'gate'   : 동적 게이트(Linear)
        * 'conv'   : 1×1 Conv2d (옵션: DW 3×3) 블렌더
    """
    def __init__(self, embed_dim, params):
        super().__init__()        
        learnable_ch_lag = True
        max_abs_lag = 0.2
        value_roll = False
        delay_impl = "grid"            # 'sinc'|'lagrange4'|'grid'
        fir_kernel_size = 12
        fir_window = "hamming"
        pad_mode = "replicate"
        # blending
        blend_impl = "gate"          # 'scalar'|'vector'|'gate'|'conv'
        use_dw_context = True        # conv 블렌더에서 DW 3×3 사용할지

        self.dropout_rate = params['dropout_rate']
        self.embed_dim = embed_dim
        self.num_heads = params['nb_heads']
        self.head_dim  = embed_dim // self.num_heads

        # Q, K, V: 채널-선형사상
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout  = nn.Dropout(self.dropout_rate)

        # 어텐션 스케일/혼합
        self.log_tau  = nn.Parameter(torch.zeros(1))


        #블랜딩 Instant 어텐션 결과와 지연 어텐션 결과를 어떻게 섞을지 결정정
        self.blend_impl = blend_impl
        self.beta_lag = nn.Parameter(torch.tensor(0.5))  # scalar 용
        if blend_impl == "vector":
            self.beta_param = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))  # (H,Dh)
        elif blend_impl == "gate":
            self.beta_gate = nn.Linear(2*self.head_dim, self.head_dim)
            nn.init.zeros_(self.beta_gate.weight)
            nn.init.zeros_(self.beta_gate.bias)

        # 지연 파라미터
        self.learnable_ch_lag = learnable_ch_lag
        self.max_abs_lag = float(max_abs_lag)
        self.value_roll  = value_roll
        if learnable_ch_lag:
            self.lag_raw = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))  # (H, Dh)

        # 지연 구현 옵션
        self.delay_impl = delay_impl
        self.fir_kernel_size = int(fir_kernel_size)
        self.fir_window = fir_window
        self.pad_mode = pad_mode
        self.use_dw_context = use_dw_context
        
        assert embed_dim % self.num_heads == 0, "embed_dim % num_heads == 0 이어야 합니다."
        assert delay_impl in ("sinc", "lagrange4", "grid")
        assert blend_impl in ("scalar", "vector", "gate")
    # ---------------------------
    # 유틸: 시간축 L2 정규화
    # ---------------------------
    @staticmethod
    def _l2_norm_over_time_2d(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # x: (B,H,T,F,Dh)  — 시간축(T)에 대해 L2 정규화
        denom = torch.sqrt(torch.clamp((x ** 2).sum(dim=2, keepdim=True), min=eps))
        return x / denom

    # ---------------------------
    # grid_sample 기반 분수 지연 (1D -> 2D)
    #  - 좌표(x/y) 수정 + AMP 안전 캐스팅
    # ---------------------------
    def _shift_per_channel_grid_1d(self, x: torch.Tensor, lag_hc: torch.Tensor,
                                   padding_mode: str = "border") -> torch.Tensor:
        # x: (B,H,T,Dhf)  / lag_hc: (H,Dhf)  → (B,H,T,Dhf)
        B, H, T, Dhf = x.shape
        device = x.device
        orig_dtype = x.dtype

        i   = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, T, 1) #원래 시간표
        lag = lag_hc.view(1, H, 1, Dhf).to(torch.float32)
        src = i - lag  # (1,H,T,Dhf) #에서 지연시킬 만큼 뺌.

        if T > 1:   
            y_norm = 2.0 * src / float(T - 1) - 1.0  # -1에서 1 사이로 지연 시간 인덱스 정규화 넘어가는 값은 border 패딩 처리.
        else:
            y_norm = torch.zeros_like(src)
        x_fixed = torch.zeros_like(y_norm)  # x(W=1)=0

        X = x.permute(0, 1, 3, 2).contiguous().view(B * H * Dhf, 1, T, 1)    # batch, channel, height, width
        grid = torch.stack([                        #(x,y(시간 인덱스)) 좌표 그리드 생성
            x_fixed.expand(B, H, T, Dhf).permute(0, 1, 3, 2).reshape(B * H * Dhf, T, 1),
            y_norm .expand(B, H, T, Dhf).permute(0, 1, 3, 2).reshape(B * H * Dhf, T, 1)
        ], dim=-1) #(B*H*Dhf,T,1,2)

        Y = F.grid_sample( 
            X.float(), grid.float(),
            mode="bilinear", padding_mode=padding_mode, align_corners=True
        )
        y = Y.to(orig_dtype).view(B, H, Dhf, T).permute(0, 1, 3, 2).contiguous()
        return y

    def _shift_per_channel_grid_2d(self, x: torch.Tensor, lag_hc: torch.Tensor,
                                   padding_mode: str = "border") -> torch.Tensor:
        # x: (B,H,T,F,Dh), lag_hc: (H,Dh) → (B,H,T,F,Dh)
        B, H, T, Fq, Dh = x.shape
        x_flat = x.permute(0, 1, 2, 4, 3).contiguous().view(B, H, T, Dh * Fq)
        lag_rep = lag_hc.unsqueeze(-1).expand(H, Dh, Fq).reshape(H, Dh * Fq)
        y_flat = self._shift_per_channel_grid_1d(x_flat, lag_rep, padding_mode)
        y = y_flat.view(B, H, T, Dh, Fq).permute(0, 1, 2, 4, 3).contiguous()
        return y

    # ---------------------------
    # FIR 커널 생성: windowed-sinc / Lagrange-4
    # ---------------------------
    def _make_windowed_sinc_kernel(self, delays_hd: torch.Tensor,
                                   K: Optional[int] = None,
                                   window: str = "hamming",
                                   eps: float = 1e-8) -> torch.Tensor:
        # delays_hd: (H,Dh) → (H,Dh,K)
        H, Dh = delays_hd.shape
        if K is None:
            K = self.fir_kernel_size
        device, dtype = delays_hd.device, delays_hd.dtype

        n = torch.arange(K, device=device, dtype=dtype).view(1, 1, K)
        D = delays_hd.view(H, Dh, 1)
        h = torch.sinc(n - D)

        if window == "hamming":
            nn = torch.arange(K, device=device, dtype=dtype)
            w = 0.54 - 0.46 * torch.cos(2 * math.pi * nn / (K - 1))
            w = w.view(1, 1, K)
        else:
            w = torch.ones(1, 1, K, device=device, dtype=dtype)

        h = h * w
        h = h / h.sum(dim=-1, keepdim=True).clamp_min(eps)
        h = torch.flip(h, dims=[-1])  # correlation 보정
        return h

    def _make_lagrange4_kernel(self, delays_hd: torch.Tensor) -> torch.Tensor:
        # delays_hd: (H,Dh) → (H,Dh,4)
        H, Dh = delays_hd.shape
        delta = torch.remainder(delays_hd, 1.0).view(H, Dh, 1)
        d = delta
        c_m1 = -d * (1 - d) * (2 - d) / 6.0
        c_0  =  (d + 1) * (d - 1) * (d - 2) / 2.0
        c_1  = -(d + 1) * d * (d - 2) / 2.0
        c_2  =  (d + 1) * d * (d - 1) / 6.0
        h = torch.cat([c_m1, c_0, c_1, c_2], dim=-1)
        h = torch.flip(h, dims=[-1])  # correlation 보정
        return h

    # ---------------------------
    # FIR 적용 (Depth-wise Conv1d)
    # ---------------------------
    def _apply_fracdelay_fir_2d(self, x: torch.Tensor, delays_hd: torch.Tensor,
                                kernel_maker, pad_mode: str = "replicate") -> torch.Tensor:
        # x: (B,H,T,F,Dh), delays_hd: (H,Dh) → (B,H,T,F,Dh)
        B, H, T, Fq, Dh = x.shape
        h = kernel_maker(delays_hd)                     # (H,Dh,K)
        K = h.shape[-1]

        h_rep = h.unsqueeze(2).expand(H, Dh, Fq, K).reshape(H, Dh * Fq, K)
        weight = h_rep.reshape(H * Dh * Fq, 1, K)

        xin = x.permute(0, 1, 3, 4, 2).contiguous().view(B, H * Fq * Dh, T)

        pad_left  = K // 2
        pad_right = K - 1 - pad_left
        xin_pad = F.pad(xin, (pad_left, pad_right), mode=pad_mode)

        y = F.conv1d(xin_pad, weight=weight, bias=None, stride=1, padding=0,
                     groups=H * Fq * Dh)
        y = y.view(B, H, Fq, Dh, T).permute(0, 1, 4, 2, 3).contiguous()
        return y

    # ---------------------------
    # 지연 적용 디스패처
    # ---------------------------
    def _shift_per_channel(self, x: torch.Tensor, lag_hc: torch.Tensor) -> torch.Tensor:
        # x: (B,H,T,F,Dh), lag_hc: (H,Dh)
        if self.delay_impl == "grid":
            return self._shift_per_channel_grid_2d(x, lag_hc, padding_mode="border")
        elif self.delay_impl == "lagrange4":
            return self._apply_fracdelay_fir_2d(x, lag_hc, self._make_lagrange4_kernel,
                                                pad_mode=self.pad_mode)
        elif self.delay_impl == "sinc":
            maker = lambda d: self._make_windowed_sinc_kernel(
                d, K=self.fir_kernel_size, window=self.fir_window
            )
            return self._apply_fracdelay_fir_2d(x, lag_hc, maker, pad_mode=self.pad_mode)
        else:
            raise ValueError(f"Unknown delay_impl: {self.delay_impl}")

    # ---------------------------
    # 내부 유틸: 헤드→채널 병합
    # ---------------------------
    @staticmethod
    def _merge_heads_to_channels(x_bhtfd: torch.Tensor) -> torch.Tensor:
        # (B,H,T,F,Dh) → (B,C,T,F)
        B,H,T,F,Dh = x_bhtfd.shape
        return x_bhtfd.permute(0,1,4,2,3).reshape(B, H*Dh, T, F)

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        x: (B, C, T, F) → out: (B, C, T, F)
        """
        B, C, T, Fq = x.shape
        H, Dh = self.num_heads, self.head_dim
        assert C == self.embed_dim, "embed_dim은 입력 채널 C와 같아야 합니다."

        # (t,f) 위치별 채널-선형사상
        xf = x.permute(0, 2, 3, 1).contiguous()  # (B,T,F,C)
        Q = self.q_proj(xf)
        K = self.k_proj(xf)
        V = self.v_proj(xf)

        # 헤드 분할: (B,H,T,F,Dh)
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, Fq, H, Dh).permute(0, 3, 1, 2, 4).contiguous()
        Qh, Kh, Vh = split(Q), split(K), split(V)

        # 시간축 L2 정규화
        Qh_hat = self._l2_norm_over_time_2d(Qh)
        Kh_hat = self._l2_norm_over_time_2d(Kh)

        # 채널-채널 어텐션 (즉시)
        tau = torch.exp(self.log_tau).clamp(1e-4, 10.0)
        KT = Kh_hat.permute(0, 1, 3, 4, 2)  # (B,H,F,Dh,T)
        QT = Qh_hat.permute(0, 1, 3, 2, 4)  # (B,H,F,T,Dh)
        cov0 = KT @ QT                      # (B,H,F,Dh,Dh)
        att0 = (cov0 / tau).softmax(dim=-1)

        Vtf  = Vh.permute(0, 1, 3, 2, 4)    # (B,H,F,T,Dh)
        inst = (Vtf @ att0).permute(0, 1, 3, 2, 4).contiguous()  # (B,H,T,F,Dh)

        # 지연 적용
        if self.learnable_ch_lag:
            lag = self.max_abs_lag * torch.tanh(self.lag_raw)     # (H,Dh)
            Kshift = self._shift_per_channel(Kh_hat, lag)
            KTl = Kshift.permute(0, 1, 3, 4, 2)                   # (B,H,F,Dh,T)
            cov_l = KTl @ QT                                      # (B,H,F,Dh,Dh)
            att_l = (cov_l / tau).softmax(dim=-1)

            Vsel = self._shift_per_channel(Vh, lag) if self.value_roll else Vh
            Vsel_tf = Vsel.permute(0, 1, 3, 2, 4)                 # (B,H,F,T,Dh)
            lagged  = (Vsel_tf @ att_l).permute(0, 1, 3, 2, 4).contiguous()
        else:
            lag    = None
            att_l  = None
            lagged = torch.zeros_like(inst)

        # ---------------------------
        # 블렌딩: out_c ∈ (B,C,T,F)
        # ---------------------------
        if self.blend_impl == "scalar":
            beta  = torch.clamp(self.beta_lag, 0.0, 1.0)                  # scalar
            out_h = (1.0 - beta) * inst + beta * lagged                   # (B,H,T,F,Dh)
            out_c = self._merge_heads_to_channels(out_h)                  # (B,C,T,F)
        elif self.blend_impl == "vector":
            beta_map = torch.sigmoid(self.beta_param).view(1, H, 1, 1, Dh)  # (1,H,1,1,Dh)
            out_h = (1.0 - beta_map) * inst + beta_map * lagged
            out_c = self._merge_heads_to_channels(out_h)
        elif self.blend_impl == "gate":
            z = torch.cat([inst, lagged], dim=-1)                         # (B,H,T,F,2*Dh)
            beta_map = torch.sigmoid(self.beta_gate(z))                   # (B,H,T,F,Dh)
            out_h = (1.0 - beta_map) * inst + beta_map * lagged
            out_c = self._merge_heads_to_channels(out_h)
        else:
            raise ValueError(f"Unknown blend_impl: {self.blend_impl}")

        # 드롭아웃 → 출력 투영
        out_c = self.dropout(out_c)                                       # (B,C,T,F)
        out_tf_d = out_c.permute(0, 2, 3, 1).contiguous()                 # (B,T,F,C)
        out_tf_d = self.out_proj(out_tf_d)
        out      = out_tf_d.permute(0, 3, 1, 2).contiguous()              # (B,C,T,F)

        if need_weights:
            weights = {
                "att_instant": att0,                 # (B,H,F,Dh,Dh)
                "att_lagged":  att_l,                # (B,H,F,Dh,Dh) | None
                "lag":         lag,                  # (H,Dh) | None
                "tau":         tau.detach(),
                "blend_impl":  self.blend_impl,
                "beta_scalar": torch.clamp(self.beta_lag, 0.0, 1.0).detach() if self.blend_impl=="scalar" else None,
                "fir_kernel_size": self.fir_kernel_size if self.delay_impl != "grid" else None,
                "delay_impl":  self.delay_impl,
                "use_dw_context": self.use_dw_context,
            }
            return out, weights
        else:
            return out, None