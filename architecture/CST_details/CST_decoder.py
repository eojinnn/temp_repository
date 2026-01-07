import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

class CorrelatedAttentionBlock(nn.Module):
    """
    [FOA Optimized Version]
    입력  x: (B, C, T, F)
    출력 out: (B, C, T, F)
    
    기능:
    - 시간 지연(Delay) 로직 제거 (FOA는 TDOA가 0이므로)
    - 주파수 별(Frequency-wise) 채널 상관관계(Covariance) 학습에 집중
    - Channel Self-Attention 메커니즘 유지
    """
    def __init__(self, embed_dim, params):
        super().__init__()        
        self.embed_dim = embed_dim
        self.num_heads = params['nb_heads']
        self.dropout_rate = params['dropout_rate']
        
        # embed_dim이 head로 나누어 떨어지는지 확인
        assert embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim  = embed_dim // self.num_heads

        # Q, K, V: 채널-선형사상 (1x1 Conv와 유사)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout  = nn.Dropout(self.dropout_rate)

        # 어텐션 스케일 (Learnable temperature)
        self.log_tau  = nn.Parameter(torch.zeros(1))

    # ---------------------------
    # 유틸: 시간축 L2 정규화 (유지: 에너지 정규화 측면에서 유효함)
    # ---------------------------
    @staticmethod
    def _l2_norm_over_time_2d(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # x: (B,H,T,F,Dh)
        denom = torch.sqrt(torch.clamp((x ** 2).sum(dim=2, keepdim=True), min=eps))
        return x / denom

    # ---------------------------
    # 내부 유틸: 헤드→채널 병합
    # ---------------------------
    @staticmethod
    def _merge_heads_to_channels(x_bhtfd: torch.Tensor) -> torch.Tensor:
        # (B,H,T,F,Dh) → (B,C,T,F)
        B,H,T,F,Dh = x_bhtfd.shape
        return x_bhtfd.permute(0,1,4,2,3).reshape(B, H*Dh, T, F)

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        x: (B, C, T, F) → out: (B, C, T, F)
        """
        B, C, T, Fq = x.shape
        H, Dh = self.num_heads, self.head_dim

        # 1. (t,f) 위치별 선형사상
        # xf: (B,T,F,C)
        xf = x.permute(0, 2, 3, 1).contiguous()
        Q = self.q_proj(xf)
        K = self.k_proj(xf)
        V = self.v_proj(xf)

        # 2. 헤드 분할: (B, T, F, H, Dh) -> (B, H, T, F, Dh)
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, Fq, H, Dh).permute(0, 3, 1, 2, 4).contiguous()
        
        Qh, Kh, Vh = split(Q), split(K), split(V)

        # 3. 시간축 정규화 (옵션: 필요에 따라 제거 가능하나, 안정성을 위해 유지 추천)
        Qh_hat = self._l2_norm_over_time_2d(Qh)
        Kh_hat = self._l2_norm_over_time_2d(Kh)

        # 4. 채널-채널 어텐션 (Instant Correlation)
        # FOA의 핵심: "이 주파수에서 W, X, Y, Z 채널 간의 공분산 패턴은?"
        tau = torch.exp(self.log_tau).clamp(1e-4, 10.0)
        
        # (B,H,T,F,Dh) -> (B,H,F,Dh,T)
        KT = Kh_hat.permute(0, 1, 3, 4, 2) 
        # (B,H,T,F,Dh) -> (B,H,F,T,Dh)
        QT = Qh_hat.permute(0, 1, 3, 2, 4)
        
        # 공분산 계산 (Covariance Matrix): (B,H,F,Dh,Dh)
        # T(시간) 차원이 사라짐 -> 즉, 시간 평균적인 채널 간 관계를 주파수 별로 계산
        cov = KT @ QT  
        
        # Softmax: 채널 간의 Attention Score 생성
        att = (cov / tau).softmax(dim=-1)

        # 5. Value에 어텐션 적용
        # (B,H,F,T,Dh) @ (B,H,F,Dh,Dh) -> (B,H,F,T,Dh)
        Vtf  = Vh.permute(0, 1, 3, 2, 4)
        out_h = (Vtf @ att).permute(0, 1, 3, 2, 4).contiguous() # (B,H,T,F,Dh)

        # 6. 헤드 병합 및 출력 투영
        out_c = self._merge_heads_to_channels(out_h) # (B,C,T,F)
        out_c = self.dropout(out_c)
        
        out_tf = out_c.permute(0, 2, 3, 1).contiguous() # (B,T,F,C)
        out_tf = self.out_proj(out_tf)
        out    = out_tf.permute(0, 3, 1, 2).contiguous() # (B,C,T,F)

        if need_weights:
            return out, {"att": att, "tau": tau.detach()}
        else:
            return out, None