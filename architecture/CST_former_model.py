# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.CST_details.CST_encoder import SingleChannelEncoder

class SlotAttention(nn.Module):
    """
    기존의 복잡한 구조를 버리고, 오직 '정보 수집(Attention)'에만 집중합니다.
    단, 학습 안정성을 위해 Residual + Norm은 남겨둡니다.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, slots, memory):
        # 1. Attention (정보 가져오기)
        # Q=Slots, K=V=Memory
        attn_out, _ = self.mha(query=slots, key=memory, value=memory)
        
        # 2. Residual + Norm (필수 안전장치)
        # (기존 정보 + 가져온 정보)를 합쳐서 안정화
        # 여기서 FFN을 통과시키지 않고 바로 리턴합니다.
        slots = self.norm(slots + self.dropout(attn_out))
        
        return slots

class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.params = params
        self.input_channels = in_feat_shape[1] # 7
        self.nb_classes = params['unique_classes']
        self.rnn_size = params['rnn_size']

        self.num_tracks = int(params.get('num_tracks', 3))
        self.out_per_track = self.nb_classes * 3
        self.out_dim = self.num_tracks * self.out_per_track
        # [수정 1] Shared Encoder 인스턴스 생성
        self.encoder = SingleChannelEncoder(params, in_feat_shape)

        self.channel_embed = nn.Parameter(torch.randn(1, 1, self.input_channels, self.rnn_size))
        self.track_embed = nn.Parameter(torch.randn(1, 1, self.num_tracks, self.rnn_size))
        
        nn.init.xavier_uniform_(self.track_embed)
        nn.init.xavier_uniform_(self.channel_embed)
        
        # 1. Channel Self-Attention (Encoder)
        self.channel_sa = nn.MultiheadAttention(
            embed_dim=self.rnn_size, 
            num_heads=params['nb_heads'], 
            dropout=params['dropout_rate'], 
            batch_first=True
        )
        self.norm_sa = nn.LayerNorm(self.rnn_size)

        # 2. Slot Attention (Decoder)
        # 여기서는 섞기만 하고, 처리는 밑에 있는 fnn_list가 담당합니다.
        self.slot_attention = SlotAttention(
            d_model=self.rnn_size,
            num_heads=params['nb_heads'],
            dropout=params['dropout_rate']
        )

        # # FNN (Classifier)
        # self.fnn_list = torch.nn.ModuleList()
        # if params['nb_fnn_layers']:
        #     for fc_cnt in range(params['nb_fnn_layers']):
        #         self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        # self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))
        
        # self.doa_act = nn.Tanh()
        # FNN (Classifier)
        self.fnn_list = torch.nn.ModuleList()
        # [수정 2] 마지막 Output 차원은 out_shape[-1]이 아니라 
        # 우리가 계산한 self.out_per_track (39) 이어야 정확함
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                in_dim = self.rnn_size if fc_cnt == 0 else params['fnn_size']
                self.fnn_list.append(nn.Linear(in_dim, params['fnn_size'], bias=True))
        
        last_in_dim = params['fnn_size'] if params['nb_fnn_layers'] else self.rnn_size
        self.fnn_list.append(nn.Linear(last_in_dim, self.out_per_track, bias=True))
        
        self.doa_act = nn.Tanh()


    def forward(self, x):
        """
        ALBEF Style Cascade Structure
        Input: (B, 7, T, F)
        """
        B, C, T, F = x.shape
        
        # --- Step 1: Encoder (Channel-Independent) ---
        # (B, 7, T, F) -> (B*7, 1, T, F) -> (B*7, T, H)
        x_reshaped = x.reshape(B * C, 1, T, F)
        features_flat = self.encoder(x_reshaped) #(896,50,128) 896=batch(128)*7(channel)
        
        # 2. Key/Value 준비: "전체 문맥(Global Context)"
        # 각 Query(채널)가 "다른 모든 채널"을 참조할 수 있게 함.
        # (B*7, T, H) -> (B, 7, T, H)
        T_out = features_flat.shape[1] # 실제 줄어든 시간 길이 가져오기 50
        H = features_flat.shape[2]
        features = features_flat.view(B, C, T_out, H)

        # Memory Setup
        memory = features.permute(0, 2, 1, 3).reshape(B * T_out, C, H)
        memory = memory + self.channel_embed.squeeze(0)

        mask = torch.zeros(C, C, device=x.device)
        mask.fill_diagonal_(float('-inf'))

        # Channel SA
        sa_out, _ = self.channel_sa(memory, memory, memory, attn_mask=mask)
        memory = self.norm_sa(memory + sa_out)

        # Slot Cross-Attention (Simple)
        queries = self.track_embed.expand(B, T_out, -1, -1).reshape(B * T_out, self.num_tracks, H)
        updated_slots = self.slot_attention(queries, memory)

        # # -------------------------------------------------------
        # # [Step 3] Main Task (DOA Prediction)
        # # -------------------------------------------------------
        # # 정제된 특징을 모두 합쳐서 DOA 예측
        # # (B, 7, T, H) -> (B, T, 7*H)
        # for fnn_cnt in range(len(self.fnn_list) - 1):
        #     doa_feat = self.fnn_list[fnn_cnt](updated_slots)
        # doa = self.fnn_list[-1](doa_feat)
        # doa = self.doa_act(doa)

        # return doa
        # [수정 5] FNN 루프 로직 수정 (입력 업데이트 & ReLU 추가)
        doa_feat = updated_slots # 초기 입력 설정
        
        for fnn_cnt in range(len(self.fnn_list) - 1):
            doa_feat = self.fnn_list[fnn_cnt](doa_feat) # 이전 결과(doa_feat)를 넣음
            doa_feat = F.relu(doa_feat)                 # 활성화 함수 필수!
            # doa_feat = F.dropout(doa_feat, p=self.params['dropout_rate'], training=self.training)
            
        doa = self.fnn_list[-1](doa_feat)
        doa = self.doa_act(doa)

        # [수정 6] Shape 복구 (B*T -> B, T)
        # (B*T, Slot, Out) -> (B, T, Slot*Out)
        doa = doa.view(B, T_out, self.out_dim)

        return doa