# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x
    
class SingleChannelEncoder(nn.Module):
    def __init__(self, params, in_feat_shape):
        super().__init__()
        self.params = params
        
        # 1. CNN Blocks
        # 입력 채널은 무조건 1로 고정 (채널 독립적 처리를 위해)
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(
                    in_channels=params['nb_cnn2d_filt'] if conv_cnt else 1, 
                    out_channels=params['nb_cnn2d_filt']
                ))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        # GRU 입력 차원 계산
        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        
        # 2. GRU
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)
        
        # 3. Multi-Head Self-Attention
        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

    def forward(self, x):
        # x: (B, 1, T, F)
        # CNN
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
            
        # Reshape for GRU
        # (B, C, T, F) -> (B, T, C*F)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        
        # GRU
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2] # Bidirectional sum/slice
        
        # Self-Attention
        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)
            
        return x # (B, T, H)

class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.params = params
        self.input_channels = in_feat_shape[1] # 7
        self.nb_classes = params['unique_classes']
        
        # [수정 1] Shared Encoder 인스턴스 생성
        self.encoder = SingleChannelEncoder(params, in_feat_shape)
        
        # [수정 2] DOA 추정용 Fusion Layer
        # 7개의 채널 특징(rnn_size)을 합쳐서 하나로 만듦
        # 입력: 7 * rnn_size -> 출력: rnn_size
        self.doa_fusion = nn.Linear(self.input_channels * params['rnn_size'], params['rnn_size'])

        # [수정 3] Cross-Attentive Contrastive Learning 구성요소
        # One-vs-Rest Attention을 위한 모듈
        self.cross_attn = nn.MultiheadAttention(embed_dim=params['rnn_size'], num_heads=4, batch_first=True)
        
        # Context Projection: (N-1)개의 특징을 합친 것을 다시 rnn_size로 압축
        self.ctx_proj = nn.Linear((self.input_channels) * params['rnn_size'], params['rnn_size'])
        
        # Projection Head (Contrastive Loss용)
        self.proj_head = nn.Sequential(
            nn.Linear(params['rnn_size'], 128),
            nn.LayerNorm(128)
        )

        # FNN (Classifier)
        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))
        
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
        features_flat = self.encoder(x_reshaped)
        # (B, 7, T, H)
        # 1. Query 준비: (B*7, T, H)
        # 7개 채널 각각이 자신의 '보정된 값'을 찾기 위해 Query가 됩니다.
        # 이미 features_flat이 (B*7, ...) 형태라 그대로 쓰면 됨.
        queries = features_flat 
        
        # 2. Key/Value 준비: "전체 문맥(Global Context)"
        # 각 Query(채널)가 "다른 모든 채널"을 참조할 수 있게 함.
        # (B*7, T, H) -> (B, 7, T, H)
        T_out = features_flat.shape[1] # 실제 줄어든 시간 길이 가져오기
        features_unfold = features_flat.view(B, C, T_out, -1)
        
        # 7개 채널을 Feature 축으로 모두 합침 -> (B, T, 7*H)
        # 이것이 "모든 채널의 정보가 담긴 덩어리"입니다.
        global_context = features_unfold.permute(0, 2, 1, 3).reshape(B, T_out, -1)
        
        # 차원 압축 (7*H -> H) -> (B, T, H)
        keys_compact = self.ctx_proj(global_context)
        
        # [중요] 배치를 7배로 뻥튀기 (Expand)
        # Query는 (B*7)개인데 Key는 (B)개면 안 맞음.
        # Key도 (B*7)개로 복사해줍니다. (메모리 복사 없이 참조만 복사됨)
        # (B, T, H) -> (B, 1, T, H) -> (B, 7, T, H) -> (B*7, T, H)
        keys_expanded = keys_compact.unsqueeze(1).expand(-1, C, -1, -1).reshape(B * C, T_out, -1)
        
        # 3. Cross-Attention (한 방에 수행!)
        # Q: (B*7, T, H)
        # K, V: (B*7, T, H)
        # GPU는 이걸 그냥 "배치가 큰 한 번의 연산"으로 처리합니다.
        attn_out, _ = self.cross_attn(queries, keys_expanded, keys_expanded)
        
        # 4. Residual Connection (보정)
        refined_flat = queries + attn_out
        
        # 다시 (B, 7, T, H) 구조로 복구
        refined_global_feat = refined_flat.view(B, C, T_out, -1)

        # -------------------------------------------------------
        # [Step 3] Main Task (DOA Prediction)
        # -------------------------------------------------------
        # 정제된 특징을 모두 합쳐서 DOA 예측
        # (B, 7, T, H) -> (B, T, 7*H)
        fusion_input = refined_global_feat.permute(0, 2, 1, 3).reshape(B, T_out, -1)
        
        doa_feat = self.doa_fusion(fusion_input)
        
        for fnn_cnt in range(len(self.fnn_list) - 1):
            doa_feat = self.fnn_list[fnn_cnt](doa_feat)
        doa = self.fnn_list[-1](doa_feat)
        doa = self.doa_act(doa)

        # # -------------------------------------------------------
        # # [Step 4] Auxiliary Task (Contrastive Loss)
        # # -------------------------------------------------------
        # proj_pred = None
        # proj_target = None
        
        # if self.training:
        #     # 랜덤 채널 하나 선택 (예: 3번 채널)
        #     target_idx = torch.randint(0, C, (1,)).item()
            
        #     # Prediction: Decoder 통과한 놈 (Refined)
        #     pred_feat = refined_global_feat[:, target_idx, :, :]
            
        #     # Target: Encoder 원본 (Original)
        #     # features_unfold는 위에서 만들어둠 (B, C, T, H)
        #     target_feat = features_unfold[:, target_idx, :, :]
            
        #     proj_pred = self.proj_head(pred_feat)
        #     proj_target = self.proj_head(target_feat)

        # return doa, proj_pred, proj_target
        return doa