import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import utility.lr_sched as lr_sched
import matplotlib.pyplot as plot
import torch.nn.functional as F
plot.switch_backend('agg')

def energy_to_linear_from_power_db(db_mel: torch.Tensor,
                                  eps: float = 1e-8,
                                  per_frame_relative: bool = False) -> torch.Tensor:
    """
    db_mel: (B,T,F)  # librosa.power_to_db 출력 (power dB)
    per_frame_relative:
        True면 프레임별 max를 0dB로 맞춰서 (0,1] 범위의 '상대 에너지 가중치'로 사용.
        False면 절대 dB를 그대로 linear power로 변환 (스케일 폭이 매우 클 수 있음).
    """
    if per_frame_relative:
        db_mel = db_mel - db_mel.amax(dim=-1, keepdim=True)  # max=0dB
    E = torch.pow(10.0, db_mel / 10.0)  # power dB -> linear power
    return E.clamp_min(eps)

def iv_direction_loss(v_net: torch.Tensor,
                         gt_iv: torch.Tensor,
                         real_energy_db: torch.Tensor,
                         tau_E: float = 1e-6,
                         tau_D: float = 0.95,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    논문 의도에 맞춘 TF-bin 기반 IV 방향 정렬 loss.

    Args:
        v_net: (B, T, 3)            - 네트워크가 예측한 프레임별 방향 벡터
        gt_iv: (B, 3, T, F)         - GT 정규화 IV (I/E 계열이라면 norm이 방향성(1-D) 근사)
        real_energy_db: (B, T, F)   - W 채널 mel-spectrogram (librosa.power_to_db 출력, dB)
        tau_E:  에너지 마스크 임계값 (linear domain)
        tau_D:  diffuseness 임계값 (논문 0.95)
        eps:    수치 안정화

    Returns:
        scalar loss
    """
    # (B,3,T,F) -> (B,T,F,3)
    gt = gt_iv.permute(0, 2, 3, 1).contiguous()  # (B,T,F,3)

    # v_net: (B,T,3) -> (B,T,1,3), unit vector
    v = F.normalize(v_net, p=2, dim=-1, eps=eps).unsqueeze(2)  # (B,T,1,3)

    # GT도 방향 비교를 위해 unit vector (0벡터 대비 eps)
    gt_dir = F.normalize(gt, p=2, dim=-1, eps=eps)  # (B,T,F,3)

    # (t,f)마다 cosine
    cos_tf = (v * gt_dir).sum(dim=-1)  # (B,T,F)
    raw_tf = 1.0 - cos_tf              # (B,T,F)

    # E(t,f): dB -> linear power
    E = energy_to_linear_from_power_db(real_energy_db, eps=eps, per_frame_relative=True)  # (B,T,F)

    # (1-D) 근사: 정규화 IV의 크기(방향성/코히어런스가 강할수록 커짐)
    dir_tf = gt.norm(dim=-1)  # (B,T,F)

    # 논문 마스크: E > tau_E AND D < tau_D
    # 여기서 D < tau_D 를 dir_tf > (1 - tau_D)로 근사(= 0.05)
    mask = (E > tau_E) & (dir_tf > (1.0 - tau_D))
    w = E * dir_tf * mask.float()  # (B,T,F)

    w_sum = w.sum()
    if w_sum < eps:
        # 학습 안정성: 가중치가 전부 0이면 loss를 0으로
        return torch.zeros((), device=v_net.device, requires_grad=True)

    loss = (w * raw_tf).sum() / (w_sum + eps)
    return loss

def train_epoch(data_generator, optimizer, model, criterion, params, device, epoch_cnt, total_batches):
    nb_train_batches, train_loss = 0, 0.
    scaler = GradScaler()
    model.train()

    #total_batches = data_generator.get_total_batches_in_data()
    aux_loss = params.get('aux_loss', False)
    lambda_align = params.get('lambda_aux_loss_out', 0.00)

    with tqdm(total=total_batches) as pbar:
        for data, target in data_generator:
            # Learning rate scheduler logic (기존 동일)
            if params['lr_scheduler']:
                if params['lr_by_epoch']:
                    lr_sched.adjust_lr_by_epoch(optimizer, nb_train_batches / total_batches + epoch_cnt, params)
                elif params['lr_ramp']:
                    lr_sched.adjust_learning_rate_ramp(optimizer, nb_train_batches / total_batches + epoch_cnt, params)
                else:
                    lr_sched.adjust_learning_rate(optimizer, nb_train_batches / total_batches + epoch_cnt, params)

            #data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
            data, target = data.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            
            with autocast():
                # 모델 Forward
                output = model(data.contiguous())
                if aux_loss and isinstance(output, tuple):
                    doa_pred, aux_dict = output
                    pred_v_net = aux_dict['v_net']
                else:
                    doa_pred = output
                    pred_v_net = None
                
                # 1. Main Task Loss (DOA)
                loss_doa = criterion(doa_pred, target)

                # 3. Aux Loss (IV Alignment)
                if aux_loss and pred_v_net is not None:
                    # [1] GT IV 추출 (4~6번 채널) -> 방향 정보
                    gt_iv_feature = data[:, 4:, :, :] # (B, 3, T, F)
                    
                    # [NEW] 실제 에너지 추출 (0번째 채널: Mel-Spectrogram Intensity)
                    # 보통 0번 채널(W, Omni)이 전체 에너지를 대변합니다.
                    gt_energy_feature = data[:, 0, :, :] # (B, T, F)
                    
                    # [2] Time Pooling (IV와 Energy 둘 다 줄여야 함)
                    B, T_out, _ = pred_v_net.shape
                    T_in = gt_iv_feature.shape[2]
                    
                    if T_in != T_out:
                        pool_size = T_in // T_out
                        
                        # IV Pooling
                        gt_iv_feature = gt_iv_feature.view(B, 3, T_out, pool_size, -1)
                        gt_iv_feature = gt_iv_feature.mean(dim=3) # (B, 3, 50, F)
                        
                        # Energy Pooling (똑같이 평균)
                        gt_energy_feature = gt_energy_feature.view(B, T_out, pool_size, -1)
                        gt_energy_feature = gt_energy_feature.mean(dim=2) # (B, 50, F)
                    
                    # [3] Loss 계산 (에너지 전달!)
                    loss_iv = iv_direction_loss(pred_v_net, gt_iv_feature, gt_energy_feature)
                    total_loss = loss_doa + (lambda_align * loss_iv)
                else:
                    total_loss = loss_doa
                
                if nb_train_batches % 20 == 0:  # 20 step마다 출력
                    pbar.set_postfix({
                        "L_doa": f"{loss_doa.item():.4f}",
                        "L_iv": f"{(loss_iv.item() if (aux_loss and pred_v_net is not None) else 0.0):.4f}",
                        "lam*iv": f"{(lambda_align*loss_iv).item():.4f}" if (aux_loss and pred_v_net is not None) else "0.0000",
                        "L_tot": f"{total_loss.item():.4f}",
                    })

            
            # total_loss.backward()
            # optimizer.step()
            scaler.scale(total_loss).backward()

            # # [체크] 메인 Head vs 보조 Head 체급 비교
            # if hasattr(model, 'module'):
            #     main_grad = model.module.sed_doa_head[0].weight.grad # 모델 구조에 따라 이름 다를 수 있음
            #     aux_grad = model.module.iv_aux_head[0].weight.grad
            # else:
            #     # SeldModel 구조에 따라 sed_doa_head, fnn_list 등 이름 확인 필요
            #     # 보통 마지막 Linear 층을 찍어보면 됩니다.
            #     main_grad = model.fnn_list[-1].weight.grad 
            #     aux_grad = model.iv_aux_head[0].weight.grad

            # if main_grad is not None and aux_grad is not None:
            #     m_val = main_grad.abs().mean().item()
            #     a_val = aux_grad.abs().mean().item()
            #     print(f"Main Grad: {m_val:.6f} vs Aux Grad: {a_val:.6f} (Ratio: {a_val/m_val:.1f}배)")

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            nb_train_batches += 1
            if params['quick_test'] and nb_train_batches == 4:
                break
            pbar.update(1)
            
        train_loss /= nb_train_batches

    return train_loss, optimizer.param_groups[0]["lr"]