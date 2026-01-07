import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import utility.lr_sched as lr_sched
import matplotlib.pyplot as plot
import torch.nn.functional as F
plot.switch_backend('agg')

def train_epoch(data_generator, optimizer, model, criterion, params, device, epoch_cnt, total_batches):
    nb_train_batches, train_loss = 0, 0.
    scaler = GradScaler()
    model.train()

    #total_batches = data_generator.get_total_batches_in_data()
    
    # [수정 1] Contrastive Loss 함수 정의 (Cosine Embedding Loss 사용)
    # 두 벡터가 비슷해지도록(Target=1) 학습하는 함수입니다.
    criterion_contrastive = torch.nn.CosineEmbeddingLoss(margin=0.0).to(device)

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

                # [수정 2] 모델 출력 Unpacking 변경
                # 기존: doa_pred, recon_pred, target_foa_origin = output
                # 변경: doa_pred, proj_pred, proj_target = output
                if isinstance(output, tuple):
                    doa_pred, proj_pred, proj_target = output
                else:
                    doa_pred = output
                    proj_pred = None
                    proj_target = None
                
                # 1. Main Task Loss (DOA)
                loss_doa = criterion(doa_pred, target)

            # 2. Auxiliary Task Loss (Contrastive Learning)
            if proj_pred is not None and proj_target is not None:
                # proj_pred, proj_target shape: (Batch, Time, 128)
                
                # Loss 계산을 위해 (Batch * Time, 128) 형태로 폅니다.
                pred_flat = proj_pred.view(-1, proj_pred.shape[-1])
                target_flat = proj_target.view(-1, proj_target.shape[-1])
                
                # 정답 레이블 생성: 모든 쌍이 "유사해야 한다(1)"로 설정
                target_label = torch.ones(pred_flat.shape[0]).to(device)
                
                # Cosine Embedding Loss 계산
                # 1 - cos_sim(pred, target) 와 유사한 효과
                loss_con = criterion_contrastive(pred_flat, target_flat, target_label)

                # # 로그 출력 (비율 확인용, 필요시 주석 처리)
                # if nb_train_batches % 100 == 0:
                #     print(f" [Loss Check] DOA: {loss_doa.item():.4f} | Contrastive: {loss_con.item():.4f}")

                # Loss 합산 (가중치는 0.1 정도 추천)
                # Reconstruction Loss보다 값이 작게 나오므로(0~1 사이), 가중치를 조금 더 줘도 됩니다(0.1 ~ 0.5)
                total_loss = loss_doa + (0.1 * loss_con)
            else:
                total_loss = loss_doa

            # total_loss.backward()
            # optimizer.step()
            scaler.scale(total_loss).backward()
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