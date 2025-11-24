# FAAP WGAN 상세 가이드

## 목표
- 사전 학습된 DETR를 고정(frozen)한 상태에서 여성 이미지만 미세한 노이즈를 추가해 성별 판별 신호를 약화시키면서, 객체 검출 성능과 신뢰도 분포를 남성 샘플과 가깝게 만드는 공정성 공격을 학습한다.
- 코드 기준: `train_faap_wgan.py`, `models.py`.

## 구성 요소
- 데이터: `build_faap_dataloader(..., include_gender=True, balance_genders=True)`로 남녀가 동일한 비율로 배치된다. 각 샘플은 이미지 텐서, DETR 라벨(`boxes`, `labels`), 성별 문자열을 가진다.
- Frozen DETR: 모든 파라미터를 동결하고, 마지막 디코더 출력(`hs[-1]`)을 특징으로 사용하며 검출 손실 계산을 제공한다.
- PerturbationGenerator: 얕은 U-Net, 출력 `delta = epsilon * tanh(...)`, `clamp_normalized(x+delta)`로 정규화된 픽셀 범위를 유지하며 `||delta||_inf <= epsilon` 제약을 강제한다.
- GenderDiscriminator: DETR 특징을 평균 풀링 후 2-way MLP로 성별을 분류한다.
- 옵티마: Adam(G/D). G는 `max_norm`으로 gradient clipping. D는 배치마다 `k_d`회 더 자주 업데이트된다.

## 학습 파이프라인 (1 배치 흐름)
```
[균형 데이터로더]
        │ samples, targets, genders
        ├─ female indices ──> PerturbationGenerator ─┐
        │                                           │
        │                        clamp_normalized(x + δ; ||δ||∞≤ε)
        │                                           │
        │                               Frozen DETR (frozen)
        │                                           ├─ pred_logits/boxes ──> detection loss L_det
        │                                           ├─ features ──> GenderDiscriminator ──> CE_f + entropy
        │                                           └─ pred_logits ──> matched scores_f
        └─ male indices ─────> Frozen DETR (frozen) ─┤
                                                    ├─ features ──> GenderDiscriminator ──> CE_m
                                                    └─ pred_logits ──> matched scores_m
matched scores_f & matched scores_m ──> Wasserstein L_w
```
주요 단계:
1) 배치에서 성별로 인덱스를 나누고, 로그 초깃값(0 텐서)을 준비.
2) D 업데이트 `k_d`회: 여성은 `generator`를 통과시킨 후 DETR 특징을 얕게 추출(gradient 차단), 남성은 원본 특징. CE를 남녀 각각 계산 후 평균해 역전파.
3) G 업데이트(여성 샘플이 있을 때만):
   - `δ = G(x_f)`, `x_f' = clamp_normalized(x_f + δ)`.
   - Frozen DETR 추론으로 `outputs_f`, 특징 `feat_f` 획득.
   - Discriminator 로짓으로 공정성 손실 계산, DETR 검출 손실과 Wasserstein 정렬 손실을 더해 역전파 후 gradient clipping, `opt_g.step()`.
4) 로깅: `||δ||_inf`, `||δ||_2`, 객체 신뢰도 평균/임계 이상 비율, 각 손실을 기록.
5) 에폭 종료 시 JSONL 로그에 집계값을 남기고, `save_every`마다 체크포인트를 저장.

## 손실 정의
### 1) Discriminator (성별 분류기)
- 입력: 남녀 모두 Frozen DETR 특징(`feat`)을 평균 풀링한 텐서.
- 라벨: 여성=1, 남성=0.
- 손실: `L_D = mean( CE(logits_f, 1), CE(logits_m, 0) )` (여성/남성 샘플이 있는 것만 평균).
- 업데이트: 배치마다 `k_d`회 반복, G/DETR는 `torch.no_grad()`로 고정하여 D만 학습.

### 2) Generator - 성별 흔들기(fairness)
- 여성 로짓 `logits_f`에 대해
  - `CE_f = CE(logits_f, 1)` (D가 여성이라고 확신하도록 만드는 손실을 역부호 처리)
  - `H = -E[ softmax * log softmax ]` (엔트로피; 불확실성을 키움)
- 공정성 손실: `L_fair = -( CE_f + alpha * H )`. 값이 작아질수록 D는 여성 여부를 확신하지 못하게 된다.

### 3) Generator - 검출 보존
- Frozen DETR의 기준 손실(`Hungarian` 매칭 포함) 사용: `L_det = DETR.detection_loss(outputs_f, female_targets)`.
- 총합에서는 `beta * L_det`로 가중해, 변형이 검출 품질을 크게 해치지 않도록 한다.

### 4) Generator - Wasserstein 정렬 (검출 신뢰도 분포 맞추기)
1. 여성 예측에서 매칭된 클래스의 softmax 점수만 추출: `_matched_detection_scores`.
2. 남성도 동일하게 추출(없으면 빈 텐서).
3. 두 벡터를 정렬 후, 길이가 다르면 `_resize_sorted`로 선형 보간해 길이를 맞춘다.
4. `L_w = mean_i | sort(f)_i - sort(m)_i |` (1D Wasserstein-1 근사).
- 총합에서는 `lambda_w * L_w`로 추가되어 여성 검출 신뢰도가 남성과 가깝도록 유도한다.

### 5) Generator 총합 및 업데이트 순서
- `L_G = L_fair + beta * L_det + lambda_w * L_w`.
- 여성 배치가 없으면 G는 건너뛴다. G step 이후 `opt_d`는 건드리지 않는다.

### 6) Epsilon 스케줄과 제약
- 선형 워밍업: `epsilon(epoch) = eps_start + (eps_final - eps_start) * clip( epoch / (warmup_epochs-1), 0, 1 )`.
- 매 에폭 시작 시 G의 `epsilon` 속성에 주입되어 허용 노이즈 크기를 점진적으로 확대한다.
- 생성된 `delta`는 항상 `clamp_normalized` 후 사용, 로그로 `||delta||_inf`, `||delta||_2`가 남는다.

## 한 배치 학습 예시 (수치)
가정: `alpha=0.2`, `beta=0.7`, `lambda_w=0.05`, `k_d=2`, `max_norm=0.1`, `obj_conf_thresh=0.5`. 에폭 2(0부터 시작), `eps_start=0.05`, `eps_final=0.12`, `warmup_epochs=5`라면
- `progress = 2 / (5-1) = 0.5`, `epsilon = 0.05 + 0.07*0.5 = 0.085`.
- 배치: female 2장, male 2장.

### Discriminator 단계 (`k_d=2`회)
- 1회차: `CE_female=0.95`, `CE_male=0.40` → `L_D1 = (0.95+0.40)/2 = 0.675`. `opt_d.step()`.
- 2회차: `CE_female=0.90`, `CE_male=0.45` → `L_D2 = 0.675`. 다시 `opt_d.step()`. (로깅되는 값은 마지막 스텝 결과.)

### Generator 단계 (여성만 사용)
1. `δ = G(x_f)` 생성 후 `||δ||_inf=0.070`, `||δ||_2=1.85`로 측정되고 `epsilon=0.085`보다 작아 제약을 만족.
2. Frozen DETR 추론 → `outputs_f`, 특징 `feat_f`, D 로짓 `logits_f` 획득.
   - 예시 로짓 두 개: `[-0.1, 0.1]`, `[-0.2, 0.4]` (남성/여성 순). softmax 확률은 각각 `[0.45,0.55]`, `[0.354,0.646]`.
   - `CE_f = mean( -log 0.55, -log 0.646 ) = (0.598 + 0.437)/2 ≈ 0.518`.
   - 엔트로피 `H = mean(0.687, 0.641) ≈ 0.664`.
   - `L_fair = -(0.518 + 0.2*0.664) = -(0.518 + 0.133) ≈ -0.651`.
3. 검출 보존: DETR 손실이 `L_det=1.05`라고 하면 `beta * L_det = 0.7 * 1.05 = 0.735`.
4. Wasserstein 정렬: 매칭된 softmax 점수
   - 여성: `[0.62, 0.58, 0.55]` → 정렬 `[0.55, 0.58, 0.62]`
   - 남성: `[0.70, 0.65, 0.60]` → 정렬 `[0.60, 0.65, 0.70]`
   - 차이: `[0.05, 0.07, 0.08]`, 평균 `L_w = 0.0667`, `lambda_w * L_w ≈ 0.0033`.
5. 총손실: `L_G = -0.651 + 0.735 + 0.0033 ≈ 0.0873`. 이를 역전파 → gradient clipping → `opt_g.step()`.
6. 로깅: `obj_score`는 여성 예측의 클래스 확률 최대값 평균(예: 0.64), `obj_frac`는 그중 0.5 초과 비율(예: 0.67)로 기록된다.

## 로그와 체크포인트
- `output_dir/train_log.jsonl`에 에폭별 평균 손실, `epsilon`, 노이즈 노름, 객체 신뢰도 지표를 기록.
- `output_dir/checkpoints/epoch_XXXX.pth`에 G/D/옵티마 상태와 에폭 번호를 저장해 재시작 가능.
