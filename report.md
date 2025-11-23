# FAAP-style DETR 공정성 실험 보고서

## 1. 연구 목표와 전반 구조
- **목표**: 동결된 DETR 객체 검출기(배포 환경과 동일)를 교란하지 않으면서, 여성/남성 간 검출 성능(AP/AR) 격차를 줄이는 **입력 전처리용 perturbation 생성기 G**를 학습한다.  
- **주요 아이디어**: 여성 이미지만을 대상으로 한 **adversarial fairness training**. 생성기 G가 DETR 내부 특성에서 성별 판별을 어렵게 만드는 방향으로 perturbation을 생성하고, 동시에 검출 성능을 보존하는 loss를 함께 최적화한다.  
- **세 축**:
  1) **Frozen DETR**: 사전학습 가중치를 고정해 배포 환경을 그대로 재현.  
  2) **Perturbation Generator (G)**: U-Net 계열 경량 네트워크로 입력 해상도 그대로의 bounded noise(±epsilon*tanh)를 생성.  
  3) **Gender Discriminator (D)**: DETR decoder feature(질의별 피처) 평균을 입력받아 성별을 분류.  
- **훈련 시 여성만 perturbation 적용, 남성은 원본으로 훈련**. 평가/이미지 생성 단계에서는 generator checkpoint가 주어지면 남녀 모두 perturbation을 적용해 대조군/처치군을 공정하게 비교한다(Agents.md 정책 반영).

## 2. 데이터 파이프라인
- **루트**: `/home/dohyeong/Desktop/faap_dataset` (기본값, `--dataset_root`로 변경 가능).  
- **구조** (`datasets.py`):
  - `women_split/{train,val,test}/` + `gender_women_{split}.json`
  - `men_split/{train,val,test}/` + `gender_men_{split}.json`
  - COCO 형식만 지원하며 파일명/경로 불일치 시 즉시 오류.  
- **로더 구성** (`build_faap_dataloader`):
  - `train`: 여성/남성 COCO를 concat 후 **WeightedRandomSampler**로 성별 균형(0.5/0.5) 유지. DDP 모드에서는 `DistributedSampler`로 샤딩되고 균형 샘플러는 비활성화.  
  - `val/test`: 순차 샘플링, concat 순서 유지.  
  - `include_gender=True`일 때 배치 반환 형태: `(NestedTensor images, List[targets], List[genders])`.  
- **전처리**: DETR 기본 `make_coco_transforms` 사용. 학습 시 강한 augmentation, 평가 시 validation 프로파일.  
- **데이터 스냅샷**: 학습 시작 시 `faap_outputs/dataset_layout.json`에 split·성별별 이미지 수/확장자/annotation 존재 여부를 기록해 재현성 확보.

## 3. 모델 구성 (`models.py`)
### 3.1 Frozen DETR
- **로드**: `/home/dohyeong/Desktop/detr`(또는 `DETR_REPO` env)에서 표준 DETR 빌드 후 사전학습 체크포인트(`detr-r50-e632da11.pth` 기본) 로드.  
- **동결**: 모든 파라미터 `requires_grad=False`, `eval()` 고정.  
- **출력**:  
  - `forward_with_features(samples) → (outputs, hs[-1])`  
    - `outputs`: `pred_logits`(B,100,92) / `pred_boxes`(B,100,4) + aux_loss 옵션.  
    - `hs[-1]`: transformer decoder 최종 feature (B,100,d_model=256).  
  - `detection_loss(outputs, targets)`: DETR criterion 사용, 가중합(`weight_dict`)으로 총 검출 손실 계산.

### 3.2 PerturbationGenerator G
- **구조**: 경량 U-Net.  
  - Down: ConvBlock(3→32, stride1) → ConvBlock(32→64, stride2) → ConvBlock(64→128, stride2).  
  - Bottleneck: ConvBlock(128→128).  
  - Up: UpBlock(128→64, bilinear resize to skip size) + skip, UpBlock(64→32) + skip.  
  - Output: Conv2d(32→3), `tanh` 후 **epsilon** 스케일.  
- **출력 범위**: `delta = epsilon * tanh(out)`, 입력과 동일 해상도.  
- **후처리**: `clamp_normalized`로 ImageNet 정규화 공간에서 유효 픽셀 범위로 clipping.  
- **스케줄**: `epsilon`을 epoch 기반 선형 워밍업(`_scheduled_epsilon`):  
  - 기본: 시작 0.05 → 목표 0.12, `epsilon_warmup_epochs=5`.  
  - `epoch < warmup` 구간에서 선형 보간, 이후 고정.

### 3.3 GenderDiscriminator D
- **입력**: decoder feature `hs` (B,100,256).  
- **전처리**: 질의 차원 평균 pooling → LayerNorm.  
- **헤드**: MLP(256→256→256→2) with ReLU.  
- **출력**: 2-way logits (`male=0`, `female=1`).

## 4. 학습 루프 (`train_faap.py`)
### 4.1 기본 설정
- 디바이스: CUDA 우선(`--device`), DDP(`--distributed`) 시 `torch.cuda.set_device(args.gpu)`.  
- 시드: `seed + rank`로 보정, GPU 시드 동기화.  
- 옵티마: Adam(lr_g=1e-4 for G, lr_d=1e-4 for D).  
- 배치/워커 기본값: 8/8.  
- 로그 주기: `log_every=10` step마다 MetricLogger 출력.  
- 체크포인트: `faap_outputs/checkpoints/epoch_XXXX.pth`, `save_every=1` epoch.  
- 재시작: `--resume`로 G/D/opt/epoch 상태 복원(`start_epoch = ckpt.epoch + 1`).

### 4.2 배치 분리
- 입력 `(samples, targets, genders)`에서 gender 문자열을 소문자화 후 **여성/남성 인덱스 분리**.  
- `_split_nested`로 NestedTensor를 성별별로 슬라이싱. 여성 배치가 없으면 이후 G 업데이트는 skip.

### 4.3 Discriminator 단계 (k_d회 반복, 기본 2)
1) **여성 흐름**: `x_f`에 G 적용 → DETR feature 추출 (backprop 없음) → logits_f = D(h_f) → CE(logits_f, 1).  
2) **남성 흐름**: 원본 `x_m`으로 DETR feature → logits_m = D(h_m) → CE(logits_m, 0).  
3) 손실 리스트 평균 → 역전파/opt_d.step. 여성·남성 모두 없으면 `d_loss=0`.

### 4.4 Generator 단계 (여성만)
1) `x_f` → G → `x_f_pert` → DETR forward: (outputs_f, feat_f).  
2) **공정성 손실** `fairness_loss = -( CE(logits_f,1) + alpha * entropy(logits_f) )`  
   - CE(logits_f,1): 여성으로 판별되도록 하는 D의 CE를 **음수로 뒤집어** D를 속이는 방향(즉, D loss에 대한 적대적 gradient).  
   - entropy(logits_f): 분포를 평탄하게 만들어 결정적 성별 단서를 약화. `alpha=0.2` 가중.  
3) **검출 손실** `det_loss = DETR criterion(outputs_f, targets_f)`에 `beta=0.7` 곱.  
4) **총 손실** `L_G = fairness_loss + beta * det_loss`. grad clip(`max_norm=0.1`) 후 opt_g.step.  
5) **로깅 측정치** (여성 배치 존재 시):  
   - `delta_linf`: ||delta||_∞ 평균(배치별 max-abs).  
   - `delta_l2`: ||delta||_2 평균.  
   - `obj_score`: DETR class prob(배경 제외) 최대값의 평균.  
   - `obj_frac`: `obj_score > obj_conf_thresh(0.5)` 비율.  
   - 함께 `eps` 기록. 여성 배치가 없을 땐 0으로 채움.

### 4.5 분산 처리
- `utils.init_distributed_mode` 사용, rank/world_size 설정.  
- DDP로 G/D 래핑, optimizer는 unwrap된 모듈 파라미터 사용.  
- Sampler: train 시 rank별 `DistributedSampler`, epoch마다 `set_epoch`.  
- 로그/체크포인트/데이터셋 스냅샷은 **main process만** 기록, epoch 말에 `dist.barrier()`로 정렬.  
- MetricLogger는 `synchronize_between_processes`로 평균 집계.

### 4.6 출력 아티팩트
- `train_log.jsonl`: epoch별 평균 손실·epsilon·delta norm·obj proxy.  
- `checkpoints/epoch_XXXX.pth`: G/D state_dict, optim 상태, args, epoch.  
- `dataset_layout.json`: 각 split의 이미지 수/확장자/annotation 존재 여부.

## 5. 손실 함수 총정리
기호:  
- `x_f, x_m`: 여성/남성 입력 이미지(Normalized).  
- `G(x) = x + delta`, `delta = eps * tanh(g(x))`, `clamp`로 픽셀 범위 제한.  
- `D(h)`: decoder feature h → 성별 logits.  
- `C = CE`, `H = entropy`.  
- `L_det`: DETR criterion 가중합.

1) **Discriminator 손실 (남녀 모두)**  
   - 여성: `L_D^f = C(D(h_f), 1)` with `h_f` from DETR(G(x_f)).  
   - 남성: `L_D^m = C(D(h_m), 0)` with `h_m` from DETR(x_m).  
   - 총합: `L_D = mean(L_D^f + L_D^m)` (존재하는 항만 평균).

2) **Generator 공정성 손실 (여성만)**  
   - `L_fair = - [ C(D(h_f), 1) + alpha * H(D(h_f)) ]`  
   - CE 부호를 뒤집어 D 입장에서 손실을 키워 성별 판별을 어렵게 함.  
   - 엔트로피 항으로 결정성을 추가로 낮춰 보호 속성 노출을 완화.

3) **Generator 검출 보존 손실 (여성만)**  
   - `L_det`: DETR detection loss(weighted sum of cls/box/GIoU 등).  
   - 가중 합: `L_G = L_fair + beta * L_det`.  
   - 목적: 공정성 개선과 검출 성능 유지 사이의 균형. `beta`↓ → 공정성 우선, ↑ → 성능 보존 우선.

## 6. 전체 파이프라인 요약 (학습→평가/생성)
1) **데이터 적재**: 성별 분리 COCO → concat → 균형 샘플링(또는 분산 샤딩).  
2) **전처리**: DETR 표준 변환, ImageNet 정규화.  
3) **Forward (학습)**:  
   - 여성 배치: G로 perturb → DETR → (검출 손실, feature) → D → fairness 손실.  
   - 남성 배치: G 미적용, DETR feature만 추출해 D 업데이트에 사용.  
4) **역전파**: D를 k_d회 업데이트 후, 여성 배치로만 G 업데이트. epsilon은 epoch별 스케줄로 점증.  
5) **로깅/체크포인트**: epoch마다 JSONL·pth 기록, 분산 시 rank0만.  
6) **평가 (`eval_faap.py`, Agents.md 요약)**:  
   - baseline: G 없이 남/녀 각각 AP/AR 측정.  
   - perturbed: generator checkpoint 주어지면 남녀 모두 perturbation 적용 후 AP/AR·세부 COCO 지표·델타 기록.  
   - 결과는 `faap_outputs/faap_metrics.json`에 baseline/perturbed/gaps/deltas/details/details_text 저장.  
7) **시각화 (`gen_images.py`)**:  
   - 원본/perturbed(노이즈 맵)/combined(노이즈 적용) 이미지를 남녀 모두에 대해 생성.  
   - collate_fn 패딩 제거 후 원본 해상도로 복원, 노이즈는 단일 채널 회색 스케일로 저장.

## 7. 연구 관점 연결 고찰
### 7.1 보호 속성 비가시화 전략
- **왜 decoder feature에 D를 붙였는가?** DETR는 객체 질의 수준에서 표현을 학습하므로, decoder feature는 최종 검출 의사결정에 직접 영향. 이 공간에서 성별 판별 능력을 무력화하면 downstream 검출 판단에서 성별 편향이 줄어들 가능성이 높다.  
- **엔트로피 항의 역할**: 단순 CE 역부호만 쓰면 D를 속이는 특정 방향으로 수렴할 수 있다. 엔트로피를 추가해 D 출력 분포를 평탄화하면, 편향적 단서 제거가 더 강제된다(Representation Debiasing).

### 7.2 공정성-성능 트레이드오프
- **`beta`**: 검출 보존 가중. 낮추면 성별 구분이 더 약해지지만 AP/AR 하락 위험. 높이면 성능 유지되나 공정성 개선 폭 감소. 기본값 0.7은 여성 AR 개선을 노리면서 성능 손실을 억제하도록 조정.  
- **`epsilon` 워밍업**: 초기에 작은 노이즈로 안정 학습 → 5 epoch에 걸쳐 0.12까지 확대해 더 강한 교란을 허용. 이는 큰 perturbation이 초기 학습을 망가뜨리는 것을 방지하며, 후반부 공정성 효과를 극대화한다.  
- **로깅된 `obj_score/obj_frac`**: 여성 배치에서 객체 존재 확률의 유지 정도를 proxy로 감시. fairness 향상 과정에서 검출 확신도가 급락하면 beta나 epsilon 조정 필요.

### 7.3 데이터 균형과 분산 학습
- **균형 샘플러**: 단일 프로세스 학습 시 성별 비율을 1:1로 유지해 D와 G가 균형 신호를 받도록 설계.  
- **DDP 모드**: 대규모 실험 확장성을 확보하되, 균형 샘플링이 비활성화되므로 데이터 분포가 치우칠 수 있음. 공정성 실험 시 단일 GPU 또는 별도 rebalancing sampler가 권장.

### 7.4 배포/재현성 고려
- **Frozen DETR**: 배포 환경 동일성 유지 → 실험 결과가 실제 시스템에 바로 이식 가능.  
- **`dataset_layout.json` + checkpoint args**: 실험 로그와 함께 데이터 상태, 하이퍼파라미터를 기록해 재현 가능한 fairness 실험 보고에 활용.  
- **평가 정책(남녀 모두 perturbation)**: 배포 시 perturbation을 전 사용자에게 적용하는 시나리오를 가정. baseline 대비 delta와 gap을 함께 기록해, 실제 배포 결정 시 성능/격차 변화를 동시에 검토할 수 있다.

## 8. 재현 및 확장 가이드
- **실행 예시**:  
  - 단일 GPU: `python train_faap.py --dataset_root /home/.../faap_dataset --output_dir faap_outputs`  
  - 분산: `torchrun --nproc_per_node=4 train_faap.py --distributed --batch_size 4 --dist_url env://`  
- **하이퍼파라미터 튜닝 포인트**:  
  1) `beta` ↓: 공정성 우선, 성능 손실 가능.  
  2) `epsilon_final` ↑: 더 강한 노이즈, 공정성↑/성능↓ 가능 → warmup 길이(`epsilon_warmup_epochs`) 조절.  
  3) `k_d` ↑: D를 더 강하게, G가 더 어려운 적대적 환경을 학습.  
  4) `obj_conf_thresh` 조절로 로그 민감도 변경.  
- **평가/보고**: `faap_outputs/faap_metrics.json`의 `baseline`, `perturbed`, `gaps`, `deltas`, `details`, `details_text`를 교수님 보고용 테이블/그래프로 바로 변환 가능. 특히 `deltas`는 perturbation 적용 전후 변동폭을 남녀별로 보여 공정성 개선 효과를 정량화한다.

## 9. 결론
본 파이프라인은 **입력 전처리 기반 공정성 개선**을 목표로, frozen DETR 위에서 adversarial perturbation을 학습하는 구조다. 여성 데이터에 집중해 성별 판별 단서를 제거(또는 약화)하면서 검출 성능을 보존하도록 설계되었고, epsilon 워밍업과 다중 손실 조합(alpha, beta)으로 공정성-성능 균형을 세밀히 조절한다. 로그/체크포인트/평가 아티팩트가 모두 자동화되어 있어, 교수님께 보고할 때 **(1) 학습 설정, (2) 공정성 손실 구성, (3) 성능 및 격차 변화**를 체계적으로 제시할 수 있다. 필요 시 위 가중치와 스케줄을 조절해 여성 AR을 우선 개선하거나, 배포 시나리오에 맞춰 perturbation 적용 범위를 재설정하면 된다.
