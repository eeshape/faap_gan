아래 내용은 `Agent.md`로 저장해서 AI agent에게 넘길 최종 스펙입니다.

---

# FAAP-style Fairness-aware Adversarial Perturbation for DETR

## 1. 목표

Deployed 상태의 **DETR 객체 검출 모델(동결, frozen)** 에 대해,
남성/여성 간 **검출 성능(AP, AR)의 격차를 줄이기 위해** FAAP 스타일의 **입력 전처리용 perturbation 생성기 G** 를 학습한다.

* **여성 이미지**에 perturbation을 적용해,
  * 여성 테스트 세트 AP/AR을 끌어올린다.
* **남성 이미지**: 학습 시에는 perturbation 미적용, 평가/이미지 생성 시에는 **generator checkpoint가 주어지면 남녀 모두에 perturbation을 적용**한다.  
  <!-- CHANGED: 원문은 남성에는 항상 perturbation 미적용이었으나, eval/gen_images 코드가 generator 제공 시 남녀 모두에 적용함 -->

---

## 2. 코드/데이터 위치

* **DETR 코드 및 pretrained 가중치(동결)**
  * 경로: `/home/dohyeong/Desktop/detr` (기본 탐색은 패키지 기준 sibling `detr`, 없으면 env `DETR_REPO` 사용)
  * 표준 DETR 구현을 불러오고 모든 파라미터 `requires_grad=False`.

* **데이터셋**
  * 루트: `/home/dohyeong/Desktop/faap_dataset`
  * 구조:
    ```
    /home/dohyeong/Desktop/faap_dataset
      ├── women_split
      │   ├── train
      │   ├── val
      │   └── test
      └── men_split
          ├── train
          ├── val
          └── test
    ```
  * **COCO JSON 포맷만 지원**, 파일명 강제:
    * `women_split/gender_women_{split}.json`
    * `men_split/gender_men_{split}.json`
  * 자동 포맷 탐지/파서는 없음. 해당 JSON이 없으면 즉시 오류 발생.  
    <!-- CHANGED: 원문은 포맷 자동 탐색/어댑터 작성 요구였으나, 코드가 COCO 고정/필수 -->

* **성별 레이블**
  * `women_split/*` → `female`
  * `men_split/*` → `male`

---

## 3. 주요 모듈

* `faap_gan/datasets.py`
  * `GenderCocoDataset`: COCO JSON 로딩, `__getitem__`에서 `(image, target, gender)` 반환 가능.
  * `build_faap_dataloader`: train은 female/male concat 후 가중 샘플러로 성별 균형, val/test는 시퀀셜.
  * `inspect_faap_dataset`: 각 split의 이미지 확장자 카운트/annotation 존재 여부만 기록.

* `faap_gan/models.py`
  * `FrozenDETR`: pretrained 로드 후 동결, `forward_with_features`로 DETR decoder feature(`hs[-1]`) 노출, `detection_loss`로 DETR criterion 사용.
  * `PerturbationGenerator`: U-Net 스타일, `tanh` 출력에 `epsilon` 곱, `clamp_normalized`로 입력 범위 유지.
  * `GenderDiscriminator`: decoder feature 평균 pooling 후 MLP(2-way logits).

* `faap_gan/train_faap.py`
  * 학습 루프: D를 `k_d`번 업데이트, G는 여성 배치에만 업데이트.
  * 체크포인트: `faap_outputs/checkpoints/epoch_XXXX.pth`
  * 로그: `faap_outputs/train_log.jsonl`, `faap_outputs/dataset_layout.json`

* `faap_gan/eval_faap.py`
  * baseline/perturbed AP, AR 계산.
  * **generator checkpoint 주어지면 남녀 모두에 perturbation 적용**.  
    <!-- CHANGED: 원문은 여성만 perturbation 적용, 남성은 미적용 -->
  * 결과 JSON 기본 경로: `faap_outputs/faap_metrics.json`
    * 필드: `baseline`, `perturbed`, `deltas`, `details`, `details_text`, `gaps`, `hyperparams`, `notes`(남성/여성에 perturbation 적용 여부 기록)

* `faap_gan/gen_images.py`
  * generator checkpoint로 남녀 모두에 대해 original/perturbed/combined 이미지 생성 및 저장.  
    <!-- CHANGED: 원문은 남성 perturbation 미생성 -->
  * perturb 시각화는 RGB delta를 평균해 단일 채널로 만든 회색 노이즈 이미지를 저장.
  * train split 사용 시 augmentation 때문에 주석 경고 출력.

---

## 4. 모델 및 학습 세부

### 4.1 Frozen DETR
* 백본/트랜스포머/헤드 모두 동결, eval 모드.
* `forward_with_features` 반환: `(outputs, decoder_features)`
* detection loss: DETR criterion(weight_dict 포함) 합.

### 4.2 Generator G
* 입력: ImageNet 정규화된 `x`.
* 출력: `delta = epsilon * tanh(out_conv(...))`, 입력과 동일 해상도.
* 적용: 학습 시 여성 배치에만 `x + delta` 사용, 이후 `clamp_normalized`.

### 4.3 Discriminator D
* 입력: DETR decoder feature `(B, num_queries, d_model)`.
* 처리: mean pool → LayerNorm → MLP → 2 logits (male=0, female=1).

### 4.4 Loss
* D loss: CE(D(h), z) for male/female (male label 0, female 1).
* G fairness loss (여성만): `-(CE(logits_f, 1) + alpha * entropy(logits_f))`.
* G detection loss (여성만): DETR criterion 사용.
* 총합: `L_G = fairness_loss + beta * det_loss`, grad clip `max_norm`.

---

## 5. 학습/평가 절차

* 학습 데이터: train concat(female+male), gender-balanced WeightedRandomSampler.
* 검증/테스트 로더: gender별 dataset을 concat해 시퀀셜로 순회.
* 체크포인트 주기: `save_every` epoch마다 저장.
* 평가:
  * baseline: generator 없이 남/녀 각각 AP, AR.
  * perturbed: **generator 제공 시 남/녀 모두 perturbation 적용 후 AP, AR**.
  * `gaps`: 남−녀 차이 기록, `deltas`: perturbation 적용 전후 변화.

---

## 6. 출력 아티팩트

* 학습:
  * `faap_outputs/checkpoints/epoch_XXXX.pth` (G/D/opt/epoch/args)
  * `faap_outputs/train_log.jsonl`
  * `faap_outputs/dataset_layout.json` (확장자/파일 존재 정보)
* 평가:
  * `faap_outputs/faap_metrics.json` (위 필드 포함)
* 이미지 생성:
  * `faap_outputs/generated_images/{original,perturbed,combined}/...` (남녀 모두)

---

추가 제약이나 하이퍼파라미터 변경이 필요하면 알려주세요. 코드에 맞춰 문서를 더 조정하겠습니다.


## 7. 코드 수정 내용

**1차 수정**

1번 수정: 스크립트 실행 시 `faap_gan` 모듈을 찾지 못하는 `ModuleNotFoundError`가 발생했던 문제(`ModuleNotFoundError: No module named 'faap_gan'`)를 해결했습니다. `train_faap.py`와 `eval_faap.py` 상단에서 부모 디렉터리를 `sys.path`에 추가하도록 수정해, `python train_faap.py` 형태로 실행해도 패키지 임포트가 동작합니다.

2번 수정: 평가 시 남녀 모두에 대해 원본/perturb 네 가지 결과를 기록하도록 `eval_faap.py`를 확장했습니다. generator 체크포인트를 주면 male/female 각각 perturbed AP/AR을 계산해 JSON에 포함하고, gap 계산도 perturbed 기준으로 변경했습니다.

3번 수정: L40S 환경에 맞춰 학습 기본 하이퍼파라미터를 조정했습니다. `train_faap.py` 기본값을 `batch_size=8`, `num_workers=8`으로 올렸으며, 필요 시 CLI 옵션으로 더 낮출 수 있습니다.

4번 수정: 패키지 위치 이동에 맞춰 상대 임포트와 경로 유틸을 정리했습니다. `.datasets/.models/.path_utils`를 사용하는 형태로 변경하고, `path_utils.py`는 상위 디렉터리에서 `detr` 레포를 자동 탐색하도록 개선했습니다.

**2차 수정**

5번 수정: `inspect_faap_dataset` 반환값의 키가 tuple이라 `json.dump`에 실패하던 문제(`TypeError: keys must be str, int, float, bool or None, not tuple`)를 수정했습니다. 이제 gender_split 문자열 키로 직렬화되며 학습 시 `dataset_layout.json` 저장이 정상 동작합니다.

**3차 수정**

6번 수정: 학습 루프에서 정의되지 않은 `step` 변수를 증가시키다 `UnboundLocalError: cannot access local variable 'step' where it is not associated with a value`가 발생하는 문제를 수정했습니다. 사용되지 않는 `step` 증가 코드를 제거하여 학습이 정상 진행됩니다.

**4차 수정**

7번 수정: Generator 업샘플 시 feature map 크기가 어긋나 `RuntimeError: The size of tensor a (...) must match the size of tensor b (...)`가 발생하던 문제를 수정했습니다. `UpBlock`이 skip-connection 대상의 spatial 크기에 맞춰 interpolate하도록 바꿔, `u2 = up(h3, h2.shape)` / `u1 = up(u2, h1.shape)`로 안전하게 합산합니다.

**5차 수정**

8번 수정: 학습 재시작을 위해 `--resume` 옵션을 추가했습니다. 저장된 체크포인트에서 G/D/옵티마 상태와 마지막 epoch(`epoch`+1부터)로 이어서 학습할 수 있습니다.

**6차 수정**

9번 수정: 퍼터브 결과를 눈으로 확인할 수 있도록 `faap_gan/gen_images.py` 스크립트를 추가했습니다. 제너레이터 체크포인트를 불러와 원본/퍼터브/좌우 결합 이미지를 생성해 저장합니다. 입력은 기존 FAAP 데이터셋 구조를 그대로 사용하고, 출력 경로는 `output_root/{original,perturbed,combined}/men_split|women_split/<split>/file_name` 형태로 annotation과 동일한 파일명/디렉터리를 유지합니다. (train split은 증강 때문에 좌표 불일치 가능성이 있어 val/test 권장)

10번 수정: 위 스크립트에서 annotation JSON을 출력 폴더에 자동 복사하던 변경은 취소했습니다. 현재는 이미지 파일만 생성/저장합니다.

**7차 수정**

11번 수정: `gen_images.py`에서 서로 다른 크기의 이미지를 스택하다 발생하던 오류(RuntimeError: stack expects each tensor to be equal size)를 해결했습니다. 평가/학습과 동일하게 `utils.collate_fn`으로 패딩된 배치를 만들고, 해당 배치(tensors) 위에 제너레이터를 적용하도록 수정했습니다.

**[오류코드]**

1) ModuleNotFoundError: No module named 'faap_gan' — 패키지 경로 추가로 해결(train_faap.py / eval_faap.py 상단에 sys.path 확장).  
2) TypeError: keys must be str, int, float, bool or None, not tuple — inspect_faap_dataset 직렬화 실패를 gender_split 문자열 키로 변경해 해결.  
3) UnboundLocalError: cannot access local variable 'step' — 학습 루프의 미사용 step 증가 코드 제거.  
4) RuntimeError: The size of tensor a ... must match the size of tensor b ... — Generator 업샘플 시 skip 연결 크기 불일치, UpBlock에서 대상 크기로 interpolate하도록 수정.  
5) RuntimeError: stack expects each tensor to be equal size — gen_images.py에서 서로 다른 H,W를 직접 stack, `utils.collate_fn`으로 패딩된 배치 사용으로 해결.  
6) ModuleNotFoundError: No module named 'util' — gen_images.py에서 util.misc를 임포트하기 전에 DETR repo를 sys.path에 추가하도록 변경하여 해결.
7) ValueError (interpolate) : "Input and output must have the same number of spatial dimensions" — gen_images.py에서 torch.interpolate로 원본 해상도로 복원할 때 차원 해석 오류가 발생하여, PIL resize로 복원하도록 변경해 해결.
8) ValueError: pic should be 2/3 dimensional. Got 4 dimensions. — gen_images.py에서 드물게 4D 텐서가 생길 때 `squeeze(0)`로 3D로 맞춘 뒤 PIL 변환하도록 수정.
9) 패딩 영역이 함께 저장되어 퍼터브 이미지가 깨져 보이던 문제 — collate_fn이 추가한 패딩을 원본 H,W로 크롭한 뒤 덴ORMALIZE/저장하도록 gen_images.py를 수정.
10) 시각화 출력 요구사항 변경 — gen_images.py에서 `perturbed`를 노이즈 맵(visualized noise-only)으로, `combined`를 원본+노이즈 적용 이미지 한 장으로 저장하도록 변경(기존 좌우 병합 제거).
11) ValueError: pic should be 2/3 dimensional (combined 4D) — combined 텐서가 4D일 때 squeeze로 3D로 맞춘 뒤 PIL 변환하도록 gen_images.py를 수정.
12) 노이즈/결합 이미지가 어긋나 보이던 문제 — collate_fn 패딩 제거 시 `orig_size` 대신 현재 변환된 `size`로 크롭한 뒤 최종적으로 `orig_size`로 리사이즈하도록 gen_images.py를 수정.
13) 진행 상황 확인을 위해 gen_images.py DataLoader 루프에 tqdm 진행바를 추가.
14) generated_images/combined 평가 시 annotation 부재로 FileNotFoundError 발생 — women_split/men_split의 gender_*_val.json을 원본 데이터셋에서 복사해 넣어야 함(코드 미변경, 평가 절차 기록).
15) COCO annotation에 `info` 필드가 없어 eval 시 KeyError('info')가 발생 — eval_faap.py에서 `get_coco_api_from_dataset` 후 `info`가 없으면 빈 dict로 채우도록 방어 코드 추가.

**8차 수정**

16번 수정: `eval_faap.py` 결과 JSON에 남녀 각각의 baseline 대비 성능 변화량(`perturbed - baseline`)을 `deltas` 항목으로 추가했습니다. `male/female` 별로 AP/AR delta가 포함되어, perturb 적용 시 성능 유지/변동 폭을 바로 확인할 수 있습니다(perturb를 남녀 모두 적용하는 정책 그대로 유지).

**9차 수정**

17번 수정: `eval_faap.py`에서 각 평가 블록 시작 시 어떤 설정인지 명확히 보기 위해 로그 라벨을 추가했습니다. 순서대로 baseline(male), baseline(female), perturbed(male), perturbed(female) 평가가 시작될 때 `=== Evaluating ... ===` 메시지를 출력합니다.

**10차 수정**

18번 수정: `eval_faap.py` JSON에 COCO 세부 지표를 포함하는 `details` 항목을 추가했습니다. baseline/perturbed 각 케이스에 대해 AP/AR 세부 항목(예: `AP@[0.50:0.95]_all_100`, `AP@[0.50]_all_100`, `AR@[0.50:0.95]_small_100` 등 12개)을 남녀별로 기록해, 콘솔 요약(평균 정밀도/재현율 breakdown)과 동일한 정보를 JSON으로도 확인할 수 있습니다.

**11차 수정**

19번 수정: 콘솔에 표시되는 COCO 요약 블록을 그대로 JSON에 문자열로 저장하는 `details_text` 항목을 추가했습니다. baseline/perturbed 각 남녀 케이스마다 터미널 로그와 동일한 AP/AR breakdown 텍스트를 담아, JSON만 봐도 콘솔 출력을 재현할 수 있습니다.


## 3rd 수정 plan

- 목표: 여성 AR 우선 개선 + perturb 강도 상승. epsilon을 0.05→0.12로 5 epoch 동안 워밍업해 학습 초반 안정 후 강한 노이즈 적용.
- 변경 전: epsilon=0.05 고정, `k_d=1`, `alpha=0.1`, `beta=1.0`, perturb 크기/obj_score 로깅 없음.
- 학습 기본값 상향: `k_d=2`, `alpha=0.2`, `beta=0.7`, `epsilon_final=0.12`, `epsilon_warmup_epochs=5`.
- 로깅 추가: train_log에 `eps`, `||delta||_inf`, `||delta||_2`, `obj_score`(bg 제외 최대 class prob 평균), `obj_frac`(obj_score > 0.5 비율) 기록해 perturb 규모와 여성 AR proxy를 추적.
- 코드 포인트: `train_faap.py`에서 epoch별 epsilon 스케줄 적용 후 G epsilon 갱신, 여성 배치로 delta norm/objectness 계산해 MetricLogger에 넣음(여성 없음 시 0 기록).
- 모니터링: eval_faap.py로 실제 AR 확인, obj_score/obj_frac은 학습 중 감시용. 필요 시 beta/epsilon 조절해 여성 AR 하락 시 대응.

## DDP 지원 추가

- `train_faap.py`에 DDP 초기화(`utils.init_distributed_mode`)를 추가하고 Generator/Discriminator를 DDP로 감싸 분산 학습을 지원합니다. optimizer는 unwrap된 모듈 파라미터 기준으로 생성/로드합니다.
- 분산 시 train DataLoader는 `DistributedSampler`로 샤딩하며, gender-balanced WeightedRandomSampler는 비활성화됩니다(정확한 성별 균형이 필요하면 단일 프로세스 모드 사용).
- 로그/체크포인트/데이터셋 스냅샷은 rank0만 기록하고, MetricLogger는 `synchronize_between_processes`로 평균을 동기화합니다. epoch 끝에 barrier로 정렬합니다.
- 실행 예: `torchrun --nproc_per_node=4 train_faap.py --output_dir faap_outputs_ddp --batch_size 4 --dist_url env:// ...` (CUDA 필수, env 기반 주소 사용).

**12차 수정**

20번 수정: `gen_images.py`에서 perturb(노이즈-only) 저장 시 RGB delta를 평균해 단일 채널로 만든 뒤 0~1로 스케일링하여 회색 노이즈로 저장합니다. 시각화 색조가 섞이지 않도록 수정했으며, 실제 delta/combined 값은 그대로입니다.