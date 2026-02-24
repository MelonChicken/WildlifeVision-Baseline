related commit : 03144533d070e514973e06bbf0ff41641f6bcf7f
### 날짜 `2026-02-24`
### 실험 ID

* `EXP_LOGREG_HOG_003`  (tag: `logreg_hog_grid_v4_focus_bal_ns6`, best run_id: `5e699b8381`)

### 설정

* Resize:`128 x 128`
* Color: `Grayscale`
* Feature: HOG

  * `pixels_per_cell=(8, 8)`
  * `cells_per_block=(2, 2)`
  * `orientations=9`
  * `block_norm="L2-Hys"`
* Model: LogisticRegression

  * `C=0.003`
  * `use_scaler=False`
  * `class_weight="balanced"`
  * `max_iter=6000`
  * `solver="lbfgs"` `(Limited-memory BFGS)`

### 검증 전략

* `GroupKFold(n_splits=6), group=site`

### 결과

* Fold log loss:

  * `[1.937881, 1.962835, 1.913209, 1.924022, 1.696595, 1.699670]`
* Mean log loss:

  * `1.855702` (std `0.112447`)
* 클래스별 주요 혼동(간단히):

  * 

### 관찰/해석

* 잘된 점:

  * 성능이 아주 약간 개선됨
* 실패 케이스 특징(예: 어두움/흔들림/배경 유사):

  * 

### 다음 액션

* 다음 실험에서 바꿀 1가지:

  * HOG 이전 이미지 전처리 단계의 파라미터를 조정해볼 예정