### 날짜 `2026-02-14`
### 실험 ID

* `EXP_LOGREG_HOG_002`  (tag: `logreg_hog_grid_v2`, best run_id: `7d251ed19c`)

### 설정

* Resize: `128 x 128`
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
  * `max_iter=2000`
  * `solver="lbfgs"` `(Limited-memory BFGS)`

### 검증 전략

* `GroupKFold(n_splits=5), group=site`

### 결과

* Fold log loss:

  * `[1.9685455, 1.9743465, 1.8355728, 1.7264473, 1.8148375]`
* Mean log loss:

  * `1.8639499` (std `0.0951335`)
* 클래스별 주요 혼동(간단히):

  * (현재 로그에 혼동 정보 없음 → confusion matrix/분석 로그 추가 필요)

### 관찰/해석

* 잘된 점:

  * `use_scaler=False`가 일관되게 유리했고, `C`를 **0.002~0.005** 구간으로 낮추면 log loss가 최소화됨.
  * `class_weight="balanced"`가 평균 성능을 소폭 개선함(최저 mean 1.86대).
    ![img.png](res/imgs/EXP_LOGREG_HOG_002/img.png)
    ![img.png](res/imgs/EXP_LOGREG_HOG_002/img_1.png)
* 실패 케이스 특징(예: 어두움/흔들림/배경 유사):

  * (현재 로그만으로 이미지 실패 유형 판단 불가 → fold별 오분류 샘플 저장/시각화 필요)

### 다음 액션

* 다음 실험에서 바꿀 1가지:

  * **Feature만 변경**: HOG “타일링(예: 2×2 분할 후 HOG concat)” 또는 HOG 파라미터 1개(orients/pixels_per_cell/cells_per_block)만 바꿔 성능 상한(1.8 이하) 돌파 시도