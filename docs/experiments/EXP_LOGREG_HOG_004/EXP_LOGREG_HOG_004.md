related commit : 253cc314c8fd67ebe0116bff5c1ae4f2e60e1461
### 날짜 `2026-03-01`
### 실험 ID

* `EXP_LOGREG_HOG_004`  (tag: `hog_tune_v1__ori_12`, best run_id: `58ba15d713`)

### 설정

* Resize: `128 x 128`
* Color: `Grayscale`
* Feature: HOG

  * `pixels_per_cell=(8, 8)`
  * `cells_per_block=(2, 2)`
  * `orientations=12`
  * `block_norm="L2-Hys"`
  * `tiled=false`
* Model: LogisticRegression

  * `C=0.003`
  * `use_scaler=False`
  * `class_weight="balanced"`
  * `max_iter=6000`
  * `solver="lbfgs"` `(Limited-memory BFGS)`

### 검증 전략

* `GroupKFold(n_splits=5), group=site`

### 결과

* Fold log loss:

  * ` [1.9218773613732805, 1.9035517042217733, 1.7812001179522925, 1.6220636556963934, 1.7858337130252149]`
* Mean log loss:

  * `1.8029053104537909` (std: `0.10747120953462243`)

### 관찰/해석

* 잘된 점:

  * `mean log loss`가 눈에 띄게 `(0.1)`만큼 향상되었다.
  * 이전의 최고 성능 세팅과 비교해볼 때 orientation 횟수를 높인게 긍정적인 방향으로 작용한 듯 보인다.
    * 방향(orientation): 변화가 어느 방향인지 (보통 HOG는 방향을 0~180도로 사용)
  * 이 밖에서 tiling을 진행하거나 iteration 횟수를 증가하거나 다양한 방식으로 진행해보았지만 눈에 띄는 성능 향상은 없었다.

### 다음 액션

* 다음 실험에서 바꿀 1가지:
  * 성능을 유의미하게 바꿀 수 있는 모델 모색