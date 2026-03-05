related commit : 8e401307e0b2d59780ab86b2ad9b04ab71c90bb6

### 날짜 `2026-03-05`

### 실험 ID

* `EXP_MODELSEARCH_HOG_001`  (tag: `model_family_sweep_v1__family_(a, b, c)__ori_16`, best run_id: `163fee3712`)

### 설정

* Resize: `128 x 128`
* Color: `Grayscale`
* Feature: HOG

  * `pixels_per_cell=(8, 8)`
  * `cells_per_block=(2, 2)`
  * `orientations=16`
  * `block_norm="L2-Hys"`
  * `tiled=false`
* Model: HistGradientBoostingClassifier

  * `learning_rate=0.05`
  * `max_depth=2`
  * `max_iter=150`
  * `random_state=42`
  * `early_stopping=False`

### 검증 전략

* `GroupKFold(n_splits=5), group=site`

* Fold log loss:

  * `[1.910537332789214, 1.8353669638541161, 1.764281861809374, 1.7581814136357443, 1.830474622885167]`
* Mean log loss:

  * `1.819768438994723` (std: `0.05562130953559767`)

	| submission                                     | 결과      |
    |------------------------------------------------|---------|
    | family b (`Margin + Probability Calibration`)	 | 2.5653  |
    | family c (`Nonlinear: HistGradientBoosting`)	| 2.0429 |

### 관찰/해석

* 잘된 점:

  * `mean loss`는 향상되지 않고 오히려 조금 성능이 하락하였다. 
  * 하지만 `standard deviation`이 이전에 비에 줄어들으며 모델 성능의 일관성이 향상되었다고 생각된다.

* 해석:
  * 모델을 바꾸고 특히 `family_c`의 미세조정을 진행하면서 느낀 점은 HOG feature 기반의 예측 방식에서 성능을 더 향상시키는 것이 조금 어려울 것 같다는 생각이다.

### 다음 액션

* 다음 실험에서 바꿀 1가지:

  * HOG feature 기반 예측을 잠시 멈추고 benchmark로 제시되어있는 CNN 모델의 흐름을 개선하는 것으로 접근
  * CNN을 연습해보고 이번 실험 프로젝트를 마무리해볼 예정