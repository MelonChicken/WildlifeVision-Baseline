related commit : None

### 날짜 `2026-03-07`

### 실험 ID

* `EXP_CNN_001` (tag: `cnn`, best run_id: `none`)

### 설정

* Data split

  * `frac=1`, `random_state=42`
  * `train_test_split(test_size=0.25, stratify=y)`
* Input transform

  * Color: `RGB`
  * Resize: `224 x 224`
  * `transforms.ToTensor()`
  * `Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`
* DataLoader

  * `batch_size=32` (train / eval / test)
* Model

  * Backbone: `torchvision.models.resnet50(pretrained=True)`
  * Head:

    * `Linear(2048, 100)`
    * `ReLU(inplace=True)`
    * `Dropout(0.1)`
    * `Linear(100, 8)`
* Training

  * Loss: `CrossEntropyLoss`
  * Optimizer: `SGD(lr=0.001, momentum=0.9)`
  * Epochs: `2` `로컬 노트북 성능 제한으로 인해 Epochs를 더 늘리기 부담스러웠음`
* Inference / submission

  * `softmax(dim=1)`

### 검증 전략

* eval inference ran with `batch_size=32`
* 노트: 평가 정확도는 대략 `50%` (benchmark-level)
* 
### 관찰/해석

| submission | score |
|------------|-------|
| `submission_df.csv` | `2.1253` |

* `data/benchmark_refactoring.ipynb`에서 실습 진행 (업로드 X).
* 오리지널 benchmark note (`~1.8`)와 비교했을 때, 오히려 성능이 하락함. traditional ML (기존 접근)과 비교해도 상당히 차이나는 수치
