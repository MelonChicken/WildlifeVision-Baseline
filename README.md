# WildlifeVision-Baseline

The baseline with scikit-learn to practice how to make a model for the object classification for wildlife species.

카메라트랩 이미지를 scikit-learn 기반 전통 특징으로 분류하고, site 단위 일반화 성능을 검증·제출까지 완주하는 학습형 CV 프로젝트

> **Traditional CV → Probabilistic Classifier → GroupKFold (site) → Submission**

공모전: https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/

---

## Project Status

- **Local CV best (mean log loss)**: **1.8029** (GroupKFold by **site**, `n_splits=6`)
- **Leaderboard best (log loss : rank)**: **1.9154** : `357/581`
- **Best run_id (submission 기준)**: `58ba15d713`
- **Last updated**: **2026-03-01**

> 현재 단계: **HOG feature 기반 Logistic Regression 모델의 성능 한계점 포착(1.8029) 및 다른 모델 방향성 탐색**

---

## Quickstart (5 min)

### 0) Setup / Install
```bash
# (예시) venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
````

### 1) Data 준비 (경로 규칙)

* 대회 원본 데이터를 `data/` 아래에 압축 해제합니다. (repo에는 업로드하지 않음)
* `data/`는 read-only로 취급하고, 가공 산출물은 `artifacts/`에만 생성합니다.

### 2) HOG 튜닝 실험 실행 (로컬 CV)

```bash
python -m src.experiments.run_hog_tune --base_dir . --n_splits 5 --seed 42
```

* 로그는 `artifacts/experiments/experiments.jsonl`에 JSONL 1줄씩 append 됩니다.
* 각 실험은 `run_id`로 재현/제출이 가능합니다.

### 3) run_id로 제출 파일 생성

```bash
python -m src.submit --run_id 58ba15d713 --base_dir .
```

* 동작 순서:

  1. run_id로 실험 로그 조회
  2. 로그 파라미터로 HOG + 모델 재현
  3. test 추론
  4. submission CSV 저장
* 선택한 실험 로그에 `error`가 기록되어 있으면 제출 생성을 차단합니다.

---

## Executive Summary

* 문제: 카메라트랩 이미지 분류는 **촬영 환경 변화**가 크고, 무작위 분할로 평가하면 실제 성능을 과대평가하기 쉽다.
* 해결: 이미지에서 **HOG(Histogram of Oriented Gradients)** 특징을 추출하고 **scikit-learn 확률 분류기**로 학습하되, 검증은 **GroupKFold(site)** 로 고정하여 “site 일반화”를 반영한다.

---

## Performance Progress

| current best score (Log Loss, Competition Submission 기준): **Mar 1, 2026** |
| ------------------------------------------------------------------------- |
| `~=1.9154`                                                                |

![img.svg](docs/experiments/plot/experiment_progress.svg)

---

## Experiments

| EXP ID             | Summary                                                                                  | CV (mean log loss) | Leaderboard (log loss : rank) | Report                                                              | Assets                                       |
| ------------------ | ---------------------------------------------------------------------------------------- | -----------------: | ----------------------------: | ------------------------------------------------------------------- | -------------------------------------------- |
| EXP_LOGREG_HOG_002 | LogReg + HOG baseline & grid (GroupKFold by site)                                        |         **1.8639** |        **1.9354** : `358/557` | [Report](docs/experiments/EXP_LOGREG_HOG_002.md)                    | [Imgs](docs/assets/imgs/EXP_LOGREG_HOG_002/) |
| EXP_LOGREG_HOG_003 | Focused tuning around best log (`n_splits=6`, `C around 0.003`, `class_weight=balanced`) |         **1.8557** |        **1.9318** : `359/579` | [Report](docs/experiments/EXP_LOGREG_HOG_003/EXP_LOGREG_HOG_003.md) | -                                            |
| EXP_LOGREG_HOG_004 | Changing and Testing HOG parameters (tiling, orientation, etc.)                          |         **1.8029** |        **1.9154** : `357/581` | [Report](docs/experiments/EXP_LOGREG_HOG_004/EXP_LOGREG_HOG_004.md) | -                                            |

---

## Repository Structure

* `data/` : 원본 zip 압축 해제 파일/폴더(읽기 전용, repo에 업로드하지 않음)
* `notebooks/` : 탐색/시각화/실험 기록용 노트북
* `src/` : 전처리/특징추출/학습/평가 로직 (함수 단위 모듈)
* `artifacts/` : 실행 산출물 (가공 테이블, 특징 캐시, 모델, 리포트, 제출물) (로컬 전용)

  * `processed/` : 학습/테스트 테이블 등 가공 데이터 저장
  * `features/` : HOG 등 특징 벡터 캐시 저장
  * `models/` : 학습된 모델 파일 저장
  * `reports/` : 실험 결과 요약/그래프 저장
  * `submissions/` : 제출 CSV 파일 보관(버전 관리)

---

## Runbook: HOG 튜닝 -> `run_id` 기반 제출 (상세)

### 1) HOG 튜닝 실험 실행

```bash
python -m src.experiments.run_hog_tune --base_dir . --n_splits 5 --seed 42
```

### 2) `experiments.jsonl`에서 원하는 실험의 `run_id` 선택

* 로그는 `artifacts/experiments/experiments.jsonl`에 한 줄씩 추가됩니다.
* 각 실험은 `tag`로 구분됩니다(예: `hog_tune_v1__ori_12`).

### 3) 선택한 `run_id`로 모델 재현 + 제출 생성

```bash
python -m src.submit --run_id 58ba15d713 --base_dir .
```

### Logging Schema

* 다음 필드를 유지합니다:

  * `run_id, timestamp, env, tag, feature_name, model_name, cv, params, metrics, data_signature, cv_checks, fold_site_counts`

---

## References

The Pan African Programme: The Cultured Chimpanzee, Wild Chimpanzee Foundation, DrivenData. (2022).
Conser-vision Practice Area: Image Classification. Retrieved 02-11-2026 from
[https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/)

