# WildlifeVision-Baseline
The baseline with scikit-learn to practice how to make a model for the object classification for wildlife species. 
移대찓?쇳듃???대?吏瑜?scikit-learn 湲곕컲 ?꾪넻 ?뱀쭠?쇰줈 遺꾨쪟?섍퀬, site ?⑥쐞 ?쇰컲???깅뒫??寃利씲룹젣異쒓퉴吏 ?꾩＜?섎뒗 ?숈뒿??CV ?꾨줈?앺듃

> **Traditional CV ??Probabilistic Classifier ??GroupKFold (site) ??Submission**
> 

> 怨듬え?? https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/
> 
> 
> ?꾩옱 ?④퀎: **HOG feature 湲곕컲 Logistic Regression 紐⑤뜽???깅뒫 ?μ긽???꾪븳 誘몄꽭 議곗젙**
> 

---

> ## Executive Summary
> 
- 臾몄젣: 移대찓?쇳듃???대?吏 遺꾨쪟??**珥ъ쁺 ?섍꼍 蹂??*媛 ?ш퀬, 臾댁옉??遺꾪븷濡??됯??섎㈃ ?ㅼ젣 ?깅뒫??怨쇰??됯??섍린 ?쎈떎.
- ?닿껐: ?대?吏?먯꽌 **?쏦istogram of Oriented Gradients, HOG???뱀쭠??異붿텧**?섍퀬 **scikit-learn ?뺣쪧 遺꾨쪟湲?*濡??숈뒿?섎릺, 寃利앹? **GroupKFold**濡?怨좎젙?섏뿬 ?쐓ite ?쇰컲?붴앸? 諛섏쁺?쒕떎.

> Performance Progress
> |**current best score (Log Loss): Feb 24, 2026**|
> |---|
> | `~=1.855702` |

![img.svg](docs/experiments/plot/experiment_progress.svg)

> ## Experiments

| EXP ID | Summary | CV (mean log loss) | Leaderboard (log loss : rank) | Report | Assets |
|---|---|---:|---:|---|---|
| EXP_LOGREG_HOG_002 | LogReg + HOG baseline & grid (GroupKFold by site) | **1.86395** | **1.9354** : `358/557` | [Report](docs/experiments/EXP_LOGREG_HOG_002.md) | [Imgs](docs/assets/imgs/EXP_LOGREG_HOG_002/) |
| EXP_LOGREG_HOG_003 | Focused tuning around best log (`n_splits=6`, `C around 0.003`, `class_weight=balanced`) | **1.85570** | **1.9318** : `359/579` | [Report](docs/experiments/EXP_LOGREG_HOG_003/EXP_LOGREG_HOG_003.md) | - |


> ## Repository Structure
>
* `data/` : ?먮낯 zip ?뺤텞 ?댁젣 ?뚯씪/?대뜑(?쎄린 ?꾩슜, repo???낅줈?쒗븯吏 ?딆쓬)
* `notebooks/` : ?먯깋/?쒓컖???ㅽ뿕 湲곕줉???명듃遺?
* `src/` : ?꾩쿂由??뱀쭠異붿텧/?숈뒿/?됯? 濡쒖쭅 (?⑥닔 ?⑥쐞 紐⑤뱢)
* `artifacts/` :  ?ㅽ뻾 ?곗텧臾?(媛怨??뚯씠釉? ?뱀쭠 罹먯떆, 紐⑤뜽, 由ы룷?? ?쒖텧臾? (濡쒖뺄 ?꾩슜)
  
  * `processed/` :  ?숈뒿/?뚯뒪???뚯씠釉???媛怨??곗씠?????
  * `features/` :  HOG ???뱀쭠 踰≫꽣 罹먯떆 ???
  * `models/` :  ?숈뒿??紐⑤뜽 ?뚯씪 ???
  * `reports/` :  ?ㅽ뿕 寃곌낵 ?붿빟/洹몃옒?????
  * `submissions/` :  ?쒖텧 CSV ?뚯씪 蹂닿?(踰꾩쟾 愿由?


>The Pan African Programme: The Cultured Chimpanzee, Wild Chimpanzee Foundation, DrivenData. (2022). Conser-vision Practice Area: Image Classification. Retrieved 02-11-2026 from https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/.


## Runbook: HOG 튜닝 -> `run_id` 기반 제출

1. HOG 튜닝 실험 실행:

```bash
python -m src.experiments.run_hog_tune --base_dir . --n_splits 5 --seed 42
```

- 로그는 `artifacts/experiments/experiments.jsonl`에 한 줄씩 추가됩니다.
- 각 실험은 `tag`로 구분됩니다(예: `hog_tune_v1__ori_12`).
- 로그 스키마는 아래 필드를 유지합니다:
  `run_id, timestamp, env, tag, feature_name, model_name, cv, params, metrics, data_signature, cv_checks, fold_site_counts`
- 아래는 실제 실험 로그 1줄입니다.
  ```
    {"run_id": "91d73dbc9d", "timestamp": "2026-03-01T05:33:39.689359+00:00", "env": {"python": "3.12.10", "sklearn": "1.8.0"}, "tag": "hog_tune_v1__cpb_3x3", "feature_name": "hog", "model_name": "logreg", "cv": {"type": "GroupKFold(site)", "n_splits": 5}, "params": {"logreg": {"C": 0.003, "max_iter": 6000, "use_scaler": false, "class_weight": "balanced", "solver": "lbfgs", "random_state": 42}, "hog": {"pixels_per_cell": [8, 8], "cells_per_block": [3, 3], "orientations": 9, "block_norm": "L2-Hys", "tiled": false}}, "metrics": {"mean_log_loss": 1.8572156815071872, "std_log_loss": 0.09373077381970896, "fold_log_loss": [1.967952720563294, 1.9480761662756538, 1.8303029869951282, 1.708933614853802, 1.8308129188480584]}, "data_signature": {"n_samples": 16488, "n_sites": 148, "class_counts": {"antelope_duiker": 2474, "bird": 1641, "blank": 2213, "civet_genet": 2423, "hog": 978, "leopard": 2254, "monkey_prosimian": 2492, "rodent": 2013}}, "cv_checks": {"site_single_fold": true, "no_site_overlap": true, "random_state": 42}, "fold_site_counts": {"0": 30, "1": 30, "2": 30, "3": 29, "4": 29}}```

2. `experiments.jsonl`에서 원하는 실험의 `run_id`를 선택합니다.

3. 선택한 `run_id`의 로그 파라미터로 모델을 재현한 뒤, test 추론 및 제출 파일 생성:

```bash
python -m src.submit --run_id 58ba15d713 --base_dir .
```

- `src.submit` 동작 순서:
  1. run_id로 실험 로그 조회
  2. 로그 파라미터로 HOG + LogReg 재현
  3. test 추론 
  4. submission CSV 저장
- 선택한 실험 로그에 `error`가 기록되어 있으면 제출 생성을 차단합니다.

