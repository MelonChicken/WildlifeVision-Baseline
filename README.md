# WildlifeVision-Baseline
The baseline with scikit-learn to practice how to make a model for the object classification for wildlife species. 
카메라트랩 이미지를 scikit-learn 기반 전통 특징으로 분류하고, site 단위 일반화 성능을 검증·제출까지 완주하는 학습형 CV 프로젝트

> **Traditional CV → Probabilistic Classifier → GroupKFold (site) → Submission**
> 

> 공모전: https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/
> 
> 
> 현재 단계: **베이스라인 구축 및 1차 제출 목표**
> 

---

> 🎯 Executive Summary
> 
- 문제: 카메라트랩 이미지 분류는 **촬영 환경 변화**가 크고, 무작위 분할로 평가하면 실제 성능을 과대평가하기 쉽다.
- 해결: 이미지에서 **“Histogram of Oriented Gradients, HOG” 특징을 추출**하고 **scikit-learn 확률 분류기**로 학습하되, 검증은 **GroupKFold**로 고정하여 “site 일반화”를 반영한다.
