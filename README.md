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

> Executive Summary
> 
- 문제: 카메라트랩 이미지 분류는 **촬영 환경 변화**가 크고, 무작위 분할로 평가하면 실제 성능을 과대평가하기 쉽다.
- 해결: 이미지에서 **“Histogram of Oriented Gradients, HOG” 특징을 추출**하고 **scikit-learn 확률 분류기**로 학습하되, 검증은 **GroupKFold**로 고정하여 “site 일반화”를 반영한다.

>Repository Structure
>
* `data/` : 원본 zip 압축 해제 파일/폴더(읽기 전용, repo에 업로드하지 않음)
* `notebooks/` : 탐색/시각화/실험 기록용 노트북
* `src/` : 전처리/특징추출/학습/평가 로직 (함수 단위 모듈)
* `artifacts/` :  실행 산출물 (가공 테이블, 특징 캐시, 모델, 리포트, 제출물) (로컬 전용)
  
  * `processed/` :  학습/테스트 테이블 등 가공 데이터 저장
  * `features/` :  HOG 등 특징 벡터 캐시 저장
  * `models/` :  학습된 모델 파일 저장
  * `reports/` :  실험 결과 요약/그래프 저장
  * `submissions/` :  제출 CSV 파일 보관(버전 관리)


>The Pan African Programme: The Cultured Chimpanzee, Wild Chimpanzee Foundation, DrivenData. (2022). Conser-vision Practice Area: Image Classification. Retrieved 02-11-2026 from https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/.
