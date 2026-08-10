# KADaP GPUaaS 실습 리포지토리

한국자동차연구원 자동차데이터플랫폼(KADaP) 인공지능개발솔루션(GPUaaS)의
**리포지토리 기능**을 GitHub 연동 방식으로 익히기 위한 실습 저장소입니다.

- 대상 플랫폼: https://ide.bigdata-car.kr
- 근거 자료: KADaP 사용자 매뉴얼(2026.02), 자동차데이터플랫폼 소개자료(2026.05)

## 이 저장소로 배우는 것

| 실습 | 스크립트 | 익히는 KADaP 기능 |
|---|---|---|
| 00 | `src/00_env_report.py` | 워크로드 자원 확인 (GPU·VRAM·마이디스크) |
| 01 | `src/01_mount_probe.py` | **리포지토리 3종(소스코드·데이터셋·모델) 마운트 원리** ★핵심 |
| 02 | `src/02_dataset_read.py` | 데이터셋 공급 방식 2가지 비교 |
| 03 | `src/03_train_checkpoint.py` | 모델 경로에 체크포인트 저장 |
| 04 | `src/04_resume_from_model.py` | 워크로드 간 작업 이어받기 |
| 05 | `src/05_params_env.py` | 환경변수·파라미터 주입 |

## 실행 순서

```bash
python src/00_env_report.py        # 환경 확인
python src/01_mount_probe.py       # 마운트 경로 관찰 (가장 중요)
python src/02_dataset_read.py      # 데이터 로드
python src/03_train_checkpoint.py  # 학습 + 저장
python src/04_resume_from_model.py # 복원 + 예측
python src/05_params_env.py        # 설정 주입 확인
```

모든 스크립트는 실행 결과를 마이디스크의 `practice/reports/` 에 JSON 리포트로 남깁니다.

## 다루는 문제

2D 평면응력 외팔보의 FEM 해석 결과를 학습하는 **서로게이트 모델**입니다.

- 설계변수 3개: 단면 높이 `H`, 두께 `t`, 하중 `P`
- 응답 2개: 최대 von Mises 응력, 자유단 처짐
- `data/beam_cases.csv` 에 사전 계산된 300 케이스 포함 (15KB)

자동차 CAE 업무에서 FEM 해석 결과를 AI로 대체하는 구조의 최소 형태입니다.

## 환경변수

| 변수 | 기본값 | 용도 |
|---|---|---|
| `KADAP_DATASET_DIR` | (자동 탐색) | 데이터셋 경로 마운트 지점 |
| `KADAP_MODEL_DIR` | (자동 탐색) | 모델 경로 마운트 지점 |
| `KADAP_EPOCHS` | 300 | 학습 반복 횟수 |
| `KADAP_LR` | 0.003 | 학습률 |
| `KADAP_TAG` | default | 실험 태그 |

## 요구사항

`numpy` 만 있으면 전 스크립트가 동작합니다.
`torch` 가 있으면 신경망으로, 없으면 numpy 최소제곱으로 자동 대체 실행됩니다.
KADaP Built-in 이미지(`xiilab/astrago:pytorch-*-cuda*`)에는 둘 다 포함되어 있습니다.
