"""[실습 05] 환경변수·파라미터 주입 확인

목적: 워크로드 생성 시 주입한 설정이 코드에 어떻게 전달되는지 확인한다.
      - '환경 변수 추가': API 키, 학습 조건, 프레임워크 동작 제어 (매뉴얼 127p)
      - '파라미터': 소스코드 등록 시 Batch Job 실행에 적용되는 인자 (매뉴얼 142p)
      - '시간 예측 파라미터': Batch Job 학습시간 예측용 (매뉴얼 128p)

실습 방법: 워크로드 생성 시 아래 환경변수를 넣고 실행해 값이 바뀌는지 확인
    KADAP_EPOCHS=500
    KADAP_LR=0.005
    KADAP_TAG=exp-001

명령행 인자도 함께 지원한다 (Batch Job 파라미터 대응):
    python src/05_params_env.py --epochs 500 --lr 0.005 --tag exp-001
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kadap_util as k

# (환경변수명, 명령행옵션, 기본값, 타입, 설명)
SPEC = [
    ("KADAP_EPOCHS", "--epochs", 300, int, "학습 반복 횟수"),
    ("KADAP_LR", "--lr", 0.003, float, "학습률"),
    ("KADAP_BATCH", "--batch", 64, int, "배치 크기"),
    ("KADAP_TAG", "--tag", "default", str, "실험 태그(결과 구분용)"),
]


def resolve():
    """우선순위: 명령행 인자 > 환경변수 > 기본값"""
    ap = argparse.ArgumentParser(add_help=True)
    for _, opt, default, typ, helptext in SPEC:
        ap.add_argument(opt, type=typ, default=None, help=helptext)
    args, unknown = ap.parse_known_args()

    result = {}
    for envname, opt, default, typ, helptext in SPEC:
        key = opt.lstrip("-")
        cli_val = getattr(args, key.replace("-", "_"), None)
        env_val = os.environ.get(envname)
        if cli_val is not None:
            result[key] = (cli_val, "명령행 인자", opt)
        elif env_val is not None:
            try:
                result[key] = (typ(env_val), "환경변수", envname)
            except ValueError:
                result[key] = (default, f"환경변수 파싱실패({env_val}) → 기본값", envname)
        else:
            result[key] = (default, "기본값", "-")
    return result, unknown


def main():
    k.hr("05. 환경변수·파라미터 주입 확인")

    params, unknown = resolve()

    print(f"  {'파라미터':<10}{'값':>12}   {'출처':<24}{'키'}")
    k.hr()
    for name, (val, src, key) in params.items():
        print(f"  {name:<10}{str(val):>12}   {src:<24}{key}")

    if unknown:
        print(f"\n  [참고] 인식하지 못한 인자: {' '.join(unknown)}")

    k.hr("주입 경로별 용도 (매뉴얼 근거)")
    print("  환경 변수 (127p)")
    print("    · 워크로드 생성 화면에서 키=값 형태로 추가")
    print("    · API 키·인증 토큰, 학습 조건, 프레임워크 동작 제어에 사용")
    print("    · 코드 수정 없이 조건을 바꿔 재실행할 수 있어 실험 관리에 유용")
    print()
    print("  파라미터 (142p)")
    print("    · 리포지토리 > 소스코드 등록 시 'Batch job 실행 시 적용 파라미터'로 지정")
    print("    · 명령행 인자 형태로 스크립트에 전달됨")
    print()
    print("  시간 예측 파라미터 (128p)")
    print("    · Batch Job 전용. 학습 소요시간 예측치를 계산해 표시")
    print("    · 허브(Hub) 모델 사용 시 자동 적용되며 값만 수정 가능")

    k.hr("현재 환경의 관련 환경변수")
    env = k.env_snapshot(keywords=("KADAP",))
    if env:
        for key in sorted(env):
            print(f"  {key} = {env[key]}")
    else:
        print("  KADAP_* 환경변수가 없습니다.")
        print("  → 워크로드 생성 시 '환경 변수 추가'로 KADAP_EPOCHS 등을 넣어 재실행해보세요.")

    k.hr("학습 포인트")
    print("  · 하드코딩 대신 환경변수·인자로 설정을 빼두면 같은 코드로 여러 실험 가능")
    print("  · Interactive Job 에서는 환경변수가, Batch Job 에서는 파라미터가 주로 쓰인다")
    print("  · 두 경로를 모두 지원하게 만들면 IDE 개발 → Batch 학습 전환이 매끄럽다")

    p = k.save_report("05_params_env", {
        "resolved": {kk: {"value": v[0], "source": v[1], "key": v[2]}
                     for kk, v in params.items()},
        "kadap_env": env,
    })
    print(f"\n  리포트 저장: {p}")


if __name__ == "__main__":
    main()
