"""[실습 04] 모델 경로에서 체크포인트 복원 — 워크로드 간 작업 이어받기

목적: 이전 워크로드가 남긴 학습 결과를 새 워크로드에서 이어받는 방법을 확인한다.
      실무에서 "야간 학습 결과를 다음날 이어서 쓰는" 방식이자,
      스터디에서 정리한 DGX(학습) → Jetson(배포) 의 배포 측에 해당.

전제: 실습 03 을 먼저 실행해 checkpoint.json 이 저장되어 있어야 한다.

실행: python src/04_resume_from_model.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kadap_util as k


def find_checkpoint():
    """체크포인트 파일을 탐색해 (경로, 탐색근거) 반환."""
    env_dir = os.environ.get("KADAP_MODEL_DIR")
    cands = []
    if env_dir:
        cands.append((env_dir, "환경변수 KADAP_MODEL_DIR"))
    for c in ("/workspace/model", "/model", "/mnt/model"):
        cands.append((c, f"마운트된 모델 경로({c})"))
    md = k.find_mydisk()
    if md:
        cands.append((os.path.join(md, "practice", "models"), "마이디스크 practice/models"))
    cands.append((os.path.join("/tmp", "practice", "models"), "임시 경로 /tmp"))

    for d, why in cands:
        p = os.path.join(d, "checkpoint.json")
        if os.path.isfile(p):
            return p, why
    return None, None


def predict_linear(W, x_norm):
    """numpy-lstsq 백엔드용 예측 (가중치 행렬 곱)."""
    import numpy as np
    A = np.r_[x_norm, 1.0]
    return A @ np.array(W)


def main():
    k.hr("04. 모델 경로에서 체크포인트 복원")

    ck_path, why = find_checkpoint()
    if not ck_path:
        print("  checkpoint.json 을 찾지 못했습니다.")
        print("  → 실습 03(03_train_checkpoint.py)을 먼저 실행하세요.")
        print("\n  탐색한 경로:")
        for c in ("KADAP_MODEL_DIR", "/workspace/model", "/model", "/mnt/model",
                  "마이디스크 practice/models", "/tmp/practice/models"):
            print(f"    - {c}")
        sys.exit(1)

    print(f"  체크포인트 : {ck_path}")
    print(f"  탐색 근거  : {why}")

    with open(ck_path, encoding="utf-8") as f:
        ck = json.load(f)

    k.hr("복원된 체크포인트 정보")
    print(f"  백엔드        : {ck['backend']}")
    print(f"  학습 epochs   : {ck['epochs']}")
    print(f"  학습 소요시간  : {ck['train_seconds']} s")
    print(f"  입력 변수     : {', '.join(ck['input_names'])}")
    print(f"  출력 변수     : {', '.join(ck['output_names'])}")
    print(f"  기록된 평균오차: 응력 {ck['mean_err_pct'][0]}% / 처짐 {ck['mean_err_pct'][1]}%")
    print(f"  가중치 항목   : {', '.join(list(ck['weights'].keys())[:6])}"
          + (" ..." if len(ck["weights"]) > 6 else ""))

    try:
        import numpy as np
    except ImportError:
        print("\n  numpy 가 필요합니다.")
        sys.exit(1)

    norm = ck["norm"]
    x_mu, x_sd = np.array(norm["x_mu"]), np.array(norm["x_sd"])
    y_mu, y_sd = np.array(norm["y_mu"]), np.array(norm["y_sd"])

    k.hr("복원한 모델로 즉시 예측")

    # PyTorch 체크포인트가 함께 있으면 그것을 우선 사용
    model = None
    pt_path = os.path.join(os.path.dirname(ck_path), "surrogate.pt")
    if ck["backend"] == "pytorch" and os.path.isfile(pt_path):
        try:
            import torch
            import torch.nn as nn
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = nn.Sequential(nn.Linear(3, 64), nn.SiLU(),
                                  nn.Linear(64, 64), nn.SiLU(),
                                  nn.Linear(64, 2)).to(dev)
            sd = torch.load(pt_path, map_location=dev, weights_only=False)["state_dict"]
            model.load_state_dict(sd)
            model.eval()
            print(f"  PyTorch 체크포인트 복원 성공 ({pt_path})")
            print(f"  디바이스: {dev}")
        except Exception as e:  # noqa: BLE001
            print(f"  PyTorch 복원 실패({type(e).__name__}) → JSON 가중치로 대체")
            model = None

    def predict(H_mm, t_mm, P_N):
        x = np.log(np.array([H_mm / 1000, t_mm / 1000, P_N]))
        xn = (x - x_mu) / x_sd
        if model is not None:
            import torch
            with torch.no_grad():
                out = model(torch.tensor(xn, dtype=torch.float32).unsqueeze(0)
                            .to(next(model.parameters()).device)).cpu().numpy()[0]
        else:
            out = predict_linear(ck["weights"]["W"], xn)
        y = np.exp(out * y_sd + y_mu)
        return y[0] / 1e6, y[1] * 1000   # MPa, mm

    print()
    print(f"  {'H[mm]':>7}{'t[mm]':>7}{'P[N]':>8} | {'응력[MPa]':>12}{'처짐[mm]':>12}")
    k.hr()
    for H_mm, t_mm, P_N in [(80, 10, 2000), (120, 15, 3000), (160, 25, 4500)]:
        s, d = predict(H_mm, t_mm, P_N)
        print(f"  {H_mm:>7.0f}{t_mm:>7.0f}{P_N:>8.0f} | {s:>12.2f}{d:>12.4f}")

    k.hr("학습 포인트")
    print("  · 모델 경로에 남긴 체크포인트로 다른 워크로드에서 작업을 이어받을 수 있다")
    print("  · 가중치만으로는 부족하다 — 정규화 파라미터가 함께 저장돼야 예측이 재현된다")
    print("  · 워크스페이스 > 공유 리포지토리에 모델 경로를 등록하면 팀 멤버와 공유 가능")
    print("  · 이 구조가 '학습 환경과 배포 환경의 분리'의 최소 형태")

    p = k.save_report("04_resume_from_model", {
        "checkpoint": ck_path, "why": why,
        "backend": ck["backend"], "restored_with_torch": model is not None,
    })
    print(f"\n  리포트 저장: {p}")


if __name__ == "__main__":
    main()
