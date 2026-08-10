"""[실습 03] 학습 후 모델 경로에 체크포인트 저장

목적: 리포지토리 > 모델 경로 기능의 용도를 이해한다.
      학습 산출물(가중치·설정·체크포인트)을 어디에 남겨야 다음 워크로드가 이어받을 수 있는지 확인.

저장 위치 우선순위:
  1) 환경변수 KADAP_MODEL_DIR (워크로드 생성 시 '환경 변수 추가'로 주입)
  2) /workspace/model, /model, /mnt/model 중 존재하는 경로
  3) 마이디스크의 practice/models  (권장 - 워크로드 종료 후에도 유지)

파라미터도 환경변수로 조정 가능: KADAP_EPOCHS (기본 300)

실행: python src/03_train_checkpoint.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kadap_util as k


CSV_NAME = "beam_cases.csv"


def find_csv():
    env_dir = os.environ.get("KADAP_DATASET_DIR")
    cands = ([env_dir] if env_dir else []) + ["/workspace/data", "/data", "/mnt/data",
                                             os.path.join(k.repo_root(), "data")]
    for c in cands:
        if not c:
            continue
        p = os.path.join(c, CSV_NAME)
        if os.path.isfile(p):
            return p
    return None


def resolve_model_dir():
    """체크포인트 저장 경로와 선택 근거를 반환."""
    env_dir = os.environ.get("KADAP_MODEL_DIR")
    if env_dir:
        os.makedirs(env_dir, exist_ok=True)
        return env_dir, "환경변수 KADAP_MODEL_DIR"

    for c in ("/workspace/model", "/model", "/mnt/model"):
        if os.path.isdir(c) and os.access(c, os.W_OK):
            return c, f"마운트된 모델 경로 자동 탐색({c})"

    md = k.find_mydisk()
    base = md if md else "/tmp"
    p = os.path.join(base, "practice", "models")
    os.makedirs(p, exist_ok=True)
    reason = "마이디스크(종료 후에도 유지)" if md else "임시 경로 /tmp (종료 시 소실 주의)"
    return p, reason


def main():
    k.hr("03. 학습 후 모델 경로에 체크포인트 저장")

    csv_path = find_csv()
    if not csv_path:
        print("  데이터셋을 찾지 못했습니다. 실습 02를 먼저 확인하세요.")
        sys.exit(1)
    print(f"  데이터셋 : {csv_path}")

    model_dir, why = resolve_model_dir()
    print(f"  저장 경로 : {model_dir}")
    print(f"  선택 근거 : {why}")

    epochs = int(os.environ.get("KADAP_EPOCHS", "300"))
    print(f"  epochs   : {epochs}  (환경변수 KADAP_EPOCHS 로 조정)")

    try:
        import numpy as np
    except ImportError:
        print("\n  numpy 가 필요합니다: pip install numpy")
        sys.exit(1)

    # 데이터 로드
    raw = np.genfromtxt(csv_path, delimiter=",", names=True)
    X = np.stack([raw["H_m"], raw["t_m"], raw["P_N"]], axis=1)
    Y = np.stack([raw["vonMises_Pa"], raw["tip_deflection_m"]], axis=1)
    print(f"  데이터    : X{X.shape}  Y{Y.shape}")

    # 로그 변환 + 표준화 (응답이 수 자릿수에 걸쳐 분포)
    Xl, Yl = np.log(X), np.log(Y)
    x_mu, x_sd = Xl.mean(0), Xl.std(0)
    y_mu, y_sd = Yl.mean(0), Yl.std(0)
    Xn, Yn = (Xl - x_mu) / x_sd, (Yl - y_mu) / y_sd

    n_tr = int(len(X) * 0.8)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, va = idx[:n_tr], idx[n_tr:]
    print(f"  분할      : 학습 {len(tr)} / 검증 {len(va)}")

    k.hr("학습")
    try:
        import torch
        import torch.nn as nn
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  프레임워크: PyTorch {torch.__version__} | 디바이스 {dev}")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        Xtr, Ytr, Xva, Yva = t(Xn[tr]), t(Yn[tr]), t(Xn[va]), t(Yn[va])

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(3, 64), nn.SiLU(),
                              nn.Linear(64, 64), nn.SiLU(),
                              nn.Linear(64, 2)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        lossf = nn.MSELoss()

        t0 = time.perf_counter()
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            loss = lossf(model(Xtr), Ytr)
            loss.backward(); opt.step()
            if (ep + 1) % max(1, epochs // 5) == 0:
                model.eval()
                with torch.no_grad():
                    vl = lossf(model(Xva), Yva).item()
                print(f"    epoch {ep+1:4d} | train {loss.item():.4e} | valid {vl:.4e}")
        train_s = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            pred_n = model(Xva).cpu().numpy()
        state = {kk: v.cpu().numpy().tolist() for kk, v in model.state_dict().items()}
        backend = "pytorch"

        # PyTorch 네이티브 체크포인트도 함께 저장
        torch.save({"state_dict": model.state_dict(),
                    "x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd},
                   os.path.join(model_dir, "surrogate.pt"))

    except ImportError:
        print("  PyTorch 미설치 → numpy 최소제곱(선형회귀)으로 대체 실행")
        A = np.c_[Xn[tr], np.ones(len(tr))]
        W, *_ = np.linalg.lstsq(A, Yn[tr], rcond=None)
        pred_n = np.c_[Xn[va], np.ones(len(va))] @ W
        state = {"W": W.tolist()}
        backend = "numpy-lstsq"
        train_s = 0.0

    # 역변환 후 오차 평가
    pred = np.exp(pred_n * y_sd + y_mu)
    true = Y[va]
    err = np.abs(pred - true) / true * 100

    k.hr("검증 결과")
    for i, nm in enumerate(["최대 von Mises 응력", "자유단 처짐"]):
        print(f"  {nm:<20} 평균오차 {err[:,i].mean():6.2f}%   최대오차 {err[:,i].max():6.2f}%")

    # 체크포인트(JSON) 저장 - 프레임워크 없이도 읽을 수 있는 형태
    ckpt = {
        "backend": backend,
        "epochs": epochs,
        "train_seconds": round(train_s, 3),
        "norm": {"x_mu": x_mu.tolist(), "x_sd": x_sd.tolist(),
                 "y_mu": y_mu.tolist(), "y_sd": y_sd.tolist()},
        "input_names": ["H_m", "t_m", "P_N"],
        "output_names": ["vonMises_Pa", "tip_deflection_m"],
        "mean_err_pct": [round(float(err[:, 0].mean()), 4), round(float(err[:, 1].mean()), 4)],
        "weights": state,
    }
    ck_path = os.path.join(model_dir, "checkpoint.json")
    with open(ck_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False)

    k.hr("저장 결과")
    for fn in sorted(os.listdir(model_dir)):
        fp = os.path.join(model_dir, fn)
        if os.path.isfile(fp):
            print(f"  {fn:<22} {os.path.getsize(fp):>12,} bytes")

    k.hr("학습 포인트")
    print("  · 모델 경로는 '가중치·설정·체크포인트'를 두는 곳 (매뉴얼 151p)")
    print("  · 정규화 파라미터(x_mu/x_sd 등)를 가중치와 함께 저장해야 재사용 가능")
    print("  · 워크로드 종료 시 마이디스크 외 경로의 파일은 삭제됨")
    print("  · 다음 워크로드에서 이어받으려면 실습 04를 실행")

    p = k.save_report("03_train_checkpoint", {
        "csv": csv_path, "model_dir": model_dir, "why": why,
        "backend": backend, "epochs": epochs,
        "mean_err_pct": ckpt["mean_err_pct"],
    })
    print(f"\n  리포트 저장: {p}")


if __name__ == "__main__":
    main()
