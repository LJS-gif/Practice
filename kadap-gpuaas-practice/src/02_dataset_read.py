"""[실습 02] 데이터셋 읽기 — 저장소 내 데이터 vs 데이터셋 경로 비교

목적: 학습 데이터를 공급하는 두 가지 방식의 차이를 이해한다.
  (a) 저장소에 데이터를 함께 넣어 소스코드로 마운트 → 소용량·버전관리 유리
  (b) 리포지토리 > 데이터셋 경로로 별도 연결      → 대용량·공유 유리

환경변수 KADAP_DATASET_DIR 을 지정하면 (b) 경로를 우선 사용한다.
(워크로드 생성 시 '환경 변수 추가'로 주입 가능)

실행: python src/02_dataset_read.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kadap_util as k

CSV_NAME = "beam_cases.csv"


def resolve_dataset():
    """데이터셋 CSV 경로를 결정하고, 어떤 방식으로 찾았는지 함께 반환."""
    env_dir = os.environ.get("KADAP_DATASET_DIR")
    if env_dir:
        p = os.path.join(env_dir, CSV_NAME)
        if os.path.isfile(p):
            return p, "(b) 데이터셋 경로 - 환경변수 KADAP_DATASET_DIR"
        print(f"  [경고] KADAP_DATASET_DIR={env_dir} 지정됐으나 {CSV_NAME} 없음 → 저장소 내 데이터로 대체")

    for cand in ("/workspace/data", "/data", "/mnt/data"):
        p = os.path.join(cand, CSV_NAME)
        if os.path.isfile(p):
            return p, f"(b) 데이터셋 경로 - 자동 탐색({cand})"

    p = os.path.join(k.repo_root(), "data", CSV_NAME)
    if os.path.isfile(p):
        return p, "(a) 저장소 내 데이터 - 소스코드로 마운트됨"
    return None, "찾지 못함"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def main():
    k.hr("02. 데이터셋 읽기")

    path, how = resolve_dataset()
    print(f"  탐색 결과 : {how}")
    print(f"  파일 경로 : {path}")
    if not path:
        print("\n  데이터셋을 찾지 못했습니다. 저장소가 정상적으로 마운트됐는지 확인하세요.")
        print("  (실습 01 스크립트로 마운트 경로를 먼저 점검해보세요)")
        sys.exit(1)

    rows, cols = load_csv(path)
    print(f"  레코드 수 : {len(rows)}")
    print(f"  컬럼      : {', '.join(cols)}")

    k.hr("데이터 요약")
    nums = {c: [] for c in cols}
    for row in rows:
        for c in cols:
            try:
                nums[c].append(float(row[c]))
            except (TypeError, ValueError):
                pass

    print(f"  {'컬럼':<20}{'최소':>14}{'평균':>14}{'최대':>14}")
    k.hr()
    for c in cols:
        v = nums[c]
        if not v:
            continue
        print(f"  {c:<20}{min(v):>14.4g}{sum(v)/len(v):>14.4g}{max(v):>14.4g}")

    k.hr("샘플 5건")
    print("  " + "".join(f"{c:>18}" for c in cols))
    for row in rows[:5]:
        print("  " + "".join(f"{row[c]:>18}" for c in cols))

    k.hr("학습 포인트")
    print("  · 저장소에 데이터를 넣는 방식(a)은 15KB 수준의 소용량·예제에 적합")
    print("  · 리포지토리 > 데이터셋 경로(b)는 업로드 한도가 2GB이며 워크스페이스 멤버와 공유 가능")
    print("  · 대용량은 스토리지(NFS/Object Storage/협약기관 S3)를 등록해 데이터셋 경로로 연결")
    print("  · 동일 스크립트가 두 방식을 모두 지원하게 만들면 환경 이동이 쉬워진다")

    p = k.save_report("02_dataset_read", {
        "resolved_path": path, "how": how,
        "n_rows": len(rows), "columns": cols,
    })
    print(f"\n  리포트 저장: {p}")


if __name__ == "__main__":
    main()
