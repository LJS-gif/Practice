"""[실습 01] 리포지토리 3종 마운트 경로 검증  ★핵심 실습

목적: KADaP '리포지토리' 기능의 동작 원리를 직접 관찰한다.
      워크로드 생성 시 연결한 소스코드 / 데이터셋 경로 / 모델 경로가
      컨테이너 안에서 각각 어디에 붙는지 스스로 찾아낸다.

사전 준비(워크로드 생성 시):
  - 소스코드      → 마운트 경로 예: /workspace/src
  - 데이터셋 경로  → 마운트 경로 예: /workspace/data
  - 모델 경로      → 마운트 경로 예: /workspace/model

실행: python src/01_mount_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kadap_util as k

# 워크로드 생성 시 지정할 만한 마운트 경로 후보
CANDIDATES = [
    "/workspace", "/workspace/src", "/workspace/data", "/workspace/model",
    "/data", "/dataset", "/datasets", "/model", "/models",
    "/mnt/data", "/mnt/model", "/mnt/src", "/home/jovyan", "/root",
]


def main():
    k.hr("01. 리포지토리 마운트 경로 검증")

    print("[A] 이 저장소(소스코드)가 마운트된 위치")
    k.hr()
    root = k.repo_root()
    print(f"  저장소 루트 : {root}")
    print(f"  data 폴더   : {os.path.join(root, 'data')}")
    print(f"  src 폴더    : {os.path.join(root, 'src')}")
    print("  → 이 경로가 '소스코드 추가' 시 입력한 마운트 경로와 같은지 확인")
    print()
    print("  저장소에 포함된 샘플 데이터셋 확인:")
    k.probe_dir(os.path.join(root, "data"))

    k.hr("[B] 컨테이너의 실제 마운트 목록")
    print("  (시스템 기본 마운트는 제외했습니다)")
    print()
    print(f"  {'파일시스템':<12} {'마운트 지점':<34} {'타입'}")
    k.hr()
    for dev, mp, fs in k.interesting_mounts():
        dev_short = dev if len(dev) <= 11 else dev[:8] + "..."
        print(f"  {dev_short:<12} {mp:<34} {fs}")
    print()
    print("  → 외부에서 붙은 볼륨(nfs, overlay 위의 별도 마운트 등)이 있으면")
    print("     그것이 데이터셋/모델 경로로 연결된 스토리지입니다.")

    k.hr("[C] 마운트 경로 후보 전수 조사")
    found = {}
    for c in CANDIDATES:
        if k.probe_dir(c, max_items=8):
            found[c] = True
        print()

    k.hr("[D] 마이디스크")
    md = k.find_mydisk()
    print(f"  마이디스크: {md if md else '찾지 못함'}")
    if md:
        k.probe_dir(md)

    k.hr("정리")
    print(f"  존재하는 후보 경로 {len(found)}개: {', '.join(found) if found else '없음'}")
    print()
    print("  학습 포인트")
    print("   1) 소스코드는 저장소 전체가 지정한 마운트 경로에 복제된다")
    print("   2) 데이터셋/모델 경로는 별도 스토리지가 별도 지점에 마운트된다")
    print("   3) 마이디스크만 워크로드 종료 후에도 내용이 유지된다")

    path = k.save_report("01_mount_probe", {
        "repo_root": root,
        "mounts": k.interesting_mounts(),
        "found_candidates": list(found),
        "mydisk": md,
    })
    print(f"\n  리포트 저장: {path}")


if __name__ == "__main__":
    main()
