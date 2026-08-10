"""[실습 00] 워크로드 환경 전수 조사

목적: 내가 만든 워크로드가 실제로 어떤 자원을 받았는지 확인한다.
확인 항목: Python/OS, GPU·VRAM, 마이디스크 마운트, 관련 환경변수

실행: python src/00_env_report.py
"""
import os
import sys
import platform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kadap_util as k


def main():
    k.hr("00. 워크로드 환경 전수 조사")

    print(f"호스트명   : {platform.node()}")
    print(f"OS         : {platform.system()} {platform.release()}")
    print(f"Python     : {platform.python_version()}")
    print(f"실행 위치  : {os.getcwd()}")

    k.hr("GPU 자원")
    g = k.gpu_info()
    print(f"torch 버전 : {g.get('torch_version')}")
    print(f"GPU 사용   : {g['available']}  (확인 경로: {g['source']})")
    if g["devices"]:
        for d in g["devices"]:
            if "name" in d:
                print(f"  GPU {d['index']}: {d['name']} | VRAM {d['vram_gb']} GB | SM {d['sm']}")
            else:
                print(f"  GPU {d['index']}: {d['raw']}")
    else:
        print("  → GPU 미할당. 워크로드 생성 시 GPU 노드 선택을 확인하세요.")

    k.hr("마이디스크 마운트")
    md = k.find_mydisk()
    if md:
        print(f"실제 경로  : {md}")
        print("※ 매뉴얼의 'root > 자동차데이터플랫폼(KADaP) > MyDisk' 는 UI 표시명입니다.")
        k.probe_dir(md)
    else:
        print("마이디스크를 찾지 못했습니다. 아래 후보를 직접 확인해보세요:")
        for c in k.MYDISK_CANDIDATES:
            print(f"  - {c}")

    k.hr("리포지토리(소스코드) 마운트 위치")
    print(f"이 스크립트가 속한 저장소 루트: {k.repo_root()}")
    print("→ 워크로드 생성 시 '소스코드 추가'에서 지정한 마운트 경로와 일치하는지 확인하세요.")

    k.hr("관련 환경변수")
    env = k.env_snapshot()
    if env:
        for key in sorted(env):
            print(f"  {key} = {env[key]}")
    else:
        print("  (관련 환경변수 없음)")
    print("\n※ .ipynb URL 방식이 동작하는 구조라면 URL 관련 환경변수가 보일 수 있습니다.")

    path = k.save_report("00_env_report", {
        "host": platform.node(),
        "python": platform.python_version(),
        "gpu": g,
        "mydisk": md,
        "repo_root": k.repo_root(),
        "env": env,
    })
    k.hr()
    print(f"리포트 저장: {path}")


if __name__ == "__main__":
    main()
