"""KADaP GPUaaS 실습 공통 유틸리티.

워크로드 컨테이너 안에서 '무엇이 어디에 마운트됐는지'를 스스로 찾아내는 함수들.
리포지토리(소스코드/데이터셋/모델 경로) 기능의 동작 원리를 관찰로 익히는 것이 목적.
"""
import os
import sys
import json
import platform
import subprocess
from datetime import datetime

# KADaP 마이디스크 마운트 후보 (UI 표시명과 실제 경로가 다름)
MYDISK_CANDIDATES = [
    "/root/kadap/MyDisk",
    "/root/자동차데이터플랫폼(KADaP)/MyDisk",
    "/mnt/mydisk",
]


def hr(title=""):
    """구분선 출력."""
    if title:
        print("\n" + "=" * 66)
        print(f" {title}")
        print("=" * 66)
    else:
        print("-" * 66)


def find_mydisk():
    """마이디스크 실제 마운트 경로를 반환 (없으면 None)."""
    for p in MYDISK_CANDIDATES:
        if os.path.isdir(p):
            return p
    return None


def repo_root():
    """이 스크립트가 속한 리포지토리의 루트 경로.

    KADaP 리포지토리 > 소스코드 기능으로 마운트된 위치를 알려준다.
    (src/ 의 부모 디렉토리)
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def list_mounts():
    """컨테이너의 마운트 목록을 (device, mountpoint, fstype) 로 반환."""
    out = []
    try:
        with open("/proc/mounts", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    out.append((parts[0], parts[1], parts[2]))
    except OSError:
        pass
    return out


def interesting_mounts():
    """실습과 관련 있는 마운트만 골라낸다 (시스템 기본 마운트 제외)."""
    skip_types = {
        "proc", "sysfs", "devpts", "tmpfs", "cgroup", "cgroup2", "mqueue",
        "devtmpfs", "securityfs", "debugfs", "tracefs", "configfs",
        "fusectl", "pstore", "bpf", "hugetlbfs", "autofs", "ramfs",
    }
    skip_prefix = ("/proc", "/sys", "/dev", "/run")
    res = []
    for dev, mp, fs in list_mounts():
        if fs in skip_types:
            continue
        if mp.startswith(skip_prefix):
            continue
        res.append((dev, mp, fs))
    return res


def probe_dir(path, max_items=12):
    """디렉토리 내용을 안전하게 요약 출력."""
    if not os.path.isdir(path):
        print(f"  [없음] {path}")
        return False
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        print(f"  [권한없음] {path}")
        return False
    print(f"  [존재] {path}  (항목 {len(items)}개)")
    for it in items[:max_items]:
        full = os.path.join(path, it)
        if os.path.isdir(full):
            print(f"      {it}/")
        else:
            try:
                sz = os.path.getsize(full)
                print(f"      {it}  ({sz:,} bytes)")
            except OSError:
                print(f"      {it}")
    if len(items) > max_items:
        print(f"      ... 외 {len(items)-max_items}개")
    return True


def gpu_info():
    """GPU 정보를 dict 로 반환. torch 가 없으면 nvidia-smi 로 대체."""
    info = {"available": False, "devices": [], "source": None}
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["available"] = True
            info["source"] = "torch"
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": p.name,
                    "vram_gb": round(p.total_memory / 1024 ** 3, 1),
                    "sm": f"{p.major}.{p.minor}",
                })
            return info
    except ImportError:
        info["torch_version"] = None

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            info["available"] = True
            info["source"] = "nvidia-smi"
            for i, line in enumerate(r.stdout.strip().splitlines()):
                info["devices"].append({"index": i, "raw": line.strip()})
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def env_snapshot(keywords=("KADAP", "IPYNB", "URL", "GIT", "NOTEBOOK",
                           "DATASET", "MODEL", "MOUNT", "WORKSPACE",
                           "CUDA", "NVIDIA", "HOSTNAME")):
    """관련 환경변수만 추려서 dict 로 반환 (전체 출력은 터미널을 끊을 수 있음)."""
    res = {}
    for k, v in os.environ.items():
        if any(kw in k.upper() for kw in keywords):
            res[k] = v if len(v) <= 200 else v[:200] + "...(생략)"
    return res


def save_report(name, payload, subdir="practice/reports"):
    """리포트를 마이디스크에 JSON 으로 저장. 저장 경로를 반환."""
    base = find_mydisk() or "/tmp"
    outdir = os.path.join(base, subdir)
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(outdir, f"{name}_{stamp}.json")
    meta = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "repo_root": repo_root(),
        "mydisk": find_mydisk(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "data": payload}, f, ensure_ascii=False, indent=2)
    return path
