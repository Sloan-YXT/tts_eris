"""
监控包装器：启动 TTS 服务并记录退出原因。

用法：
  python run_watch.py              # 监控 run_with_class.py
  python run_watch.py gsv          # 监控 run_with_class.py gsv
"""
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "tts_crash.log"
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def get_gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip()
        return f"GPU: {out} MB"
    except Exception:
        return "GPU: N/A"


def main() -> None:
    args = sys.argv[1:]
    cmd = [PYTHON, "-u", str(ROOT / "run_with_class.py")] + args
    out_log = ROOT / "tts_stdout.log"

    log(f"启动服务: {' '.join(cmd)}")
    log(f"stdout/stderr → {out_log}")
    log(get_gpu_info())

    # 输出写文件，watchdog 独立于子进程
    with open(out_log, "w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd, stdout=fout, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    log(f"子进程 PID={proc.pid}")

    # 持续监控子进程 + GPU 内存
    last_gpu_check = 0
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            now = time.time()
            if now - last_gpu_check > 30:
                log(f"[alive] PID={proc.pid} {get_gpu_info()}")
                last_gpu_check = now
            time.sleep(2)
    except KeyboardInterrupt:
        log("收到 Ctrl+C，终止子进程")
        proc.terminate()
        proc.wait(timeout=10)

    code = proc.returncode
    log(get_gpu_info())

    # 读取最后几行输出
    try:
        lines = out_log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-20:] if len(lines) > 20 else lines
        log(f"最后输出:\n" + "\n".join(f"  | {l}" for l in tail))
    except Exception:
        pass

    if code == 0:
        log(f"服务正常退出 (code=0)")
    elif code is not None and code > 0:
        log(f"服务异常退出 (code={code})")
    else:
        log(f"服务被杀掉 (code={code}, 段错误/OOM/外部kill)")


if __name__ == "__main__":
    main()
