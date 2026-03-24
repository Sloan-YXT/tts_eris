"""
GPT-SoVITS 一键训练脚本（v2Pro，艾莉丝）

用法（在 tts_eris/ 目录下）：
    python train.py            # 断点续训（跳过已完成步骤）
    python train.py --retrain  # 清除预处理缓存，从头重训

流程：
    1a  文本分词与特征提取
    1b  HuBERT + wav32k
    1b2 Speaker Verification（v2Pro 需要）
    1c  语义 Token 提取
    2   SoVITS 训练（音色）
    3   GPT 训练（韵律）
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from subprocess import Popen

# ── 路径配置 ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
GSV_DIR    = SCRIPT_DIR / "GPT-SoVITS"

# 切换到 GPT-SoVITS/ 作为工作目录（所有相对路径以此为基准）
os.chdir(GSV_DIR)

PYTHON = sys.executable

# ── 训练参数（按需修改） ───────────────────────────────────────────────────────

VERSION   = "v2Pro"
EXP_NAME  = "eris"
GPU       = "0"

# 合并 output/asr_opt/ 下所有 *.list 生成的联合标注文件
INP_TEXT    = "output/asr_opt/combined.list"
INP_WAV_DIR = ""   # 留空，让预处理脚本直接用 list 里的完整路径（支持子目录）

SOVITS_BATCH    = 8
SOVITS_EPOCHS   = 20
SOVITS_SAVE_EVERY = 5

GPT_BATCH       = 8
GPT_EPOCHS      = 50
GPT_SAVE_EVERY  = 10

# ── 预训练模型路径 ──────────────────────────────────────────────────────────────

BERT_DIR      = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
HUBERT_DIR    = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
SV_PATH       = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
PRETRAINED_S2G = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
PRETRAINED_S2D = "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth"
PRETRAINED_S1  = "GPT_SoVITS/pretrained_models/s1v3.ckpt"

S2_CONFIG_TPL  = "GPT_SoVITS/configs/s2v2Pro.json"
S1_CONFIG_TPL  = "GPT_SoVITS/configs/s1longer-v2.yaml"

# ── 衍生路径 ───────────────────────────────────────────────────────────────────

EXP_DIR    = f"logs/{EXP_NAME}"
TEMP_DIR   = "TEMP"
TMP_S2     = f"{TEMP_DIR}/tmp_s2.json"
TMP_S1     = f"{TEMP_DIR}/tmp_s1.yaml"

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(f"{EXP_DIR}/logs_s2_{VERSION}", exist_ok=True)
os.makedirs(f"{EXP_DIR}/logs_s1_{VERSION}", exist_ok=True)

# ── 工具函数 ───────────────────────────────────────────────────────────────────

def banner(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run(cmd: str, extra_env: dict | None = None) -> int:
    """运行命令，继承当前环境，可追加额外 env var，返回退出码。"""
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    print(f"\n>>> {cmd}")
    t0 = time.time()
    p = Popen(cmd, shell=True, env=env)
    p.wait()
    elapsed = time.time() - t0
    print(f"    -> 耗时 {elapsed:.0f}s，退出码 {p.returncode}")
    return p.returncode


def die(msg: str):
    print(f"\n[ERROR] {msg}")
    sys.exit(1)


def check_file(path: str, label: str):
    if not Path(path).exists():
        die(f"{label} 不存在: {path}")


# ── 合并多源标注 ───────────────────────────────────────────────────────────────

def merge_lists():
    """合并 output/asr_opt/ 下所有 *.list（含子目录，按路径排序），输出 combined.list。"""
    asr_dir = Path("output/asr_opt")
    sources = sorted(asr_dir.glob("**/*.list"))
    if not sources:
        die("output/asr_opt/ 下没有找到任何 *.list 文件")

    all_lines = []
    for f in sources:
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        all_lines.extend(lines)
        print(f"  {f.name}: {len(lines)} 行")

    out = asr_dir / "combined.list"
    out.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    print(f"  -> 合并完成，共 {len(all_lines)} 行 -> {out}")


# ── 前置检查 ───────────────────────────────────────────────────────────────────

def preflight():
    banner("前置检查")
    check_file(INP_TEXT,     "标注文件")
    if INP_WAV_DIR:
        check_file(INP_WAV_DIR,  "音频目录")
    check_file(BERT_DIR,     "BERT 预训练模型")
    check_file(HUBERT_DIR,   "HuBERT 预训练模型")
    check_file(SV_PATH,      "SV 预训练模型")
    check_file(PRETRAINED_S2G, "SoVITS-G 预训练模型")
    check_file(PRETRAINED_S2D, "SoVITS-D 预训练模型")
    check_file(PRETRAINED_S1,  "GPT 预训练模型")
    print("  所有预置文件检查通过。")


# ── 1a  文本分词与特征提取 ──────────────────────────────────────────────────────

def step_1a():
    banner("1a — 文本分词与特征提取")

    out_merged = f"{EXP_DIR}/2-name2text.txt"
    if Path(out_merged).exists():
        lines = Path(out_merged).read_text(encoding="utf-8").strip().splitlines()
        if len(lines) >= 2:
            print(f"  已存在且有效（{len(lines)} 行），跳过。")
            return

    env = {
        "inp_text":            INP_TEXT,
        "inp_wav_dir":         INP_WAV_DIR,
        "exp_name":            EXP_NAME,
        "opt_dir":             EXP_DIR,
        "bert_pretrained_dir": BERT_DIR,
        "is_half":             "True",
        "i_part":              "0",
        "all_parts":           "1",
        "_CUDA_VISIBLE_DEVICES": GPU,
    }
    rc = run(f'"{PYTHON}" -s GPT_SoVITS/prepare_datasets/1-get-text.py', env)
    if rc != 0:
        die("1a 失败")

    # 合并分片输出
    part_path = f"{EXP_DIR}/2-name2text-0.txt"
    check_file(part_path, "1a 输出")
    lines = Path(part_path).read_text(encoding="utf-8").strip().splitlines()
    Path(out_merged).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(part_path).unlink()
    print(f"  合并完成，共 {len(lines)} 行 → {out_merged}")


# ── 1b  HuBERT + wav32k ────────────────────────────────────────────────────────

def step_1b():
    banner("1b — HuBERT 特征 + wav32k 重采样")

    wav32k_dir = f"{EXP_DIR}/5-wav32k"
    if Path(wav32k_dir).exists() and any(Path(wav32k_dir).iterdir()):
        print(f"  {wav32k_dir} 已有内容，跳过。")
    else:
        env = {
            "inp_text":          INP_TEXT,
            "inp_wav_dir":       INP_WAV_DIR,
            "exp_name":          EXP_NAME,
            "opt_dir":           EXP_DIR,
            "cnhubert_base_dir": HUBERT_DIR,
            "sv_path":           SV_PATH,
            "i_part":            "0",
            "all_parts":         "1",
            "_CUDA_VISIBLE_DEVICES": GPU,
        }
        rc = run(f'"{PYTHON}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py', env)
        if rc != 0:
            die("1b HuBERT 失败")


def step_1b_sv():
    banner("1b2 — Speaker Verification（v2Pro）")

    sv_dir = f"{EXP_DIR}/7-SV"
    if Path(sv_dir).exists() and any(Path(sv_dir).iterdir()):
        print(f"  {sv_dir} 已有内容，跳过。")
        return

    env = {
        "inp_text":    INP_TEXT,
        "inp_wav_dir": INP_WAV_DIR,
        "exp_name":    EXP_NAME,
        "opt_dir":     EXP_DIR,
        "sv_path":     SV_PATH,
        "i_part":      "0",
        "all_parts":   "1",
        "_CUDA_VISIBLE_DEVICES": GPU,
    }
    rc = run(f'"{PYTHON}" -s GPT_SoVITS/prepare_datasets/2-get-sv.py', env)
    if rc != 0:
        die("1b2 SV 失败")


# ── 1c  语义 Token 提取 ────────────────────────────────────────────────────────

def step_1c():
    banner("1c — 语义 Token 提取")

    s2_config_file = f"GPT_SoVITS/configs/s2{VERSION}.json"
    out_merged = f"{EXP_DIR}/6-name2semantic.tsv"

    if Path(out_merged).exists() and Path(out_merged).stat().st_size > 31:
        print(f"  已存在且有效，跳过。")
        return

    env = {
        "inp_text":      INP_TEXT,
        "exp_name":      EXP_NAME,
        "opt_dir":       EXP_DIR,
        "pretrained_s2G": PRETRAINED_S2G,
        "s2config_path": s2_config_file,
        "i_part":        "0",
        "all_parts":     "1",
        "_CUDA_VISIBLE_DEVICES": GPU,
    }
    rc = run(f'"{PYTHON}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py', env)
    if rc != 0:
        die("1c 失败")

    # 合并分片输出
    part_path = f"{EXP_DIR}/6-name2semantic-0.tsv"
    check_file(part_path, "1c 输出")
    rows = Path(part_path).read_text(encoding="utf-8").strip().splitlines()
    merged = ["item_name\tsemantic_audio"] + rows
    Path(out_merged).write_text("\n".join(merged) + "\n", encoding="utf-8")
    Path(part_path).unlink()
    print(f"  合并完成，共 {len(rows)} 条 → {out_merged}")


# ── 2  SoVITS 训练 ─────────────────────────────────────────────────────────────

def step_sovits():
    banner(f"2 — SoVITS 训练（{SOVITS_EPOCHS} epoch）")

    with open(S2_CONFIG_TPL, encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["train"]["batch_size"]          = SOVITS_BATCH
    cfg["train"]["epochs"]              = SOVITS_EPOCHS
    cfg["train"]["text_low_lr_rate"]    = 0.4
    cfg["train"]["pretrained_s2G"]      = PRETRAINED_S2G
    cfg["train"]["pretrained_s2D"]      = PRETRAINED_S2D
    cfg["train"]["if_save_latest"]      = True
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["save_every_epoch"]    = SOVITS_SAVE_EVERY
    cfg["train"]["gpu_numbers"]         = GPU
    cfg["train"]["grad_ckpt"]           = False
    cfg["train"]["fp16_run"]            = True
    cfg["model"]["version"]             = VERSION
    cfg["data"]["exp_dir"]              = EXP_DIR
    cfg["s2_ckpt_dir"]                  = EXP_DIR
    cfg["save_weight_dir"]              = f"SoVITS_weights_{VERSION}"
    cfg["name"]                         = EXP_NAME
    cfg["version"]                      = VERSION

    with open(TMP_S2, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    os.makedirs(f"SoVITS_weights_{VERSION}", exist_ok=True)
    rc = run(f'"{PYTHON}" -s GPT_SoVITS/s2_train.py --config "{TMP_S2}"')
    if rc != 0:
        die("SoVITS 训练失败")


# ── 3  GPT 训练 ────────────────────────────────────────────────────────────────

def step_gpt():
    banner(f"3 — GPT 训练（{GPT_EPOCHS} epoch）")

    import yaml

    with open(S1_CONFIG_TPL, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["train"]["batch_size"]            = GPT_BATCH
    cfg["train"]["epochs"]                = GPT_EPOCHS
    cfg["pretrained_s1"]                  = PRETRAINED_S1
    cfg["train"]["save_every_n_epoch"]    = GPT_SAVE_EVERY
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["if_save_latest"]        = True
    cfg["train"]["if_dpo"]                = False
    cfg["train"]["half_weights_save_dir"] = f"GPT_weights_{VERSION}"
    cfg["train"]["exp_name"]              = EXP_NAME
    cfg["train_semantic_path"]            = f"{EXP_DIR}/6-name2semantic.tsv"
    cfg["train_phoneme_path"]             = f"{EXP_DIR}/2-name2text.txt"
    cfg["output_dir"]                     = f"{EXP_DIR}/logs_s1_{VERSION}"

    with open(TMP_S1, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    os.makedirs(f"GPT_weights_{VERSION}", exist_ok=True)
    rc = run(
        f'"{PYTHON}" -s GPT_SoVITS/s1_train.py --config_file "{TMP_S1}"',
        extra_env={"_CUDA_VISIBLE_DEVICES": GPU, "hz": "25hz"},
    )
    if rc != 0:
        die("GPT 训练失败")


# ── 完成提示 ───────────────────────────────────────────────────────────────────

def print_result():
    banner("训练完成")
    sovits_dir = GSV_DIR / f"SoVITS_weights_{VERSION}"
    gpt_dir    = GSV_DIR / f"GPT_weights_{VERSION}"

    print("  SoVITS 模型：")
    for f in sorted(sovits_dir.glob("*.pth")):
        print(f"    {f.name}")

    print("  GPT 模型：")
    for f in sorted(gpt_dir.glob("*.ckpt")):
        print(f"    {f.name}")

    print(f"""
  下一步：在 voices/eris/config.json 中填入模型路径，
  然后重启 run.py 即可使用微调后的模型推理。
""")


# ── 清除预处理缓存 ─────────────────────────────────────────────────────────────

def clean_cache():
    banner("清除预处理缓存")
    targets_files = [
        Path(f"{EXP_DIR}/2-name2text.txt"),
        Path(f"{EXP_DIR}/6-name2semantic.tsv"),
    ]
    targets_dirs = [
        Path(f"{EXP_DIR}/5-wav32k"),
        Path(f"{EXP_DIR}/7-SV"),
    ]
    for f in targets_files:
        if f.exists():
            f.unlink()
            print(f"  删除  {f}")
        else:
            print(f"  跳过  {f}（不存在）")
    for d in targets_dirs:
        if d.exists():
            shutil.rmtree(d)
            print(f"  删除  {d}/")
        else:
            print(f"  跳过  {d}/（不存在）")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="清除预处理缓存，从头重训")
    args = parser.parse_args()

    t_start = time.time()

    merge_lists()
    if args.retrain:
        clean_cache()
    preflight()
    step_1a()
    step_1b()
    step_1b_sv()
    step_1c()
    step_sovits()
    step_gpt()
    print_result()

    total = time.time() - t_start
    print(f"  总耗时：{total/60:.1f} 分钟")
