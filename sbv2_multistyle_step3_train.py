"""
多风格训练 Step 3: 训练 (JP-Extra, single-GPU)。

1. 清理旧训练 checkpoint (.pth)，准备从预训练权重开始
2. 复制 JP-Extra 预训练权重到 model 目录
3. 清理旧 model_assets/eris/ 导出（已备份到 eris_v1_neutral）
4. 启动训练（自动调用 default_style.py 检测子目录 → 生成多风格 style_vectors.npy）

使用与 sbv2_step4_train.py 相同的 DDP bypass 方案。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"
PYTHON = sys.executable
MODEL_NAME = "eris"
DATA_DIR = SBV2_DIR / "Data" / MODEL_NAME
MODEL_DIR = DATA_DIR / "models"
ASSETS_DIR = SBV2_DIR / "model_assets" / MODEL_NAME
PRETRAINED_DIR = SBV2_DIR / "pretrained_jp_extra"

# 单 GPU 训练 wrapper（与 sbv2_step4_train.py 相同）
WRAPPER_CODE = r'''"""
Single-GPU wrapper: patches torch.distributed for non-functional gloo on Windows.
"""
import os
import multiprocessing

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "10086")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import torch.distributed as dist

_fake_initialized = False

def _fake_init_process_group(*a, **kw):
    global _fake_initialized
    _fake_initialized = True

def _fake_get_rank(*a, **kw):
    return 0

def _fake_get_world_size(*a, **kw):
    return 1

def _fake_is_initialized():
    return _fake_initialized

def _fake_barrier(*a, **kw):
    pass

def _fake_all_reduce(*a, **kw):
    pass

def _fake_broadcast(*a, **kw):
    pass

def _fake_destroy_process_group(*a, **kw):
    pass

dist.init_process_group = _fake_init_process_group
dist.get_rank = _fake_get_rank
dist.get_world_size = _fake_get_world_size
dist.is_initialized = _fake_is_initialized
dist.barrier = _fake_barrier
dist.all_reduce = _fake_all_reduce
dist.broadcast = _fake_broadcast
dist.destroy_process_group = _fake_destroy_process_group

import torch.nn.parallel
class FakeDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

torch.nn.parallel.DistributedDataParallel = FakeDDP

import data_utils

class FakeBucketSampler(torch.utils.data.Sampler):
    """Drop-in replacement for DistributedBucketSampler for single-GPU."""
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.shuffle = shuffle
        self.buckets = [[] for _ in range(len(boundaries) - 1)]
        for i in range(len(dataset)):
            length = dataset.lengths[i] if hasattr(dataset, 'lengths') else 0
            for j in range(len(boundaries) - 1):
                if boundaries[j] <= length < boundaries[j+1]:
                    self.buckets[j].append(i)
                    break

    def __iter__(self):
        import random
        batches = []
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return sum(len(b) // self.batch_size for b in self.buckets)

data_utils.DistributedBucketSampler = FakeBucketSampler

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import train_ms_jp_extra
    train_ms_jp_extra.run()
'''


def prepare_model_dir() -> None:
    """清理旧 checkpoint，复制预训练权重。"""
    # 删除旧 .pth 文件（防止 resume 逻辑走错路径）
    for pth in MODEL_DIR.glob("*.pth"):
        pth.unlink()
        print(f"    删除: {pth.name}")

    # 删除旧 eval 目录
    eval_dir = MODEL_DIR / "eval"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
        print(f"    删除: eval/")

    # 删除旧 tfevents
    for tf in MODEL_DIR.glob("events.out.*"):
        tf.unlink()
        print(f"    删除: {tf.name}")

    # 复制预训练权重
    for name in ["G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"]:
        src = PRETRAINED_DIR / name
        dst = MODEL_DIR / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"    复制预训练: {name} ({src.stat().st_size // (1024*1024)}MB)")
        elif dst.exists():
            print(f"    预训练已存在: {name}")
        else:
            print(f"    [WARN] 预训练不存在: {src}")


def clean_old_exports() -> None:
    """清理 model_assets/eris/ 中的旧导出（已备份到 eris_v1_neutral）。"""
    backup = SBV2_DIR / "model_assets" / "eris_v1_neutral"
    if not backup.exists():
        print("    [WARN] 备份不存在，跳过清理")
        return

    for sf in ASSETS_DIR.glob("eris_e*_s*.safetensors"):
        sf.unlink()
        print(f"    删除旧导出: {sf.name}")


def main() -> None:
    print(f"{'='*60}")
    print(f"  多风格训练 Step 3: 训练")
    print(f"{'='*60}")

    # 1. 准备模型目录
    print(f"\n[1/3] 准备模型目录...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prepare_model_dir()

    # 2. 清理旧导出
    print(f"\n[2/3] 清理旧 model_assets 导出...")
    clean_old_exports()

    # 3. 写入 wrapper 并启动训练
    print(f"\n[3/3] 启动训练...")
    wrapper = SBV2_DIR / "_train_single_gpu.py"
    wrapper.write_text(WRAPPER_CODE, encoding="utf-8")

    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "10086"
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"

    cmd = [PYTHON, str(wrapper)]

    print(f"  模型: {MODEL_NAME}")
    print(f"  配置: Data/{MODEL_NAME}/config.json")
    print(f"  wavs 子目录 → default_style.py 自动检测多风格")
    print()

    result = subprocess.run(cmd, cwd=str(SBV2_DIR), env=env)
    if result.returncode != 0:
        print(f"\n[FAIL] 训练异常退出 (code={result.returncode})")
        sys.exit(1)

    # 检查结果
    print(f"\n{'='*60}")
    print(f"  训练完成!")
    print(f"{'='*60}")

    # 检查 style_vectors.npy
    style_vec = ASSETS_DIR / "style_vectors.npy"
    if style_vec.exists():
        import numpy as np
        sv = np.load(style_vec)
        print(f"  style_vectors.npy: shape={sv.shape} (应为 (8, 256))")

    # 检查 config.json 中的 style2id
    import json
    config_path = ASSETS_DIR / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        style2id = cfg.get("data", {}).get("style2id", {})
        num_styles = cfg.get("data", {}).get("num_styles", 0)
        print(f"  num_styles: {num_styles}")
        print(f"  style2id: {style2id}")

    # 检查导出模型
    exports = sorted(ASSETS_DIR.glob("eris_e*_s*.safetensors"), key=lambda p: p.stat().st_mtime)
    if exports:
        print(f"  最新导出: {exports[-1].name}")
    else:
        print("  [WARN] 无导出模型")

    print(f"\n下一步: 运行 sbv2_multistyle_step4_test.py 测试多风格推理")


if __name__ == "__main__":
    main()
