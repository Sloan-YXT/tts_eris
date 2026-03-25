"""
Step 4: Style-BERT-VITS2 JP-Extra 微调训练。

由于 Windows torch 2.10 的 gloo 后端不可用，
使用 monkey-patch 方式绕过 DDP，直接单 GPU 训练。
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

# 单 GPU 训练 wrapper 脚本内容
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

# Patch distributed before anything imports it
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

# Patch DDP to just return the model (no-op wrapper)
import torch.nn.parallel
class FakeDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

torch.nn.parallel.DistributedDataParallel = FakeDDP

# Patch DistributedBucketSampler and DistributedLengthGroupedSampler to work without dist
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


def export_single_style() -> None:
    """训练后导出：生成单风格 style_vectors.npy + 更新 model_assets config.json。"""
    import json
    import numpy as np

    # 1. 收集所有 .wav.npy，取平均作为 Neutral 风格向量
    wavs_dir = DATA_DIR / "wavs"
    npy_files = list(wavs_dir.glob("*.wav.npy"))
    if not npy_files:
        print("[WARN] 无 .wav.npy 文件，跳过 style_vectors 生成")
        return

    vectors = [np.load(f) for f in npy_files]
    mean_vec = np.mean(vectors, axis=0)  # shape: (256,)
    style_vectors = mean_vec.reshape(1, -1)  # shape: (1, 256)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    sv_path = ASSETS_DIR / "style_vectors.npy"
    np.save(sv_path, style_vectors)
    print(f"  style_vectors.npy: shape={style_vectors.shape} (单风格 Neutral)")

    # 2. 更新 model_assets/eris/config.json
    config_src = DATA_DIR / "config.json"
    config_dst = ASSETS_DIR / "config.json"
    if config_src.exists():
        cfg = json.loads(config_src.read_text(encoding="utf-8"))
    elif config_dst.exists():
        cfg = json.loads(config_dst.read_text(encoding="utf-8"))
    else:
        print("[WARN] 无 config.json，跳过")
        return

    cfg.setdefault("data", {})
    cfg["data"]["num_styles"] = 1
    cfg["data"]["style2id"] = {"Neutral": 0}
    config_dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  config.json: num_styles=1, style2id={{Neutral: 0}}")


def main() -> None:
    force = "--force" in sys.argv

    wrapper = SBV2_DIR / "_train_single_gpu.py"
    wrapper.write_text(WRAPPER_CODE, encoding="utf-8")

    # checkpoint 管理
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if force:
        old_ckpts = list(MODEL_DIR.glob("*.pth"))
        old_events = list(MODEL_DIR.glob("events.out.*"))
        eval_dir = MODEL_DIR / "eval"
        if old_ckpts or old_events:
            print(f"[--force] 清理旧 checkpoint: {len(old_ckpts)} .pth + {len(old_events)} events")
            for f in old_ckpts + old_events:
                f.unlink()
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
            print("[--force] 删除: eval/")
    elif list(MODEL_DIR.glob("G_*.pth")):
        print("发现已有 checkpoint，将从断点续训（使用 --force 可从头训练）")

    # 确保预训练权重存在
    for name in ["G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"]:
        src = PRETRAINED_DIR / name
        dst = MODEL_DIR / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"复制预训练: {name}")

    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "10086"
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"

    cmd = [PYTHON, str(wrapper)]

    print(f"\n{'='*60}")
    print(f"  开始训练 (JP-Extra, single-GPU)")
    print(f"{'='*60}")
    print(f"  模型: {MODEL_NAME}")
    print(f"  配置: Data/{MODEL_NAME}/config.json")
    print()

    result = subprocess.run(cmd, cwd=str(SBV2_DIR), env=env)
    if result.returncode != 0:
        print(f"\n[FAIL] 训练异常退出 (code={result.returncode})")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  训练完成!")
    print(f"{'='*60}")

    # 检查输出模型
    exports = sorted(ASSETS_DIR.glob("eris_e*_s*.safetensors"), key=lambda p: p.stat().st_mtime)
    if exports:
        print(f"  最新导出: {exports[-1].name}")
    else:
        print("  [WARN] model_assets 中无导出模型")

    # 生成单风格 style_vectors.npy + 更新 config.json
    export_single_style()

    print(f"\n下一步：运行 python run_with_class.py 启动 TTS 服务")


if __name__ == "__main__":
    main()
