"""
Step 4: Style-BERT-VITS2 JP-Extra 微调训练。

由于 Windows torch 2.10 的 gloo 后端不可用，
使用 monkey-patch 方式绕过 DDP，直接单 GPU 训练。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"
PYTHON = sys.executable
MODEL_NAME = "eris"

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


def main() -> None:
    wrapper = SBV2_DIR / "_train_single_gpu.py"
    wrapper.write_text(WRAPPER_CODE, encoding="utf-8")

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
    model_dir = SBV2_DIR / "Data" / MODEL_NAME / "models"
    if model_dir.exists():
        models = sorted(model_dir.glob("G_*.safetensors"))
        if models:
            latest = models[-1]
            print(f"  最新模型: {latest.name}")
        else:
            print("  未找到生成器模型文件")
    else:
        print(f"  模型目录不存在: {model_dir}")

    print(f"\n下一步：运行 sbv2_step5_style_gen.py 生成风格向量")


if __name__ == "__main__":
    main()
