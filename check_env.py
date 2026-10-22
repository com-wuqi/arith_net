"""Quick sanity check for the torchHere environment.

Run with:  uv run python check_env.py
"""

import platform
import time
from pathlib import Path


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


section("Python")
print(f"python      : {platform.python_version()} ({platform.architecture()[0]})")
print(f"executable  : {__import__('sys').executable}")

section("PyTorch")
import torch

print(f"torch       : {torch.__version__}")
print(f"built cuda  : {torch.version.cuda}")
print(f"cudnn       : {torch.backends.cudnn.version()}")
print(f"cuda avail  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    print(f"device      : {torch.cuda.get_device_name(idx)}")
    print(f"capability  : sm_{props.major}{props.minor}")
    print(f"vram        : {props.total_memory / 1024**3:.1f} GiB")

    n = 2048
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    tflops = 2 * n**3 / dt / 1e12
    print(f"matmul {n}x{n}: {dt * 1e3:.2f} ms  (~{tflops:.1f} TFLOP/s fp32)")
    print(f"checksum ok : {torch.isfinite(c).all().item()}")
else:
    print("WARNING: CUDA not available, torch will run on CPU")

section("torchvision")
import torchvision

print(f"torchvision : {torchvision.__version__}")

section("TensorFlow")
import tensorflow as tf

print(f"tensorflow  : {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"tf gpus     : {gpus if gpus else 'none (expected on native Windows)'}")
x = tf.random.normal((256, 256))
y = tf.matmul(x, tf.transpose(x))
print(f"tf matmul ok: {bool(tf.reduce_all(tf.math.is_finite(y)))}")

section("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"matplotlib  : {matplotlib.__version__}")
out = Path("smoke.png")
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot([0, 1, 2, 3], [0, 1, 4, 9], marker="o")
ax.set_title("matplotlib smoke test")
fig.savefig(out, dpi=100)
print(f"saved       : {out.resolve()} ({out.stat().st_size} bytes)")
out.unlink()

section("Result")
print("environment looks good")
