import torch
import time
import psutil
import os

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        out = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            out.append((correct_k * (100.0 / batch_size)).item())
        return out

def _prepare_for_model(model, x):
    # Flatten images for MLPs (e.g., MNIST 1x28x28 -> 784)
    if x.ndim == 4 and model.__class__.__name__ == "MLP":
        x = x.view(x.size(0), -1)
    return x

def measure_inference_latency(model, dataloader, device, warmup=10, iters=50, sync_cuda=True):
    model.eval()
    times = []
    measured_samples = 0
    with torch.no_grad():
        count = 0
        for x, _ in dataloader:
            x = x.to(device)
            x = _prepare_for_model(model, x)

            if sync_cuda and device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if sync_cuda and device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()

            if count >= warmup:
                times.append(end - start)
                measured_samples += x.size(0)
            count += 1
            if count >= warmup + iters:
                break
    if not times:
        return None
    import numpy as np
    total_time = sum(times)
    return {
        "mean_latency_s": float(np.mean(times)),
        "p50_latency_s": float(np.percentile(times, 50)),
        "p90_latency_s": float(np.percentile(times, 90)),
        "samples_per_second": float(measured_samples / total_time) if total_time > 0 else None,
        "batches_measured": len(times),
    }

def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)
