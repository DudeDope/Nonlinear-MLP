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

def measure_inference_latency(model, dataloader, device, warmup=10, iters=50):
    model.eval()
    times = []
    import torch
    with torch.no_grad():
        count = 0
        for x, y in dataloader:
            x = x.to(device)
            if x.ndim == 2:
                pass
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            if count >= warmup:
                times.append(end - start)
            count += 1
            if count >= warmup + iters:
                break
    if not times:
        return None
    import numpy as np
    return {
        "mean_latency_s": float(np.mean(times)),
        "p50_latency_s": float(np.percentile(times, 50)),
        "p90_latency_s": float(np.percentile(times, 90)),
        "samples_per_second": float(len(dataloader.dataset) / sum(times)) if sum(times) > 0 else None
    }

def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)