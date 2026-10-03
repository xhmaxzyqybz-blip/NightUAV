"""Safe, single-image DI-Retinex LOL-v1 normal inference; CUDA only."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

COMMIT = "3f94f95ffb181904a837532a70adc064bee793a2"
WEIGHT_SHA256 = "f7c7dce7c92dd1652947ae02c12047535d52908ee64554a94d175d2a6788e108"
SOURCE = ("https://raw.githubusercontent.com/sunshangquan/DI-retinex-low-light-enhancement/"
          + COMMIT + "/weights/latest%20%2821.54%20lolv1%29.pth")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    inp = args.input.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    metrics = args.metrics.resolve()
    if min(args.width, args.height, args.runs) < 1 or args.warmup < 0:
        raise ValueError("Invalid size or repeat count")
    if not output_dir.is_relative_to((root / "results").resolve()):
        raise ValueError("Output must be inside the project's results directory")
    if not metrics.is_relative_to((root / "experiments").resolve()):
        raise ValueError("Metrics must be inside the project's experiments directory")
    targets = [output_dir / f"input_{args.width}x{args.height}.png",
               output_dir / f"enhanced_{args.width}x{args.height}.png", metrics]
    protected = [(root / p).resolve() for p in ("datasets", "code", "_model_audit", "checkpoints")]
    for target in targets:
        resolved = target.resolve()
        if resolved == inp or any(resolved.is_relative_to(p) for p in protected):
            raise ValueError(f"Unsafe output path: {resolved}")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
    input_hash = sha256(inp)
    repo = root / "_model_audit/DI-retinex-low-light-enhancement"
    source = repo / "mymodel.py"
    weights = root / "checkpoints/di_retinex/latest (21.54 lolv1).pth"
    if not source.is_file() or not weights.is_file():
        raise FileNotFoundError("Official model source or LOL-v1 normal weights missing")
    if git(repo, "rev-parse", "HEAD") != COMMIT:
        raise RuntimeError("Unexpected model repository commit")
    if git(repo, "hash-object", str(source)) != git(repo, "rev-parse", "HEAD:mymodel.py"):
        raise RuntimeError("Model source differs from the audited official commit")
    weight_hash = sha256(weights)
    if weight_hash != WEIGHT_SHA256:
        raise RuntimeError("Official checkpoint SHA256 mismatch")

    import numpy as np
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is disabled")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = importlib.util.spec_from_file_location("nightuav_official_di_model", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start = time.perf_counter()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise RuntimeError("Expected a plain tensor state_dict; unsafe loading is not allowed")
    model = module.enhance_net_nopool(ch=6, number_f=64, linear=False)
    model.load_state_dict(state, strict=True)
    state_shapes = {k: list(v.shape) for k, v in state.items()}
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    load_ms = (time.perf_counter() - start) * 1000
    with Image.open(inp) as original:
        original_size = list(original.size)
        resized = original.convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    torch.cuda.synchronize(device)

    def forward_timed():
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        outputs = model(x)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000
        if not isinstance(outputs, tuple) or len(outputs) != 4:
            raise RuntimeError("Unexpected official model output structure")
        image = outputs[0]
        if image.device != device or image.shape != x.shape:
            raise RuntimeError("Unexpected output device or shape")
        if not all(torch.isfinite(t).all().item() for t in outputs):
            raise RuntimeError("Model output contains NaN or Inf")
        return image, elapsed

    def peak():
        return {"allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}

    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats(device)
        first, cold_ms = forward_timed()
        cold_peak = peak()
        float_range = [float(first.min()), float(first.max())]
        if float_range[0] < 0 or float_range[1] > 1:
            raise RuntimeError("Official enhanced output is outside [0,1]")
        # Save the actual first inference result, not a selected repeated result.
        output_arr = first[0].permute(1, 2, 0).cpu().numpy().copy()
        del first
        for _ in range(args.warmup):
            warm, _elapsed = forward_timed()
            del warm
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(args.runs):
            result, elapsed = forward_timed()
            times.append(elapsed)
            del result
        warm_peak = peak()

    rgb = np.rint(np.clip(output_arr, 0, 1) * 255).astype(np.uint8)
    if np.all(rgb == 0) or np.all(rgb == 255) or float(rgb.std()) == 0:
        raise RuntimeError("Degenerate black/white/constant output")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    for target, image in zip(targets[:2], (resized, Image.fromarray(rgb))):
        with target.open("xb") as stream:
            image.save(stream, format="PNG")
    validations = {}
    for target in targets[:2]:
        with Image.open(target) as saved:
            saved.load()
            if saved.mode != "RGB" or saved.size != (args.width, args.height):
                raise RuntimeError(f"Saved image validation failed: {target}")
            pixels = np.asarray(saved)
            validations[target.name] = {"size": list(saved.size), "mode": saved.mode,
                "min": int(pixels.min()), "max": int(pixels.max()),
                "mean": float(pixels.mean()), "std": float(pixels.std()),
                "fraction_zero_channels": float((pixels == 0).mean()),
                "fraction_255_channels": float((pixels == 255).mean()), "sha256": sha256(target)}
    if sha256(inp) != input_hash:
        raise RuntimeError("Original input hash changed")
    record = {
        "success": True, "utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
        "environment": {"python": sys.version, "executable": sys.executable,
            "torch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
            "numpy": np.__version__, "Pillow": importlib.metadata.version("Pillow"), "cuda": torch.version.cuda},
        "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "model": "DI-Retinex LOL-v1 normal", "repo_commit": COMMIT,
        "model_source_sha256": sha256(source), "weights": str(weights), "weight_source": SOURCE,
        "weight_size_bytes": weights.stat().st_size, "weight_sha256": weight_hash,
        "weights_only": True, "strict_load": True, "state_dict_shapes": state_shapes,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "input": str(inp), "input_sha256_before": input_hash, "input_sha256_after": sha256(inp),
        "original_size": original_size, "inference_size": [args.width, args.height],
        "resize": "Pillow LANCZOS, RGB, in memory", "dtype": "float32", "batch_size": 1,
        "tf32": False, "cudnn_benchmark": False, "model_load_ms": load_ms,
        "first_forward_ms": cold_ms, "first_peak": cold_peak,
        "extra_warmup_runs": args.warmup, "measured_runs": args.runs,
        "warm_forward_ms": times, "warm_mean_ms": statistics.mean(times),
        "warm_median_ms": statistics.median(times), "warm_peak": warm_peak,
        "timing_scope": "CUDA synchronized wall time for model forward only; excludes loading, preprocessing, transfer, validation and saving",
        "memory_scope": "PyTorch allocator peaks; includes resident model/input; reserved cache may persist from prior passes; not total system VRAM",
        "float_output_range": float_range, "finite_outputs": True,
        "saved_images": validations, "output_dir": str(output_dir)}
    with metrics.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DI_RETINEX_INFERENCE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
