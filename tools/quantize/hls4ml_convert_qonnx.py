"""Attempt the QONNX -> hls4ml conversion for one exported DeepSets model.

There is no Vivado/Vitis HLS on this system, so this does NOT run synthesis
(no LUT/DSP/BRAM/latency numbers). It goes as far as it can without the
Xilinx toolchain:

  1. qonnx cleanup + GemmToMatMul
  2. report which ops are outside hls4ml 1.3's supported-ONNX set
  3. hls4ml.utils.config_from_onnx_model
  4. hls4ml.converters.convert_from_onnx_model
  5. hls_model.compile()            (C-sim build, g++ only -- no Vitis needed)
  6. csim parity: hls_model.predict() vs the qonnx executor on a real batch

Every stage is independent and wrapped: a failure prints STAGE n FAIL <msg>
and the script keeps going so we learn how far the graph gets.

Run with the omnilearned-fpga env python.
"""

import argparse
import os
import traceback

import numpy as np

DATA_PATH = "/global/cfs/cdirs/m4567/www/"

# hls4ml.converters.onnx_to_hls.get_supported_onnx_layers() for hls4ml 1.3.0
HLS4ML_SUPPORTED = {
    "Conv", "Transpose", "Reshape", "Flatten", "Resize", "Pad", "MatMul",
    "Relu", "Tanh", "Sigmoid", "LeakyRelu", "ThresholdedRelu", "Elu", "Selu",
    "PRelu", "Softmax", "Softsign", "Softplus", "BatchNormalization", "Quant",
    "IntQuant", "BipolarQuant", "Add", "Sub", "Mul", "Div", "Average", "Max",
    "Min", "Concat", "Sum", "AveragePool", "MaxPool", "GlobalMaxPool",
    "GlobalAveragePool", "Gemm",  # Gemm is folded by GemmToMatMul
}


def stage(n, name):
    print(f"\n{'='*70}\nSTAGE {n}: {name}\n{'='*70}", flush=True)


def real_batch(size):
    try:
        from omnilearned.dataloader import load_data

        loader = load_data(
            "top", dataset_type="test", use_cond=True, path=DATA_PATH,
            batch=size, num_workers=2, rank=0, size=1, mode="classifier",
            shuffle=False,
        )
        b = next(iter(loader))
        return b["X"].float().numpy(), b["y"].numpy()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] no real batch ({exc}); synthetic input")
        x = np.random.randn(size, 150, 4).astype(np.float32)
        x[:, 50:, 2] = 0.0
        return x, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="the *_clean.onnx to ingest")
    ap.add_argument("--out-dir", required=True, help="hls4ml project output dir")
    ap.add_argument("--backend", default="Vitis")
    ap.add_argument("--io-type", default="io_stream")
    ap.add_argument("--precision", default="fixed<16,6>")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import hls4ml
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup_model
    from qonnx.transformation.gemm_to_matmul import GemmToMatMul

    print(f"hls4ml {hls4ml.__version__}   onnx: {args.onnx}")

    stage(1, "qonnx cleanup + GemmToMatMul")
    model = ModelWrapper(args.onnx)
    try:
        model = cleanup_model(model)
        model = model.transform(GemmToMatMul())
        model = cleanup_model(model)
        print("cleanup OK")
    except Exception:
        traceback.print_exc()
        print("STAGE 1 FAIL")
        return
    cleaned_path = args.onnx.replace(".onnx", "_h4ml.onnx")
    model.save(cleaned_path)
    print(f"saved -> {cleaned_path}")

    stage(2, "op coverage vs hls4ml supported set")
    hist = {}
    for node in model.graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    unsupported = sorted(op for op in hist if op not in HLS4ML_SUPPORTED)
    for op, cnt in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])):
        mark = "" if op in HLS4ML_SUPPORTED else "   <-- UNSUPPORTED"
        print(f"  {cnt:4d}  {op}{mark}")
    print(f"\n{len(model.graph.node)} nodes, "
          f"{len(unsupported)} distinct unsupported op types: {unsupported}")

    x_np, y_np = real_batch(args.batch)

    ref = None
    try:
        from qonnx.core.onnx_exec import execute_onnx

        in_name = model.graph.input[0].name
        ref = execute_onnx(model, {in_name: x_np})[model.graph.output[0].name]
        print(f"qonnx-executor reference logits: shape {ref.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] qonnx executor reference unavailable: {exc}")

    stage(3, "config_from_onnx_model")
    try:
        cfg = hls4ml.utils.config_from_onnx_model(
            model, granularity="name", backend=args.backend,
            default_precision=args.precision,
        )
        print(cfg)
    except Exception:
        traceback.print_exc()
        print("STAGE 3 FAIL -- stopping (no config, cannot convert)")
        return

    stage(4, "convert_from_onnx_model")
    try:
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model, output_dir=args.out_dir, project_name="deepsets",
            backend=args.backend, io_type=args.io_type, hls_config=cfg,
        )
        print("convert OK")
    except Exception:
        traceback.print_exc()
        print("STAGE 4 FAIL -- hls4ml cannot build an IR for this graph")
        return

    stage(5, "hls_model.compile() (C-sim, no Vitis)")
    try:
        hls_model.compile()
        print("compile OK")
    except Exception:
        traceback.print_exc()
        print("STAGE 5 FAIL")
        return

    stage(6, "csim parity vs qonnx executor")
    try:
        pred = hls_model.predict(np.ascontiguousarray(x_np))
        pred = np.asarray(pred).reshape(x_np.shape[0], -1)
        print(f"hls prediction shape {pred.shape}")
        if ref is not None:
            ref2 = ref.reshape(pred.shape)
            print(f"  max |Δ|          : {np.abs(pred - ref2).max():.3e}")
            print(f"  argmax agreement : {(pred.argmax(1) == ref2.argmax(1)).mean()*100:.2f}%")
        if y_np is not None:
            print(f"  batch acc (hls)  : {(pred.argmax(1) == y_np).mean()*100:.2f}%")
    except Exception:
        traceback.print_exc()
        print("STAGE 6 FAIL")
        return

    print("\nALL STAGES PASSED (csim only -- synthesis still needs Vivado).")


if __name__ == "__main__":
    main()
