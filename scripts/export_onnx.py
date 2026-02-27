"""
Export trained PSQ student model to ONNX format with optional INT8 quantization.

Supports both DistilBERT and DeBERTa-v3-small architectures. The model name
is read from config.json (saved during training), so the correct tokenizer
and ONNX input signature are used automatically.

Produces:
  models/psq-student/model.onnx          — full precision
  models/psq-student/model_quantized.onnx — INT8 quantized
  models/psq-student/tokenizer/           — saved tokenizer for inference
  models/psq-student/onnx_metadata.json   — export metadata

Usage:
  python scripts/export_onnx.py
  python scripts/export_onnx.py --no-quantize
  python scripts/export_onnx.py --checkpoint models/psq-student/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from distill import PSQStudent, DIMENSIONS, N_DIMS, DEFAULTS


def export_onnx(args):
    save_dir = ROOT / "models" / "psq-student"
    checkpoint = Path(args.checkpoint) if args.checkpoint else save_dir / "best.pt"

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        return

    # Load config
    config_path = save_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get("model_name", DEFAULTS["model_name"])
        max_length = config.get("max_length", DEFAULTS["max_length"])
    else:
        model_name = DEFAULTS["model_name"]
        max_length = DEFAULTS["max_length"]

    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Max length: {max_length}")

    # Load model
    print("\nLoading model...")
    model = PSQStudent(model_name, N_DIMS)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    # Force eager attention for ONNX tracing compatibility (transformers >= 5.x
    # defaults to SDPA which uses ops the legacy tracer can't handle)
    if hasattr(model.encoder, "config"):
        model.encoder.config._attn_implementation = "eager"

    # Save tokenizer for JS inference (works for both DistilBERT and DeBERTa)
    # student.js uses @huggingface/transformers AutoTokenizer which reads tokenizer.json
    print("Saving tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_dir = save_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"  Tokenizer saved to {tokenizer_dir}/")
    print(f"  Type: {tokenizer.__class__.__name__}")

    # Detect model type for logging
    is_deberta = "deberta" in model_name.lower()
    print(f"  Architecture: {'DeBERTa' if is_deberta else 'DistilBERT'}")

    # Create dummy inputs — PSQStudent.forward() takes (input_ids, attention_mask)
    # for all architectures (DeBERTa's token_type_ids is optional, defaults to None)
    dummy_ids = torch.randint(0, 1000, (1, max_length), dtype=torch.long)
    dummy_mask = torch.ones(1, max_length, dtype=torch.long)
    dummy_input = (dummy_ids, dummy_mask)
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "scores": {0: "batch"},
        "confidences": {0: "batch"},
    }

    # Export to ONNX
    onnx_path = save_dir / "model.onnx"
    print(f"\nExporting to ONNX: {onnx_path}")
    print(f"  Inputs: {input_names}")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=input_names,
        output_names=["scores", "confidences"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX model: {onnx_size:.1f} MB")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {
        "input_ids": dummy_ids.numpy(),
        "attention_mask": dummy_mask.numpy(),
    }
    ort_scores, ort_confs = session.run(None, ort_inputs)

    # Compare with PyTorch output
    with torch.no_grad():
        pt_scores, pt_confs = model(dummy_ids, dummy_mask)

    score_diff = np.abs(ort_scores - pt_scores.numpy()).max()
    conf_diff = np.abs(ort_confs - pt_confs.numpy()).max()
    print(f"  Max score diff (ONNX vs PyTorch): {score_diff:.6f}")
    print(f"  Max conf diff (ONNX vs PyTorch): {conf_diff:.6f}")

    if score_diff > 0.01 or conf_diff > 0.01:
        print("  WARNING: Large numerical difference detected!")
    else:
        print("  Verification passed.")

    # INT8 quantization
    if not args.no_quantize:
        print("\nQuantizing to INT8...")
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quant_path = save_dir / "model_quantized.onnx"
        quantize_dynamic(
            str(onnx_path),
            str(quant_path),
            weight_type=QuantType.QInt8,
        )

        quant_size = quant_path.stat().st_size / (1024 * 1024)
        print(f"  Quantized model: {quant_size:.1f} MB ({onnx_size / quant_size:.1f}x smaller)")

        # Verify quantized model
        print("  Verifying quantized model...")
        quant_session = ort.InferenceSession(str(quant_path))
        quant_scores, quant_confs = quant_session.run(None, ort_inputs)

        quant_score_diff = np.abs(quant_scores - pt_scores.numpy()).max()
        quant_conf_diff = np.abs(quant_confs - pt_confs.numpy()).max()
        print(f"  Max score diff (quantized vs PyTorch): {quant_score_diff:.6f}")
        print(f"  Max conf diff (quantized vs PyTorch): {quant_conf_diff:.6f}")

    # Save export metadata
    metadata = {
        "model_name": model_name,
        "architecture": "deberta" if is_deberta else "distilbert",
        "max_length": max_length,
        "dimensions": DIMENSIONS,
        "n_dims": N_DIMS,
        "input_names": input_names,
        "onnx_opset": 14,
        "onnx_size_mb": round(onnx_size, 1),
    }
    if not args.no_quantize:
        metadata["quantized_size_mb"] = round(quant_size, 1)

    with open(save_dir / "onnx_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExport complete. Files in {save_dir}/:")
    print(f"  model.onnx           — {onnx_size:.1f} MB (full precision)")
    if not args.no_quantize:
        print(f"  model_quantized.onnx — {quant_size:.1f} MB (INT8)")
    print(f"  tokenizer/           — tokenizer files")
    print(f"  onnx_metadata.json   — export metadata")


def main():
    parser = argparse.ArgumentParser(description="Export PSQ student model to ONNX")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")
    args = parser.parse_args()
    export_onnx(args)


if __name__ == "__main__":
    main()
