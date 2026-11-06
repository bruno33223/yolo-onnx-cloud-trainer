#!/usr/bin/env python3
"""
train_cloud.py - YOLO Cloud Trainer (Stateless & Zero Data Retention)
Runs exclusively in ephemeral cloud containers (GitHub Actions / Google Colab).
NO images or annotations are stored or persisted after training completes.
"""

import os
import sys
import glob
import json
import base64
import shutil
import argparse
from pathlib import Path
import tempfile

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Cloud Trainer - Zero Retention")
    parser.add_argument("-d", "--dataset", default=None, help="Path to dataset JSON file")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("-b", "--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("-s", "--imgsz", type=int, default=640, help="Input resolution (default: 640)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory for the final .onnx")
    parser.add_argument("-m", "--model-name", default="trained_model.onnx", help="Final ONNX file name")
    return parser.parse_args()

def find_dataset_file(specified_path=None):
    if specified_path and Path(specified_path).exists():
        return Path(specified_path)
    
    # Search for any json containing 'dataset' or general json files
    candidates = list(Path(".").glob("*dataset*.json")) + list(Path(".").glob("*.json"))
    candidates = [c for c in candidates if c.name not in ["package.json", "tsconfig.json", "metadata.json"]]
    
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]
    
    return None

def main():
    args = parse_args()
    print("=" * 70)
    print("  🚀 YOLO CLOUD TRAINER (STATELESS & ZERO-RETENTION)")
    print("=" * 70)

    dataset_path = find_dataset_file(args.dataset)
    if not dataset_path:
        print("[ERROR] No dataset .json file found for training.")
        print("Please ensure a valid dataset JSON file is provided.")
        sys.exit(1)

    print(f"[*] Identified dataset : {dataset_path.name}")
    print(f"[*] Training epochs    : {args.epochs}")
    print(f"[*] Batch size         : {args.batch}")
    print(f"[*] Resolution (imgsz) : {args.imgsz}")

    # Create an isolated ephemeral temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="yolo_cloud_trainer_"))
    print(f"[*] Ephemeral workspace created at: {temp_dir}")

    try:
        # 1. Load dataset from JSON
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_images = data.get("images", data) if isinstance(data, dict) else data
        if not isinstance(raw_images, list) or len(raw_images) == 0:
            print("[ERROR] Invalid format: image list is empty or missing in JSON.")
            sys.exit(1)

        print(f"[✓] Images loaded into memory: {len(raw_images)}")

        # 2. Dynamic class discovery
        class_list = []
        for img in raw_images:
            for ann in img.get("annotations", []):
                base_lbl = ann.get("label") or ann.get("nomePeca") or "OBJECT"
                base_lbl = base_lbl.strip()
                status = str(ann.get("status", "")).strip().upper()
                
                # Append status if present and not already suffixed
                if status in ["OK", "NOK"] and not base_lbl.upper().endswith(f"_{status}"):
                    final_label = f"{base_lbl}_{status}"
                else:
                    final_label = base_lbl
                
                if final_label not in class_list:
                    class_list.append(final_label)

        if not class_list:
            class_list = ["OBJECT"]

        class_to_id = {name: i for i, name in enumerate(class_list)}
        print(f"[✓] {len(class_list)} class(es) detected:")
        for name, idx in class_to_id.items():
            print(f"    [{idx}] {name}")

        # 3. Structure YOLO dataset on ephemeral disk
        train_img_dir = temp_dir / "images" / "train"
        train_lbl_dir = temp_dir / "labels" / "train"
        val_img_dir = temp_dir / "images" / "val"
        val_lbl_dir = temp_dir / "labels" / "val"

        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        valid_count = 0
        for idx, img in enumerate(raw_images):
            data_url = img.get("dataUrl") or img.get("imageData") or ""
            if not data_url or "," not in data_url:
                continue

            try:
                base64_data = data_url.split(",")[1]
                img_bytes = base64.b64decode(base64_data)
            except Exception as e:
                print(f"[WARN] Failed to decode image {idx}: {e}")
                continue

            # Split 85% train, 15% validation
            is_val = (idx % 7 == 0) and len(raw_images) >= 5
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            file_stem = f"frame_{idx:05d}"
            img_file = target_img_dir / f"{file_stem}.jpg"
            lbl_file = target_lbl_dir / f"{file_stem}.txt"

            with open(img_file, "wb") as f_img:
                f_img.write(img_bytes)

            # YOLO annotation lines
            annotations = img.get("annotations", [])
            lines = []
            for ann in annotations:
                base_lbl = ann.get("label") or ann.get("nomePeca") or "OBJECT"
                base_lbl = base_lbl.strip()
                status = str(ann.get("status", "")).strip().upper()
                if status in ["OK", "NOK"] and not base_lbl.upper().endswith(f"_{status}"):
                    final_label = f"{base_lbl}_{status}"
                else:
                    final_label = base_lbl

                cls_id = class_to_id.get(final_label, 0)

                # Prioritize pre-computed normalized YOLO coordinates (x_center, y_center, width, height)
                yolo_coords = ann.get("yolo")
                if yolo_coords and "x_center" in yolo_coords:
                    xc = yolo_coords["x_center"]
                    yc = yolo_coords["y_center"]
                    w = yolo_coords["width"]
                    h = yolo_coords["height"]
                else:
                    # Fallback to bbox_percent
                    bbox = ann.get("bbox_percent", ann.get("bbox", {}))
                    bx = bbox.get("x", 0)
                    by = bbox.get("y", 0)
                    bw = bbox.get("width", 0)
                    bh = bbox.get("height", 0)

                    # Normalize if given in 0..100 scale
                    if bx > 1.0 or by > 1.0 or bw > 1.0 or bh > 1.0:
                        bx /= 100.0
                        by /= 100.0
                        bw /= 100.0
                        bh /= 100.0

                    xc = bx + (bw / 2.0)
                    yc = by + (bh / 2.0)
                    w = bw
                    h = bh

                # Clamp values strictly between 0.0 and 1.0
                xc = max(0.001, min(0.999, float(xc)))
                yc = max(0.001, min(0.999, float(yc)))
                w = max(0.001, min(1.0, float(w)))
                h = max(0.001, min(1.0, float(h)))

                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            with open(lbl_file, "w", encoding="utf-8") as f_lbl:
                f_lbl.write("\n".join(lines) + "\n")

            valid_count += 1

        print(f"[✓] {valid_count} images prepared on ephemeral disk.")

        # 4. Generate data.yaml
        yaml_content = f"""
path: '{temp_dir.as_posix()}'
train: images/train
val: images/val
names:
"""
        for idx, name in enumerate(class_list):
            yaml_content += f"  {idx}: '{name}'\n"

        data_yaml = temp_dir / "data.yaml"
        with open(data_yaml, "w", encoding="utf-8") as f:
            f.write(yaml_content.strip())

        # 5. Initialize Ultralytics YOLO
        from ultralytics import YOLO
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Cloud CPU Runner"
        print(f"[*] Training device: {device_name} ({device})")

        print("[*] Loading base YOLO model (nano)...")
        model = YOLO("yolov8n.pt")

        # 6. Train model
        print(f"[*] Starting {args.epochs} training epochs...")
        train_results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            patience=15,
            workers=2,
            project=str(temp_dir / "runs"),
            name="train_job",
            exist_ok=True,
            verbose=True,
            val=True
        )

        # 7. Export to ONNX (WebAssembly Ready)
        print("[*] Exporting trained model to ONNX...")
        best_pt = temp_dir / "runs" / "train_job" / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = temp_dir / "runs" / "train_job" / "weights" / "last.pt"

        best_model = YOLO(str(best_pt))
        exported_onnx = best_model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=17,
            dynamic=False,
            simplify=True
        )

        # 8. Copy generated .onnx to output folder
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_onnx_path = output_dir / args.model_name

        shutil.copy(exported_onnx, final_onnx_path)
        onnx_size_mb = final_onnx_path.stat().st_size / (1024 * 1024)

        # Write metadata file alongside model
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f_meta:
            json.dump({
                "model_name": args.model_name,
                "classes": class_list,
                "epochs": args.epochs,
                "resolution": args.imgsz,
                "size_mb": round(onnx_size_mb, 2)
            }, f_meta, indent=2)

        print("=" * 70)
        print(f"  🎉 TRAINING SUCCESSFULLY COMPLETED!")
        print(f"  📦 Generated ONNX Model : {final_onnx_path} ({onnx_size_mb:.2f} MB)")
        print(f"  🏷️  Trained Classes     : {', '.join(class_list)}")
        print("=" * 70)

    finally:
        # 9. ZERO-RETENTION: Completely purge all raw images and temporary folders
        print("[*] Applying Zero-Retention Policy: Purging temporary images...")
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("[✓] All temporary images and folders have been completely wiped from memory.")
        except Exception as cleanup_err:
            print(f"[WARN] Cleanup warning: {cleanup_err}")

if __name__ == "__main__":
    main()
