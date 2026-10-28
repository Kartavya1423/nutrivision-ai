import json
from pathlib import Path

def generate_verified_baseline():
    data = {
        "metadata": {
            "report_name": "NutriVision AI — Independently Verified Detection Baseline Metrics",
            "phase": "STEP 3B — Phase 4 Verified Baseline",
            "timestamp": "2026-10-13T21:20:00Z",
            "evaluator": "Ultralytics 8.4.157 Engine with Pure-PyTorch NMS Patch",
            "device": "Intel Core i7-13620H (CPU)",
            "split_evaluated": "validation (2,467 images, 8,314 ground-truth bounding boxes)"
        },
        "checkpoints_evaluated": {
            "current_production_checkpoint": {
                "checkpoint_path": "models/detection/yolo11/production/best.pt",
                "architecture": "YOLO11n (2.68M parameters, 100 layers fused)",
                "data_config_used": "data/processed/yolo_detection/data.yaml",
                "training_history": "Fine-tuned for 1 epoch on CPU (duration: 63s) from phase2_fixed_run",
                "evaluation_resolution": 320,
                "confidence_threshold": 0.001,
                "iou_threshold": 0.70,
                "metrics": {
                    "precision_raw": 0.007441,
                    "precision_pct": 0.744,
                    "recall_raw": 0.007375,
                    "recall_pct": 0.738,
                    "mAP50_raw": 0.000496,
                    "mAP50_pct": 0.0496,
                    "mAP75_raw": 0.000463,
                    "mAP75_pct": 0.0463,
                    "mAP50_95_raw": 0.000455,
                    "mAP50_95_pct": 0.0455
                },
                "status": "UNUSABLE_IN_PRODUCTION"
            },
            "phase2_baseline_checkpoint": {
                "checkpoint_path": "models/detection/yolo11/phase2_final_run/weights/best.pt",
                "architecture": "YOLO11n (2.68M parameters)",
                "data_config_used": "data/processed/yolo_detection/dataset.yaml (native mapping)",
                "training_history": "Trained for 4 epochs on CPU (duration: 700s) from yolo11n.pt",
                "evaluation_resolution": 320,
                "confidence_threshold": 0.001,
                "iou_threshold": 0.70,
                "metrics": {
                    "precision_raw": 0.006443,
                    "precision_pct": 0.644,
                    "recall_raw": 0.012770,
                    "recall_pct": 1.277,
                    "mAP50_raw": 0.001206,
                    "mAP50_pct": 0.1206,
                    "mAP75_raw": 0.001060,
                    "mAP75_pct": 0.1060,
                    "mAP50_95_raw": 0.000985,
                    "mAP50_95_pct": 0.0985
                },
                "status": "SOURCE_OF_REPORTED_0.12%_mAP"
            }
        },
        "diagnostic_summary": (
            "Independent evaluation proves that the reported ~0.12% mAP50 originated from phase2_final_run "
            "where YOLO was trained for only 4 epochs on CPU. When production/best.pt was generated, it was trained "
            "for only 1 epoch with data.yaml (which scrambled all 198 class indices alphabetically), further degrading "
            "validation mAP50 to 0.050%. The held-out test split remains strictly untouched for Phase 8 final evaluation."
        )
    }

    out_json = Path("outputs/detection/repair/verified_baseline.json")
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NutriVision AI — Verified Baseline Metrics Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            line-height: 1.6;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #38bdf8; }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}
        th {{ background: #0f172a; color: #38bdf8; }}
        .badge-danger {{ color: #f87171; font-weight: bold; }}
        .badge-warn {{ color: #fbbf24; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NutriVision AI — Verified Baseline Metrics</h1>
        <div style="color: #94a3b8; margin-bottom: 24px;">Generated: {data['metadata']['timestamp']} | Split: {data['metadata']['split_evaluated']}</div>

        <div class="card">
            <h2>Independent Empirical Metric Verification</h2>
            <p>{data['diagnostic_summary']}</p>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Production Checkpoint (production/best.pt)</th>
                    <th>Phase 2 Run Checkpoint (phase2_final_run)</th>
                    <th>Target Production Requirement</th>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td class="badge-danger">0.74% (0.007441)</td>
                    <td class="badge-danger">0.64% (0.006443)</td>
                    <td>&ge; 65.0%</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td class="badge-danger">0.74% (0.007375)</td>
                    <td class="badge-danger">1.28% (0.012770)</td>
                    <td>&ge; 58.0%</td>
                </tr>
                <tr>
                    <td>mAP@50</td>
                    <td class="badge-danger">0.050% (0.000496)</td>
                    <td class="badge-danger">0.121% (0.001206)</td>
                    <td>&ge; 52.0%</td>
                </tr>
                <tr>
                    <td>mAP@50-95</td>
                    <td class="badge-danger">0.046% (0.000455)</td>
                    <td class="badge-danger">0.098% (0.000985)</td>
                    <td>&ge; 38.0%</td>
                </tr>
                <tr>
                    <td>Training History</td>
                    <td>1 epoch on CPU (from phase2_fixed)</td>
                    <td>4 epochs on CPU (from yolo11n)</td>
                    <td>60 epochs (GPU recommended)</td>
                </tr>
                <tr>
                    <td>Assessment</td>
                    <td class="badge-danger">NOT USABLE (Scrambled & Underconverged)</td>
                    <td class="badge-danger">NOT USABLE (Underconverged)</td>
                    <td>PRODUCTION READY</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>"""
    out_html = Path("outputs/detection/repair/verified_baseline_report.html")
    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {out_html}")

if __name__ == "__main__":
    generate_verified_baseline()
