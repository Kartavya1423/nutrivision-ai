"""
NutriVision AI — Dataset Report & Manifest Generator
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Generates:
1. data/processed/dataset_purpose.json (intended module mapping)
2. data/processed/dataset_manifest.json (complete dataset inventory)
3. outputs/dataset_report.json (machine-readable comprehensive metrics)
4. outputs/dataset_report.html (executive visual dashboard with KPIs, tables, and samples)
"""

import base64
import html
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image

from src.config.settings import get_settings
from src.data.dataset_scanner import DatasetScanResult, DATASET_PURPOSES
from src.data.dataset_statistics import DatasetStatistics, DatasetStatisticsSummary


class DatasetReportGenerator:
    """Compiles and exports all Phase 1 dataset manifests and reports."""

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        outputs_dir: Optional[Path] = None
    ):
        settings = get_settings()
        self.settings = settings
        self.processed_dir = Path(processed_dir).resolve() if processed_dir else settings.datasets.processed_data_root
        self.outputs_dir = Path(outputs_dir).resolve() if outputs_dir else settings.root_dir / "outputs"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _image_to_base64(path: Path, max_dim: int = 240) -> Optional[str]:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((max_dim, max_dim))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return None

    @staticmethod
    def _mask_to_base64(path: Path, max_dim: int = 240) -> Optional[str]:
        try:
            from PIL import ImageOps
            with Image.open(path) as mask:
                m = mask.convert("L").resize((max_dim, max_dim), Image.Resampling.NEAREST)
                # Colorize from black to vibrant emerald/cyan
                colorized = ImageOps.colorize(m, black=(15, 23, 42), white=(16, 185, 129))
                buf = io.BytesIO()
                colorized.save(buf, format="PNG")
                return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return None

    @staticmethod
    def _overlay_to_base64(img_path: Path, mask_path: Path, max_dim: int = 240) -> Optional[str]:
        try:
            from PIL import ImageOps
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                im = img.convert("RGB").resize((max_dim, max_dim))
                m = mask.convert("L").resize((max_dim, max_dim), Image.Resampling.NEAREST)
                colorized = ImageOps.colorize(m, black=(0, 0, 0), white=(239, 68, 68))
                blended = Image.blend(im, colorized, alpha=0.45)
                buf = io.BytesIO()
                blended.save(buf, format="JPEG", quality=85)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # 1. DATASET PURPOSE MAPPING
    # -------------------------------------------------------------------------
    def save_dataset_purpose_json(self) -> Path:
        """
        Creates data/processed/dataset_purpose.json with canonical module mappings.
        """
        purpose_file = self.processed_dir / "dataset_purpose.json"
        with open(purpose_file, "w", encoding="utf-8") as f:
            json.dump(DATASET_PURPOSES, f, indent=2)
        return purpose_file

    # -------------------------------------------------------------------------
    # 2. DATASET MANIFEST
    # -------------------------------------------------------------------------
    def save_dataset_manifest_json(
        self,
        scan_results: Dict[str, DatasetScanResult]
    ) -> Path:
        """
        Creates data/processed/dataset_manifest.json containing inventory records
        for all target datasets.
        """
        manifest_records = []
        for name, scan in scan_results.items():
            # Determine annotation type
            ann_type = "none"
            if scan.mask_available and scan.bbox_available:
                ann_type = "segmentation_mask_and_bounding_box"
            elif scan.mask_available:
                ann_type = "segmentation_mask"
            elif scan.bbox_available:
                ann_type = "bounding_box"
            elif scan.classes_count > 0:
                ann_type = "classification_folder"

            train_split = scan.splits_detected.get("Training", scan.splits_detected.get("train", 0))
            val_split = scan.splits_detected.get("val", scan.splits_detected.get("validation", 0))
            test_split = scan.splits_detected.get("Test", scan.splits_detected.get("test", 0))

            # Format path as relative project path
            try:
                rel_path = scan.path.relative_to(self.project_root).as_posix()
            except Exception:
                rel_path = f"data/raw/{scan.name}"

            manifest_records.append({
                "name": scan.name,
                "path": rel_path,
                "status": scan.status,
                "image_count": scan.image_count,
                "number_of_images": scan.image_count,
                "class_count": scan.classes_count,
                "number_of_classes": scan.classes_count,
                "class_names": scan.class_names[:50],  # Top 50 classes for summary
                "image_formats": list(scan.image_extensions.keys()),
                "image_dimensions": scan.image_dimensions,
                "annotation_type": ann_type,
                "mask_available": scan.mask_available,
                "mask_availability": scan.mask_available,
                "bbox_available": scan.bbox_available,
                "bbox_availability": scan.bbox_available,
                "official_train_split": train_split,
                "official_validation_split": val_split,
                "official_test_split": test_split,
                "official_splits": scan.official_splits,
                "validation_status": scan.validation_status,
                "warnings": scan.warnings,
                "intended_project_module": scan.intended_modules[0] if scan.intended_modules else "general"
            })

        manifest_file = self.processed_dir / "dataset_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump({
                "system": "NutriVision AI",
                "phase": 1,
                "total_datasets_tracked": len(manifest_records),
                "datasets": manifest_records
            }, f, indent=2)

        return manifest_file

    # -------------------------------------------------------------------------
    # 3. JSON REPORT
    # -------------------------------------------------------------------------
    def save_dataset_report_json(
        self,
        scan_results: Dict[str, DatasetScanResult],
        stats_summaries: Dict[str, DatasetStatisticsSummary],
        validation_summary: Dict[str, Any],
        overlap_analysis: Dict[str, Any],
        duplicate_results: List[Dict[str, Any]]
    ) -> Path:
        """
        Saves outputs/dataset_report.json covering all 12 analytical sections.
        """
        report_data = {
            "title": "NutriVision AI Phase 1 Dataset Inspection & Preparation Report",
            "datasets_summary": {name: scan.to_dict() for name, scan in scan_results.items()},
            "statistics": {name: stat.to_dict() for name, stat in stats_summaries.items()},
            "validation_summary": validation_summary,
            "cross_dataset_overlap": overlap_analysis,
            "duplicate_candidates_count": len(duplicate_results),
            "duplicates_sample": duplicate_results[:30],
            "recommended_dataset_usage": DATASET_PURPOSES
        }

        report_file = self.outputs_dir / "dataset_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        return report_file

    # -------------------------------------------------------------------------
    # 4. EXECUTIVE HTML REPORT
    # -------------------------------------------------------------------------
    def generate_html_report(
        self,
        scan_results: Dict[str, DatasetScanResult],
        stats_summaries: Dict[str, DatasetStatisticsSummary],
        validation_summary: Dict[str, Any],
        overlap_analysis: Dict[str, Any],
        duplicate_results: List[Dict[str, Any]]
    ) -> Path:
        """
        Generates a premium, responsive outputs/dataset_report.html dashboard.
        """
        total_images = sum(s.image_count for s in scan_results.values())
        available_count = sum(1 for s in scan_results.values() if s.is_available)
        total_tracked = len(scan_results)

        # HTML construction with embedded styling
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NutriVision AI — Phase 1 Dataset Inspection Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: rgba(23, 32, 54, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #10b981;
      --accent-glow: rgba(16, 185, 129, 0.25);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --warn: #f59e0b;
      --fail: #ef4444;
      --pass: #10b981;
      --info: #3b82f6;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at top right, #1e293b 0%, #0b0f19 70%);
      color: var(--text);
      min-height: 100vh;
      padding: 2.5rem 1.5rem;
      line-height: 1.6;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 2.5rem;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 1.5rem;
    }}
    .badge-phase {{
      display: inline-block;
      background: var(--accent-glow);
      color: var(--accent);
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      border: 1px solid var(--accent);
      margin-bottom: 0.75rem;
    }}
    h1 {{
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, #ffffff 40%, var(--accent) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 1.05rem;
    }}
    /* Metric Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2.5rem;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
      position: relative;
      overflow: hidden;
    }}
    .kpi-title {{
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.5rem;
    }}
    .kpi-value {{
      font-size: 2.2rem;
      font-weight: 800;
      color: #fff;
    }}
    .kpi-meta {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }}
    /* Sections */
    .section {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 2rem;
      margin-bottom: 2rem;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }}
    .section-title {{
      font-size: 1.4rem;
      font-weight: 700;
      margin-bottom: 1.25rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    /* Badges */
    .badge {{
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.2rem 0.6rem;
      border-radius: 6px;
      text-transform: uppercase;
    }}
    .badge-avail {{ background: rgba(16, 185, 129, 0.2); color: var(--pass); }}
    .badge-notavail {{ background: rgba(156, 163, 175, 0.15); color: var(--text-muted); }}
    .badge-pass {{ background: rgba(16, 185, 129, 0.2); color: var(--pass); }}
    .badge-warn {{ background: rgba(245, 158, 11, 0.2); color: var(--warn); }}
    .badge-fail {{ background: rgba(239, 68, 68, 0.2); color: var(--fail); }}
    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 0.85rem 1rem;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }}
    th {{
      color: var(--text-muted);
      font-weight: 600;
      background: rgba(255, 255, 255, 0.02);
    }}
    tr:hover td {{
      background: rgba(255, 255, 255, 0.03);
    }}
    .code {{
      font-family: monospace;
      background: rgba(255, 255, 255, 0.06);
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-size: 0.85rem;
    }}
    .alert-box {{
      background: rgba(245, 158, 11, 0.1);
      border-left: 4px solid var(--warn);
      padding: 1rem;
      border-radius: 8px;
      margin: 1rem 0;
      font-size: 0.9rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="badge-phase">Phase 1 &bull; Verification &amp; Normalization</span>
      <h1>NutriVision AI Dataset Inspection &amp; Preparation</h1>
      <p class="subtitle">Complete non-destructive dataset discovery, image validation, class taxonomy normalization, and benchmark manifests.</p>
    </header>

    <!-- KPI Section -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">Tracked Datasets</div>
        <div class="kpi-value">{available_count} <span style="font-size: 1.2rem; color: var(--text-muted);">/ {total_tracked}</span></div>
        <div class="kpi-meta">{available_count} active on local disk</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Total Images Scanned</div>
        <div class="kpi-value">{total_images}</div>
        <div class="kpi-meta">{validation_summary.get('valid_images_count', 0)} verified valid</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Image Corruptions</div>
        <div class="kpi-value" style="color: {'var(--pass)' if validation_summary.get('corrupted_images_count', 0) == 0 else 'var(--fail)'}">{validation_summary.get('corrupted_images_count', 0)}</div>
        <div class="kpi-meta">{validation_summary.get('zero_byte_count', 0)} zero-byte files detected</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Confirmed Class Matches</div>
        <div class="kpi-value" style="color: var(--accent);">{overlap_analysis.get('total_confirmed_matches', 0)}</div>
        <div class="kpi-meta">{overlap_analysis.get('total_possible_matches', 0)} possible token alignments</div>
      </div>
    </div>

    <!-- 1. Dataset Status Inventory -->
    <div class="section">
      <div class="section-title">1. Dataset Inventory &amp; Status</div>
      <table>
        <thead>
          <tr>
            <th>Dataset Name</th>
            <th>Status</th>
            <th>Images</th>
            <th>Classes</th>
            <th>Masks</th>
            <th>BBoxes</th>
            <th>Splits Detected</th>
            <th>Validation</th>
          </tr>
        </thead>
        <tbody>
"""

        for name, s in scan_results.items():
            status_badge = f'<span class="badge badge-avail">AVAILABLE</span>' if s.is_available else f'<span class="badge badge-notavail">NOT AVAILABLE</span>'
            val_cls = "badge-pass" if s.validation_status == "PASS" else ("badge-warn" if s.validation_status == "WARN" else "badge-notavail")
            val_badge = f'<span class="badge {val_cls}">{s.validation_status}</span>'
            splits_str = ", ".join(f"{k}:{v}" for k, v in s.splits_detected.items()) if s.splits_detected else "None"
            masks_str = f"YES ({s.mask_count})" if s.mask_available else "No"
            bbox_str = f"YES ({s.bbox_count})" if s.bbox_available else "No"

            html_content += f"""
          <tr>
            <td><strong>{html.escape(name)}</strong></td>
            <td>{status_badge}</td>
            <td>{s.image_count}</td>
            <td>{s.classes_count}</td>
            <td>{masks_str}</td>
            <td>{bbox_str}</td>
            <td><span class="code">{html.escape(splits_str)}</span></td>
            <td>{val_badge}</td>
          </tr>
"""

        html_content += f"""
        </tbody>
      </table>
    </div>

    <!-- 2. Recommended Dataset Usage -->
    <div class="section">
      <div class="section-title">2. Recommended Dataset Usage &amp; Purpose Mapping</div>
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Intended Pipeline Capabilities</th>
            <th>Preservation Policy</th>
          </tr>
        </thead>
        <tbody>
"""
        for name, purposes in DATASET_PURPOSES.items():
            html_content += f"""
          <tr>
            <td><strong>{html.escape(name)}</strong></td>
            <td>{' &bull; '.join(html.escape(p) for p in purposes)}</td>
            <td><span class="code">Strictly preserved</span></td>
          </tr>
"""

        html_content += f"""
        </tbody>
      </table>
    </div>

    <!-- 3. Cross-Dataset Overlap -->
    <div class="section">
      <div class="section-title">3. Cross-Dataset Overlap (Confirmed Matches)</div>
      <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.75rem;">
        Classes shared semantically across distinct food benchmarks without forcing identical class lists:
      </p>
      <table>
        <thead>
          <tr>
            <th>Dataset A</th>
            <th>Dataset B</th>
            <th>Shared Class (Normalized)</th>
            <th>Original A</th>
            <th>Original B</th>
            <th>Match Status</th>
          </tr>
        </thead>
        <tbody>
"""
        confirmed = overlap_analysis.get("confirmed_matches", [])
        if confirmed:
            for m in confirmed[:25]:
                html_content += f"""
          <tr>
            <td>{html.escape(m['dataset_a'])}</td>
            <td>{html.escape(m['dataset_b'])}</td>
            <td><span class="code">{html.escape(m['shared_class'])}</span></td>
            <td>{html.escape(m['class_a_original'])}</td>
            <td>{html.escape(m['class_b_original'])}</td>
            <td><span class="badge badge-pass">CONFIRMED</span></td>
          </tr>
"""
        else:
            html_content += """<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No cross-dataset overlaps found.</td></tr>"""

        html_content += f"""
        </tbody>
      </table>
    </div>

    <!-- 4. Warnings & Integrity Checklist -->
    <div class="section">
      <div class="section-title">4. Integrity Audit &amp; Warnings</div>
"""
        all_warnings = []
        for name, s in scan_results.items():
            for w in s.warnings:
                all_warnings.append(f"<strong>[{html.escape(name)}]</strong>: {html.escape(w)}")

        if all_warnings:
            for w in all_warnings[:15]:
                html_content += f"""<div class="alert-box">{w}</div>"""
        else:
            html_content += """<p style="color: var(--pass);">No data integrity warnings or corruptions detected.</p>"""

        html_content += """
    <!-- 5. Visual Samples from Downloaded Datasets -->
    <div class="section">
      <div class="section-title">5. Visual Samples from Real Datasets</div>
      <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.5rem;">
        Representative visual samples extracted directly from local dataset assets:
      </p>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem;">
"""
        # Render samples
        # FoodSeg103 sample
        fs_scan = scan_results.get("FoodSeg103")
        if fs_scan and fs_scan.image_paths and fs_scan.mask_paths:
            img_p = fs_scan.image_paths[0]
            # find corresponding mask
            m_p = None
            for cand_m in fs_scan.mask_paths:
                if cand_m.stem == img_p.stem:
                    m_p = cand_m
                    break
            if not m_p:
                m_p = fs_scan.mask_paths[0]

            b_img = self._image_to_base64(img_p)
            b_mask = self._mask_to_base64(m_p)
            b_ovr = self._overlay_to_base64(img_p, m_p)

            if b_img and b_mask and b_ovr:
                html_content += f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--card-border); border-radius: 12px; padding: 1.25rem;">
          <h4 style="margin-bottom: 0.75rem; color: #fff;">FoodSeg103 Segmentation Pair ({html.escape(img_p.name)})</h4>
          <div style="display: flex; gap: 0.5rem; justify-content: space-around; text-align: center; font-size: 0.75rem;">
            <div>
              <img src="{b_img}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 6px; border: 1px solid #444;" /><br/>
              <span style="color: var(--text-muted);">Original</span>
            </div>
            <div>
              <img src="{b_mask}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 6px; border: 1px solid #444;" /><br/>
              <span style="color: var(--text-muted);">Mask</span>
            </div>
            <div>
              <img src="{b_ovr}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 6px; border: 1px solid #444;" /><br/>
              <span style="color: var(--text-muted);">Overlay</span>
            </div>
          </div>
        </div>
"""

        # UECFoodPixComplete sample
        uec_scan = scan_results.get("UECFoodPixComplete")
        if uec_scan and uec_scan.image_paths and uec_scan.mask_paths:
            img_p = uec_scan.image_paths[0]
            m_p = None
            for cand_m in uec_scan.mask_paths:
                if cand_m.stem == img_p.stem:
                    m_p = cand_m
                    break
            if not m_p:
                m_p = uec_scan.mask_paths[0]

            b_img = self._image_to_base64(img_p)
            b_mask = self._mask_to_base64(m_p)
            b_ovr = self._overlay_to_base64(img_p, m_p)

            if b_img and b_mask and b_ovr:
                html_content += f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--card-border); border-radius: 12px; padding: 1.25rem;">
          <h4 style="margin-bottom: 0.75rem; color: #fff;">UECFoodPixComplete Meal Pair ({html.escape(img_p.name)})</h4>
          <div style="display: flex; gap: 0.5rem; justify-content: space-around; text-align: center; font-size: 0.75rem;">
            <div>
              <img src="{b_img}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 6px; border: 1px solid #444;" /><br/>
              <span style="color: var(--text-muted);">Original</span>
            </div>
            <div>
              <img src="{b_mask}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 6px; border: 1px solid #444;" /><br/>
              <span style="color: var(--text-muted);">Mask</span>
            </div>
            <div>
              <img src="{b_ovr}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 6px; border: 1px solid #444;" /><br/>
              <span style="color: var(--text-muted);">Overlay</span>
            </div>
          </div>
        </div>
"""

        # Fruits-360 sample
        f360_scan = scan_results.get("Fruits360_100x100")
        if f360_scan and f360_scan.image_paths:
            samples = f360_scan.image_paths[:4]
            html_content += f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--card-border); border-radius: 12px; padding: 1.25rem;">
          <h4 style="margin-bottom: 0.75rem; color: #fff;">Fruits-360 Produce Classification Samples</h4>
          <div style="display: flex; gap: 0.5rem; justify-content: space-around; text-align: center; font-size: 0.75rem;">
"""
            for sp in samples:
                cname = sp.parent.name
                b_im = self._image_to_base64(sp)
                if b_im:
                    html_content += f"""
            <div>
              <img src="{b_im}" style="width: 72px; height: 72px; object-fit: cover; border-radius: 6px; border: 1px solid #444;" /><br/>
              <span style="color: var(--text-muted); font-size: 0.7rem;">{html.escape(cname[:12])}</span>
            </div>
"""
            html_content += """
          </div>
        </div>
"""

        html_content += """
      </div>
    </div>
  </div>
</body>
</html>
"""
        html_file = self.outputs_dir / "dataset_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_file
