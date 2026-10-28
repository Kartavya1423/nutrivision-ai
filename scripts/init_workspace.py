import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRS = [
    ROOT / "data" / "raw" / "FoodSeg103",
    ROOT / "data" / "raw" / "UECFoodPixComplete",
    ROOT / "data" / "raw" / "Fruits360",
    ROOT / "data" / "raw" / "weight_dataset",
    ROOT / "data" / "raw" / "freshness_dataset",
    ROOT / "data" / "processed",
    ROOT / "data" / "splits",
    ROOT / "models" / "detection",
    ROOT / "models" / "segmentation",
    ROOT / "models" / "classification",
    ROOT / "models" / "depth",
    ROOT / "models" / "freshness",
    ROOT / "models" / "weights",
    ROOT / "src" / "config",
    ROOT / "src" / "detection",
    ROOT / "src" / "segmentation",
    ROOT / "src" / "classification",
    ROOT / "src" / "depth",
    ROOT / "src" / "volume",
    ROOT / "src" / "weight",
    ROOT / "src" / "freshness",
    ROOT / "src" / "nutrition",
    ROOT / "src" / "explainability",
    ROOT / "src" / "pipeline",
    ROOT / "src" / "utils",
    ROOT / "api" / "routes",
    ROOT / "api" / "schemas",
    ROOT / "frontend",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "outputs" / "detections",
    ROOT / "outputs" / "segmentation",
    ROOT / "outputs" / "crops",
    ROOT / "outputs" / "depth",
    ROOT / "outputs" / "results",
    ROOT / "outputs" / "visualizations",
    ROOT / "logs",
]

def main():
    created = 0
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        gitkeep = d / ".gitkeep"
        if not any(d.iterdir()):
            gitkeep.touch(exist_ok=True)
            created += 1
    print(f"Verified {len(DIRS)} directories. Created {created} .gitkeep files.")

if __name__ == "__main__":
    main()
