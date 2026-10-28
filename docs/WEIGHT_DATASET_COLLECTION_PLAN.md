# NutriVision AI — Physical Weight & RGB-D Dataset Collection Protocol

**Document Version:** 1.0.0  
**Phase:** Step 2 — Production Dataset & Training Environment Preparation  
**System Target:** NutriVision AI Physical Food Mass Regression Engine  
**Last Updated:** 2026-10-05  

---

## 1. Executive Mandate: Why a New Dataset is Required

**Crucial Production Fact:** None of the standard computer vision food datasets (Food-101, FoodSeg103, UECFoodPix, Fruits-360) provide measured ground-truth scale weight in grams. 
* Treating generic USDA serving sizes (e.g. "1 medium apple = 182g") as ground truth creates massive real-world prediction errors ($>50\%$).
* Computing weight purely from geometric volume $	imes$ constant density fails when food composition, air pockets, cooking method, or sauce saturation vary.

To achieve production-grade weight estimation ($<15\%$ Mean Absolute Percentage Error), a **calibrated real-world RGB-D dataset** must be systematically acquired.

---

## 2. Sensor & Physical Acquisition Setup

| Component | Minimum Specification | Recommended Production Choice | Purpose |
| :--- | :--- | :--- | :--- |
| **RGB Camera** | 1080p ($1920 	imes 1080$), auto-exposure lock | Smartphone (iPhone 13+ / Samsung S22+) or DSLR | Visual feature extraction |
| **Depth Sensor** | Active stereo or LiDAR sensor | Intel RealSense D435i or Apple iPhone LiDAR | Metric depth capture ($Z$ in mm) |
| **Digital Scale** | $0.1	ext{ g}$ precision, tare function | Kitchen food scale with digital readout | Physical ground-truth weight ($W_{	ext{true}}$) |
| **Reference Marker** | Known dimensions ($85.6 	imes 53.98	ext{ mm}$) | Calibration card or ArUco Marker #42 | Camera pose & metric scale verification |
| **Lighting Setup** | Diffuse 5500K LED, $>80	ext{ CRI}$ | Dual softbox desk lamps | Shadow reduction, specular reflection suppression |

---

## 3. Recommended Food Categories & Class Balance (Phase 1 Acquisition: 30 Foods)

| Tier | Food Categories | Selected Items (30 Classes) | Expected Density ($	ext{g/cm}^3$) |
| :--- | :--- | :--- | :--- |
| **Whole Produce (Uniform)** | Fruits & Vegetables | Apple, Banana, Orange, Mango, Avocado, Tomato, Potato, Cucumber, Carrot | $0.85 - 1.05$ |
| **Carbohydrate Staples** | Rice, Breads, Noodles | Steamed White Rice, Fried Rice, Sliced Bread, Croissant, Spaghetti, Ramen | $0.40 - 0.90$ |
| **Proteins** | Poultry, Meat, Seafood, Eggs | Grilled Chicken Breast, Fried Chicken, Beef Steak, Pork Chop, Boiled Egg, Salmon | $0.95 - 1.15$ |
| **Composite Meals** | Asian & Western Bowls | Chicken & Rice Bowl, Pasta Bolognese, Pizza Slice, Hamburger, Green Salad | $0.50 - 0.95$ |

---

## 4. Multi-Angle & Portion Protocol

For every single food item:
1. **Tare the Digital Scale:** Zero the balance with the empty plate.
2. **Weigh the Portion:** Record the exact mass $W_{	ext{true}}$ to $0.1	ext{ g}$.
3. **Capture 4 Angles:**
   * **Angle 1 (Top-Down / Nadir):** $90^\circ$ orthogonal view at $45	ext{ cm}$ distance.
   * **Angle 2 (High Angle / Dining View):** $45^\circ$ elevation from table edge.
   * **Angle 3 (Low Oblique):** $30^\circ$ elevation to capture vertical profile.
   * **Angle 4 (Rotated $90^\circ$ Dining View):** $45^\circ$ elevation opposite side.
4. **Capture 3 Portion Sizes per Food:** Small (50%), Standard (100%), Large (150%).
5. **Multi-Item Plate Scans:** Plates with 2 or 3 distinct food items to capture real dining scenarios.

**Total Planned Acquisition Footprint:**
$$30	ext{ foods} 	imes 3	ext{ portions} 	imes 4	ext{ angles} = 360	ext{ single-food scans} + 120	ext{ multi-food plates} = 480	ext{ RGB-D pairs}$$

---

## 5. Storage Directory Structure & Metadata Schema

```
data/raw/weight_dataset/
├── metadata/
│   └── dataset_index.json
├── rgb/
│   └── scan_0001_apple_small_45deg.jpg
├── depth/
│   └── scan_0001_apple_small_45deg.npy   # float32 array in meters
├── masks/
│   └── scan_0001_apple_small_45deg.png   # binary instance mask
└── calibration/
    └── intrinsics_iphone14_wide.json
```

### Metadata JSON Schema per Sample:
```json
{
  "sample_id": "scan_0001",
  "food_class_id": 1,
  "food_class_name": "apple",
  "portion_category": "small",
  "actual_weight_grams": 134.6,
  "elevation_angle_degrees": 45,
  "camera_distance_cm": 42.5,
  "reference_marker_detected": true,
  "metric_scale_factor": 1.002,
  "camera_model": "iPhone 14 Pro Wide LiDAR",
  "intrinsics": {
    "fx": 1542.4,
    "fy": 1541.9,
    "cx": 962.1,
    "cy": 541.3
  }
}
```

---

## 6. Train, Validation, and Test Splitting Strategy

* **Train Partition (70%):** 21 food classes, multiple portion sizes.
* **Validation Partition (15%):** Unseen portion sizes of the 21 training food classes.
* **Test Partition (15%):** 9 completely **unseen food classes** to evaluate out-of-distribution volumetric generalization.
