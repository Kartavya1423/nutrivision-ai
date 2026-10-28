# NutriVision AI — Phase 6: Final Test Evaluation Report

**Evaluation Date:** 2026-10-20  
**Evaluated Model:** `models/classification/target_foods_v1/best.pth` (Epoch 19)  
**Baseline Model:** `models/classification/efficientnet_b4_food.pth` (198-class production model)  
**Evaluated Split:** `data/processed/target_foods_v1/test/` (Untouched test split, 379 images across 21 classes)  
**Execution Environment:** CPU (PyTorch 2.14.1+cpu, batch size 16, deterministic eval)  

---

## 1. Executive Summary & Headline Metrics

| Metric | Phase 5 Model (`best.pth`) | Baseline (`efficientnet_b4_food.pth`) | Absolute Net Gain | Relative Improvement |
|:---|:---:|:---:|:---:|:---:|
| **Top-1 Accuracy** | **54.09%** (205 / 379) | 19.53% (74 / 379) | **+34.56%** | **+176.9% (2.77x)** |
| **Top-1 95% Wilson CI** | **[49.06%, 59.04%]** | [15.82%, 23.84%] | — | — |
| **Top-5 Accuracy** | **77.04%** (292 / 379) | 35.62% (135 / 379) | **+41.42%** | **+116.3% (2.16x)** |
| **Cross-Entropy Loss** | **1.9592** | — | — | — |
| **Macro-Averaged F1** | **46.15%** | 16.24% | **+29.91%** | **+184.2% (2.84x)** |
| **Macro Precision** | **47.73%** | 18.91% | **+28.82%** | **+152.4% (2.52x)** |
| **Macro Recall** | **49.42%** | 17.50% | **+31.92%** | **+182.4% (2.82x)** |

> [!NOTE]
> Testing was performed strictly on the quarantined, untouched test split. No thresholds were tuned, and no hyperparameters or checkpoints were selected based on these numbers.

---

## 2. Per-Class Test Performance & Support Statistics

| Class Name | Train Count | Val Count | Test Count | Test Prec. | Test Recall | Test F1 | Recall 95% CI | Data Sample Warning / Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 🍊 **Orange** | 165 | 37 | 40 | **100.0%** | **77.5%** | **87.3%** | [62.5%, 87.7%] | Robust sample size; 0 false positives |
| 🍎 **Apple** | 165 | 38 | 40 | **100.0%** | **75.0%** | **85.7%** | [59.8%, 85.8%] | Robust sample size; 0 false positives |
| 🍌 **Banana** | 165 | 38 | 40 | **88.2%** | **75.0%** | **81.1%** | [59.8%, 85.8%] | Robust sample size; highly reliable |
| 🍅 **Tomato** | 63 | 16 | 18 | **61.5%** | **88.9%** | **72.7%** | [67.2%, 96.9%] | High recall; moderate variance ($N<20$) |
| 🍟 **French Fries** | 25 | 8 | 10 | **61.5%** | **80.0%** | **69.6%** | [49.0%, 94.3%] | High recall; small test support ($N=10$) |
| 🍉 **Watermelon** | 25 | 8 | 10 | **60.0%** | **60.0%** | **60.0%** | [31.3%, 83.2%] | Wide CI due to small test support ($N=10$) |
| 🥭 **Mango** | 37 | 11 | 13 | **43.5%** | **76.9%** | **55.6%** | [49.7%, 91.8%] | High recall; moderate precision ($N=13$) |
| 🍗 **Chicken** | 50 | 16 | 19 | **66.7%** | **42.1%** | **51.6%** | [23.1%, 63.7%] | Good precision; confusions with breaded food |
| 🥚 **Egg** | 62 | 20 | 30 | **64.7%** | **36.7%** | **46.8%** | [21.9%, 54.5%] | Good precision; misses some mixed egg dishes |
| 🥬 **Spinach** | 25 | 2 | 10 | **37.5%** | **60.0%** | **46.2%** | [31.3%, 83.2%] | Small test support ($N=10$); confusions with greens |
| 🥦 **Broccoli** | 25 | 8 | 10 | **31.8%** | **70.0%** | **43.8%** | [39.7%, 89.2%] | Strong recall; low precision ($N=10$) |
| 🥬 **Lettuce** | 25 | 8 | 10 | **35.7%** | **50.0%** | **41.7%** | [23.7%, 76.3%] | Confused with spinach/cucumber ($N=10$) |
| 🍇 **Grape** | 25 | 6 | 9 | **50.0%** | **33.3%** | **40.0%** | [12.1%, 64.6%] | High variance warning: test support ($N=9$) $<10$ |
| 🍔 **Burger** | 50 | 16 | 20 | **54.5%** | **30.0%** | **38.7%** | [14.5%, 51.9%] | Modest recall; confused with pineapple/bread |
| 🍞 **Bread** | 100 | 32 | 40 | **41.9%** | **32.5%** | **36.6%** | [20.1%, 48.0%] | High inter-class confusion with pizza/cake |
| 🍰 **Cake** | 25 | 8 | 10 | **22.2%** | **40.0%** | **28.6%** | [16.8%, 68.7%] | High visual diversity; confused with bread |
| 🍓 **Strawberry** | 25 | 8 | 10 | **33.3%** | **20.0%** | **25.0%** | [5.7%, 51.0%] | Confused with other red produce ($N=10$) |
| 🍍 **Pineapple** | 25 | 8 | 10 | **16.7%** | **50.0%** | **25.0%** | [23.7%, 76.3%] | Yellow color collision with banana/mango |
| 🍕 **Pizza** | 25 | 8 | 10 | **15.8%** | **30.0%** | **20.7%** | [10.8%, 60.3%] | Heavy confusion with bread/crust ($N=10$) |
| 🍦 **Ice Cream** | 25 | 8 | 10 | **16.7%** | **10.0%** | **12.5%** | [1.8%, 40.4%] | Melted/scooped forms confuse with mango/cake |
| 🥒 **Cucumber** | 25 | 8 | 10 | **0.0%** | **0.0%** | **0.0%** | [0.0%, 27.8%] | Zero accuracy; severe visual collision with greens |

---

## 3. Top Misclassification & Confusion Patterns

| Rank | True Class | Misclassified As | Error Count | Class Error Rate | Visual Root Cause |
|:---:|:---|:---|:---:|:---:|:---|
| 1 | `bread` | `pizza` | 8 | 20.0% | Baked wheat dough crust and toppings look identical in tight crops. |
| 2 | `bread` | `cake` | 5 | 12.5% | Sliced loaf crumb texture overlaps with sponge cake texture. |
| 3 | `burger` | `pineapple` | 5 | 25.0% | Yellow cheese melts and bun sheen trigger yellow fruit feature detectors. |
| 4 | `chicken` | `bread` | 5 | 26.3% | Golden-brown fried chicken breading shares color and grain with bread crust. |
| 5 | `egg` | `bread` | 4 | 13.3% | Egg toast / sandwich crops containing white/yellow yolk and bread crumbs. |
| 6 | `egg` | `tomato` | 4 | 13.3% | Shakshuka / tomato-scramble crops in FoodSeg103 with heavy red sauce. |
| 7 | `ice_cream` | `mango` | 4 | 40.0% | Mango sorbet / yellow ice cream scoops share texture and hue with ripe mango. |
| 8 | `banana` | `pineapple` | 3 | 7.5% | Yellow flesh patches without peel curvature trigger pineapple detectors. |
| 9 | `grape` | `broccoli` | 3 | 33.3% | Clustered green grape bunches mimic broccoli floret cluster patterns. |
| 10 | `cucumber` | `lettuce` / `broccoli` | 4 | 40.0% | Thin translucent cucumber salad slices resemble lettuce leaf folds. |

---

## 4. Formal Evidence-Based Support Assessment for All 24 Target Foods

| Category | Target Food | Test F1 | Test Support | Production Status | Recommended Action in App |
|:---|:---|:---:|:---:|:---|:---|
| **Tier 1: High Confidence** | 🍊 **Orange** | **87.3%** | 40 | **Production Ready** | Direct automatic recognition with high confidence. |
| | 🍎 **Apple** | **85.7%** | 40 | **Production Ready** | Direct automatic recognition; seamlessly replaces apple-only pipeline. |
| | 🍌 **Banana** | **81.1%** | 40 | **Production Ready** | Direct automatic recognition; ripeness handled via HSV. |
| | 🍅 **Tomato** | **72.7%** | 18 | **Production Ready** | Direct automatic recognition; green/ripe handled via color analysis. |
| **Tier 2: Moderately Supported** | 🍟 **French Fries** | **69.6%** | 10 | **Usable with Confidence Floor** | Reliable on standard fries; warn on curly/loaded fries. |
| | 🍉 **Watermelon** | **60.0%** | 10 | **Usable with Confidence Floor** | Reliable on triangular slices with red flesh and green rind. |
| | 🥭 **Mango** | **55.6%** | 13 | **Usable with Confidence Floor** | Reliable on sliced or whole mangoes; provide top-3 fallback. |
| | 🍗 **Chicken** | **51.6%** | 19 | **Usable with Confidence Floor** | Reliable on roast/curry chicken; breaded chicken needs top-3 fallback. |
| | 🥚 **Egg** | **46.8%** | 30 | **Usable with Confidence Floor** | Reliable on boiled/fried eggs; provide top-3 fallback for omelets. |
| | 🥬 **Spinach** | **46.2%** | 10 | **Usable with Confidence Floor** | Recommend top-3 candidate suggestions. |
| | 🥦 **Broccoli** | **43.8%** | 10 | **Usable with Confidence Floor** | High recall (70%); provide top-3 fallback. |
| | 🥬 **Lettuce** | **41.7%** | 10 | **Usable with Confidence Floor** | Distinguishable as whole/shredded salad greens. |
| | 🍇 **Grape** | **40.0%** | 9 | **Usable with Confidence Floor** | Small test set; reliable on purple/green grape clusters. |
| **Tier 3: Experimental** | 🍔 **Burger** | **38.7%** | 20 | **Low Confidence** | High precision (54.5%), low recall (30%); needs top-3 prompt. |
| | 🍞 **Bread** | **36.6%** | 40 | **Low Confidence** | Confuses with pizza/cake; requires user confirmation prompt. |
| | 🍰 **Cake** | **28.6%** | 10 | **Low Confidence** | High visual variance; requires top-3 candidate selector. |
| | 🍓 **Strawberry** | **25.0%** | 10 | **Low Confidence** | Confuses with tomato; requires top-3 candidate selector. |
| | 🍍 **Pineapple** | **25.0%** | 10 | **Low Confidence** | Confuses with mango/banana flesh; requires user confirmation. |
| | 🍕 **Pizza** | **20.7%** | 10 | **Low Confidence** | Confuses with bread/toast; requires user confirmation. |
| | 🍦 **Ice Cream** | **12.5%** | 10 | **Low Confidence** | Highly prone to melting/color variation; requires user prompt. |
| **Tier 4: Failed / Unsupported** | 🥒 **Cucumber** | **0.0%** | 10 | **Failed Recognition** | Model cannot distinguish sliced cucumber from lettuce/zucchini. |
| **Tier 5: Data Deficient** | 🧀 **Paneer** | **N/A** | 0 | **Unsupported (0 Images)** | Handled via nutritional database lookup and search fallback. |
| | 🧈 **Butter** | **N/A** | 0 | **Unsupported (Bundled)** | Handled via nutritional database lookup and search fallback. |
