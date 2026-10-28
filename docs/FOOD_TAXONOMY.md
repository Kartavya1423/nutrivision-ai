# NutriVision AI — Food Class Taxonomy Specification

**Specification Version:** 2.0.0  
**Phase:** Step 2 — Production Dataset & Training Environment Preparation  
**System Target:** NutriVision AI Real-World Meal Recognition & Nutritional Analysis  
**Last Updated:** 2026-10-05  

---

## 1. Executive Summary

NutriVision AI is designed as a **real-world meal scanning and nutritional estimation engine**, far transcending single-fruit laboratory classification. This document establishes a **198-class unified food taxonomy** grouped into **19 culinary meal categories**.

Every class maps directly to:
1. **YOLO11 Detection:** Bounding boxes from unified food segmentation datasets.
2. **EfficientNet-B4 Classification:** Fine-grained 198-class recognition on cropped food regions.
3. **USDA FoodData Central:** Canonical nutritional profiling (calories, protein, carbs, fat, fiber).
4. **Volumetric Density Profile:** Standard specific gravity (g/cm³) for weight estimation.

---

## 2. The 19 Culinary Categories Breakdown

| Category | Class Count | Representative Items | Culinary Role |
| :--- | :--- | :--- | :--- |
| **fruits** | 22 | Apple, Banana, Orange, Mango, Avocado, Strawberry, Blueberry | Whole produce, raw portions, desserts |
| **vegetables** | 34 | Broccoli, Cabbage, Carrot, Spinach, Potato, Tomato, Corn, Onion | Raw & cooked sides, salad ingredients |
| **rice** | 14 | White Rice, Fried Rice, Beef Bowl, Sushi Bowl, Pilaf, Congee | Meal staples, bowls, carbohydrate base |
| **bread** | 11 | Sliced Bread, Toast, Croissant, Sandwich, Hamburger, Hot Dog | Bakery, handheld composite meals |
| **pasta** | 4 | Spaghetti, Pasta with Meat Sauce, Gratin | Italian & Western noodle entrees |
| **noodles** | 10 | Ramen, Soba, Udon, Chilled Noodles, Yakisoba, Dipping Noodles | Asian noodle bowls, broths, stir-fries |
| **meat** | 13 | Beef Steak, Pork, Sausage, Simmered Pork, Hamburger Steak | Red meat, pork, processed meats |
| **chicken** | 5 | Fried Chicken (Karaage), Roast Chicken, Yakitori, Duck | Poultry portions, grilled & fried |
| **fish** | 16 | Grilled Salmon, Sashimi, Sushi, Fried Fish, Shrimp, Dried Fish | Seafood, white fish, oily fish, shellfish |
| **eggs** | 6 | Fried Egg, Omelet, Boiled Egg, Rolled Omelet, Chawanmushi | Breakfast staples, egg preparations |
| **dairy** | 3 | Cheese & Butter, Whole Milk, Milkshake | Dairy fats, dairy beverages |
| **legumes** | 9 | Tofu, Cold Tofu, Fermented Soybeans (Natto), Green Beans, Red Beans | Plant-based protein, soy preparations |
| **soups** | 7 | Miso Soup, Chinese Soup, Pork Miso Soup, Stew, Potage, Oden | Liquid meal components, broths |
| **curries** | 3 | Beef Curry, Katsu Cutlet Curry, Standard Curry Sauce | Thick sauces, spiced gravies on rice |
| **salads** | 4 | Green Salad, Macaroni Salad, Potato Salad | Cold composite side dishes |
| **snacks** | 15 | French Fries, Almond, Biscuit, Cashew, Popcorn, Dumpling, Gyoza | Finger foods, nuts, appetizers |
| **desserts** | 11 | Cake, Ice Cream, Pudding, Egg Tart, Pancake, Chocolate, Waffle | Sweet baked goods, confectionery |
| **beverages** | 5 | Coffee, Tea, Fruit Juice, Wine, General Beverage | Hydration, liquid calories |
| **grains** | 2 | Pizza, Bibimbap | Multi-grain / composite grain dishes |

*Total Unified Canonical Classes:* **198**

---

## 3. Class Synonyms, Aliases, and Normalization Dictionary

Real-world food scanning requires flexible linguistic normalization to map user voice queries, regional terms, and OCR labels to canonical model IDs:

```json
{
  "french_fries": [
    "fries",
    "chips",
    "french fries",
    "potato fries"
  ],
  "hamburger": [
    "burger",
    "cheeseburger",
    "beef burger"
  ],
  "hot_dog": [
    "hotdog",
    "frankfurter",
    "sausage roll"
  ],
  "ramen_noodle": [
    "ramen",
    "ramen bowl",
    "japanese noodle soup"
  ],
  "soba_noodle": [
    "soba",
    "buckwheat noodle"
  ],
  "udon_noodle": [
    "udon",
    "wheat noodle"
  ],
  "fried_rice": [
    "stir-fried rice",
    "egg fried rice",
    "chahan"
  ],
  "rice": [
    "white rice",
    "steamed rice",
    "boiled rice"
  ],
  "curry": [
    "japanese curry",
    "curry sauce",
    "curry rice"
  ],
  "beef_curry": [
    "beef curry rice"
  ],
  "cutlet_curry": [
    "katsu curry",
    "pork katsu curry"
  ],
  "pizza": [
    "pizza slice",
    "cheese pizza",
    "pepperoni pizza"
  ],
  "spaghetti": [
    "pasta",
    "spaghetti noodles"
  ],
  "spaghetti_meat_sauce": [
    "bolognese",
    "spaghetti bolognese",
    "pasta with meat sauce"
  ],
  "sushi": [
    "nigiri",
    "maki",
    "sushi roll",
    "sushi platter"
  ],
  "sashimi": [
    "raw fish",
    "sliced fish"
  ],
  "dumpling": [
    "potsticker",
    "gyoza"
  ],
  "jiaozi": [
    "chinese dumpling",
    "boiled dumpling"
  ],
  "steamed_meat_dumpling": [
    "shumai",
    "siu mai",
    "dim sum dumpling"
  ],
  "croissant": [
    "crescent roll",
    "pastry"
  ],
  "pancake": [
    "hotcake",
    "flapjacks"
  ],
  "toast": [
    "toasted bread",
    "sliced toast"
  ],
  "miso_soup": [
    "japanese miso soup",
    "fermented soybean soup"
  ],
  "green_salad": [
    "garden salad",
    "tossed salad",
    "leaf salad"
  ],
  "chicken_duck": [
    "poultry",
    "chicken pieces",
    "duck meat"
  ],
  "fried_chicken": [
    "crispy chicken",
    "karaage",
    "chicken tenders"
  ],
  "boiled_egg": [
    "hard boiled egg",
    "soft boiled egg"
  ],
  "egg": [
    "chicken egg",
    "whole egg"
  ],
  "apple": [
    "red apple",
    "green apple"
  ],
  "banana": [
    "ripe banana",
    "yellow banana"
  ],
  "orange": [
    "sweet orange",
    "citrus"
  ],
  "cheese_butter": [
    "cheese",
    "cheddar",
    "butter slice",
    "dairy fat"
  ]
}
```

---

## 4. Unsupported Foods & Expansion Roadmap (Phase 3+)

The following major culinary traditions are scheduled for targeted fine-tuning and few-shot expansion:

| Dish | Target Category | Primary Reason for Current Exclusion | Action Plan |
| :--- | :--- | :--- | :--- |
| **Indian Biryani** | `rice` | Layered rice + spiced meat not in FoodSeg/UEC | Add 500 annotated images via Food-101 / custom scrape |
| **Lentil Dal** | `curries` | Liquid pulse dishes require viscosity estimation | Depth-assisted surface area pooling |
| **Tacos / Burritos** | `bread` | Folded composite wrap obscures internal components | Train multi-label classification on interior cross-section |
| **Middle Eastern Shawarma** | `meat` | Vertically roasted shaved meat texture | Annotate spit-roasted poultry/beef crops |

---

## 5. Extensibility Framework

New food classes can be integrated without redesigning the architecture by providing:
1. `class_id`: Unique integer $\ge 198$.
2. `class_name`: Normalized snake_case identifier.
3. `category`: One of the 19 canonical culinary tiers.
4. `usda_fdc_id`: Corresponding USDA FoodData Central foundation food entry.
5. `density_g_cm3`: Measured specific gravity for weight calculation.
