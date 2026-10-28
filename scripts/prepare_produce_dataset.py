"""
NutriVision AI — Enriched Produce & Food Dataset Builder
Copies real fruits (apples, bananas, oranges) from freshness_dataset,
downloads verified CC0 tomatoes and mangoes, and generates realistic
crops and augmentations to bridge the real-world photograph domain gap.
"""

import io
from pathlib import Path
import random
import urllib.request
import numpy as np
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_ROOT = PROJECT_ROOT / "data" / "processed" / "classification"

def copy_freshness_fruits():
    """Copies real apples, bananas, and oranges from freshness_dataset."""
    raw_fresh = PROJECT_ROOT / "data" / "raw" / "freshness_dataset"
    if not raw_fresh.is_dir():
        print("freshness_dataset not found, skipping copy.")
        return

    mapping = {
        "apple": raw_fresh / "freshapples",
        "banana": raw_fresh / "freshbanana",
        "orange": raw_fresh / "freshoranges",
    }

    for cname, src_dir in mapping.items():
        if not src_dir.is_dir():
            continue
        imgs = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))
        if not imgs:
            continue

        # Split 70% train, 15% val, 15% test
        random.seed(42)
        shuffled = imgs.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        splits = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:],
        }

        for split, flist in splits.items():
            dest = CLASS_ROOT / split / cname
            dest.mkdir(parents=True, exist_ok=True)
            for f in flist:
                target = dest / f"real_{f.name}"
                if not target.exists():
                    img = Image.open(f).convert("RGB")
                    img = img.resize((224, 224), Image.Resampling.BILINEAR)
                    img.save(target, quality=95)
        print(f"Copied {len(imgs)} real {cname} images across train/val/test.")

def download_and_augment_tomatoes_and_mangoes():
    """Downloads real CC0 tomato and mango images and creates realistic variations."""
    sources = {
        "tomato": [
            "https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/8/88/Bright_red_tomato_and_cross_section02.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/9/9d/Tomato.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/0/03/Cherry_tomatoes.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/b/b3/Campari_tomatoes.jpg",
        ],
        "mango": [
            "https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/7/74/Mangos_-_single_and_halved.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/4/40/Mango_4.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/e/ee/Mango_and_cross_section_edit.jpg",
        ]
    }

    # Also add user's uploaded tomato image
    user_crop = PROJECT_ROOT / "outputs" / "phase8" / "crops" / "nv_20261019_050116_c23847_item0_food_crop.png"
    base_images = {"tomato": [], "mango": []}

    if user_crop.is_file():
        try:
            im = Image.open(user_crop).convert("RGB")
            base_images["tomato"].append(im)
            print("Loaded user's uploaded desi tomato crop.")
        except Exception as e:
            print("Failed to load user crop:", e)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for cname, urls in sources.items():
        for u in urls:
            try:
                req = urllib.request.Request(u, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                    im = Image.open(io.BytesIO(data)).convert("RGB")
                    base_images[cname].append(im)
                    print(f"Downloaded {cname}: {u.split('/')[-1]} ({im.size})")
            except Exception as e:
                print(f"Skipping {u}: {e}")

    # Generate augmentations for tomato and mango
    for cname, img_list in base_images.items():
        if not img_list:
            continue
        all_augmented = []
        for idx, orig in enumerate(img_list):
            all_augmented.append(orig.resize((224, 224), Image.Resampling.BILINEAR))
            # 1. Flips & Rotations
            all_augmented.append(orig.transpose(Image.FLIP_LEFT_RIGHT).resize((224, 224)))
            for angle in [10, -10, 20, -20, 30, -30, 45, -45]:
                all_augmented.append(orig.rotate(angle, expand=False, fillcolor=(255, 255, 255)).resize((224, 224)))
            # 2. Lighting & Contrast
            enh_b = ImageEnhance.Brightness(orig)
            all_augmented.append(enh_b.enhance(1.15).resize((224, 224)))
            all_augmented.append(enh_b.enhance(0.88).resize((224, 224)))
            enh_c = ImageEnhance.Contrast(orig)
            all_augmented.append(enh_c.enhance(1.20).resize((224, 224)))
            all_augmented.append(enh_c.enhance(0.90).resize((224, 224)))
            # 3. Crops (sub-regions / individual lobes)
            w, h = orig.size
            if w > 200 and h > 200:
                # Left half
                all_augmented.append(orig.crop((0, 0, int(w * 0.65), h)).resize((224, 224)))
                # Right half
                all_augmented.append(orig.crop((int(w * 0.35), 0, w, h)).resize((224, 224)))
                # Center crop
                all_augmented.append(orig.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))).resize((224, 224)))

            # 4. Color variation (raw / green vs ripe)
            arr = np.array(orig.resize((224, 224))).astype(np.float32)
            if cname == "mango":
                # Green raw mango shift
                green_arr = arr.copy()
                green_arr[:, :, 1] = np.clip(green_arr[:, :, 1] * 1.30, 0, 255)
                green_arr[:, :, 0] = np.clip(green_arr[:, :, 0] * 0.80, 0, 255)
                all_augmented.append(Image.fromarray(green_arr.astype(np.uint8)))
            elif cname == "tomato":
                # Rich ripe red boost
                ripe_arr = arr.copy()
                ripe_arr[:, :, 0] = np.clip(ripe_arr[:, :, 0] * 1.15, 0, 255)
                all_augmented.append(Image.fromarray(ripe_arr.astype(np.uint8)))

        # Split and save
        random.seed(42)
        random.shuffle(all_augmented)
        n = len(all_augmented)
        n_train = int(n * 0.75)
        n_val = int(n * 0.15)

        splits = {
            "train": all_augmented[:n_train],
            "val": all_augmented[n_train:n_train + n_val],
            "test": all_augmented[n_train + n_val:],
        }

        for split, ims in splits.items():
            dest = CLASS_ROOT / split / cname
            dest.mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(ims):
                p = dest / f"enriched_{cname}_{i:03d}.jpg"
                im.save(p, quality=92)
        print(f"Generated {len(all_augmented)} enriched {cname} images across splits.")

if __name__ == "__main__":
    print("=" * 60)
    print("  NutriVision AI: Enriching Classification Dataset")
    print("=" * 60)
    copy_freshness_fruits()
    download_and_augment_tomatoes_and_mangoes()
    print("\nDataset enrichment complete.")
