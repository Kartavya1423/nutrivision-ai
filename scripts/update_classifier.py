from pathlib import Path

file_path = Path("src/classification/efficientnet_classifier.py")
content = file_path.read_text(encoding="utf-8")

old_check = '''            arch = ckpt.get("architecture", "EfficientNetB4") if isinstance(ckpt, dict) else "EfficientNetB4"
            if arch == "MobileNetV2" or "features.0.0.weight" in state_dict:
                base_model = models.mobilenet_v2(weights=None)
                base_model.classifier = nn.Sequential(
                    nn.Linear(1280, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(128, self.num_classes),
                )
                base_model.load_state_dict(state_dict)
            else:
                base_model = models.efficientnet_b4(weights=None)
                in_features = base_model.classifier[1].in_features
                base_model.classifier[1] = nn.Linear(in_features, self.num_classes)
                base_model.load_state_dict(state_dict)'''

new_check = '''            arch_str = str(ckpt.get("architecture", "") if isinstance(ckpt, dict) else "")
            is_mobilenet = "mobilenet" in arch_str.lower() or "mobilenet" in str(resolved_path).lower()
            if is_mobilenet:
                base_model = models.mobilenet_v2(weights=None)
                base_model.classifier = nn.Sequential(
                    nn.Linear(1280, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(128, self.num_classes),
                )
                base_model.load_state_dict(state_dict)
            else:
                base_model = models.efficientnet_b4(weights=None)
                in_features = base_model.classifier[1].in_features
                base_model.classifier[1] = nn.Linear(in_features, self.num_classes)
                base_model.load_state_dict(state_dict)'''

if old_check in content:
    content = content.replace(old_check, new_check, 1)

file_path.write_text(content, encoding="utf-8")
print("Updated efficientnet_classifier.py with robust architecture check!")
