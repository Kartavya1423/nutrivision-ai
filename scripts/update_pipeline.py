from pathlib import Path

pipeline_path = Path("src/pipeline/nutrivision_pipeline.py")
content = pipeline_path.read_text(encoding="utf-8")

old_weight_section = """            t_wgt0 = time.perf_counter()
            try:
                w_res = self.weight_estimator.estimate_weight(class_name=cls_label, volume_output=vol_res)
                if w_res.is_success and w_res.estimated_weight_grams is not None:
                    weight_val = round(float(w_res.estimated_weight_grams), 2)
                    weight_status = "SUCCESS"
                else:
                    weight_status = w_res.status.value if hasattr(w_res.status, "value") else str(w_res.status)
            except Exception as e:
                weight_status = "WEIGHT_FAILED"
                errors.append(f"Weight estimation error for item {idx}: {str(e)}")

            # Contour-based 2D Weight Estimation (Research Paper Section III-D)
            # When depth scale is uncalibrated or volume unavailable, use empirical contour weight
            is_uncalibrated = (vol_res is None or not vol_res.is_success or 
                               getattr(vol_res, "calibration_status", "") == "VOLUME_SCALE_NOT_CALIBRATED" or
                               getattr(vol_res, "status", "") == "CALIBRATION_REQUIRED")

            if is_uncalibrated or weight_val is None or weight_val <= 0:
                try:
                    cw_res = self.contour_weight_estimator.estimate_weight(
                        image=np_img,
                        food_name=cls_label,
                        bbox=[x1, y1, x2, y2],
                        mask=mask_obj,
                    )
                    if cw_res and cw_res.estimated_weight_g > 0:
                        weight_val = round(float(cw_res.estimated_weight_g), 2)
                        weight_status = "SUCCESS"
                except Exception as e:
                    errors.append(f"Contour weight warning for item {idx}: {str(e)}")
            timings["weight_ms"] += (time.perf_counter() - t_wgt0) * 1000.0"""

new_weight_section = """            t_wgt0 = time.perf_counter()
            # Contour-based 2D Weight Estimation as primary approach (Research Paper Section III-D)
            # Formula: We = Wb * (0.6 + 1.2 * Ar)
            try:
                cw_res = self.contour_weight_estimator.estimate_weight(
                    image=np_img,
                    food_name=cls_label,
                    bbox=[x1, y1, x2, y2],
                    mask=mask_obj,
                )
                if cw_res and cw_res.estimated_weight_g > 0:
                    weight_val = round(float(cw_res.estimated_weight_g), 2)
                    weight_status = "SUCCESS"
            except Exception as e:
                errors.append(f"Contour weight error for item {idx}: {str(e)}")

            # Volumetric fallback if contour weight unavailable
            if weight_val is None or weight_val <= 0:
                try:
                    w_res = self.weight_estimator.estimate_weight(class_name=cls_label, volume_output=vol_res)
                    if w_res.is_success and w_res.estimated_weight_grams is not None:
                        weight_val = round(float(w_res.estimated_weight_grams), 2)
                        weight_status = "SUCCESS"
                    else:
                        weight_status = w_res.status.value if hasattr(w_res.status, "value") else str(w_res.status)
                except Exception as e:
                    weight_status = "WEIGHT_FAILED"
                    errors.append(f"Weight estimation error for item {idx}: {str(e)}")
            timings["weight_ms"] += (time.perf_counter() - t_wgt0) * 1000.0"""

if old_weight_section in content:
    content = content.replace(old_weight_section, new_weight_section, 1)

pipeline_path.write_text(content, encoding="utf-8")
print("Updated pipeline with Research Paper Primary Contour Weight Estimation!")
