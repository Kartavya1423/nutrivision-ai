"""
NutriVision AI — Step 6A API Endpoints Live Test Script
Spawns Uvicorn server, sends authentic HTTP requests to all documented routes,
records HTTP status codes and responses, and cleanly shuts down.
"""

import json
from pathlib import Path
import subprocess
import sys
import time
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PORT = 8008  # Use a dedicated port to avoid conflicts
BASE_URL = f"http://127.0.0.1:{PORT}"

def run_live_api_tests():
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]

    print(f"Starting API server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Poll until server is ready
    server_ready = False
    max_wait = 20
    start_wait = time.time()
    while time.time() - start_wait < max_wait:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1.0)
            if r.status_code == 200:
                server_ready = True
                print("Server is UP and HEALTHY!")
                break
        except Exception:
            time.sleep(0.5)

    if not server_ready:
        print("ERROR: Server failed to start within timeout.")
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        print("STDOUT:\n", stdout)
        print("STDERR:\n", stderr)
        sys.exit(1)

    test_results = {}
    endpoints_to_test = [
        ("GET", "/", None),
        ("GET", "/health", None),
        ("GET", "/api/v1/health", None),
        ("GET", "/models/status", None),
        ("GET", "/api/v1/models/status", None),
        ("GET", "/docs", None),
        ("GET", "/openapi.json", None),
    ]

    print("\n=== Testing Standard Discovery & Health Endpoints ===")
    for method, path, data in endpoints_to_test:
        url = f"{BASE_URL}{path}"
        t0 = time.time()
        try:
            resp = requests.get(url, timeout=5.0)
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            content_type = resp.headers.get("content-type", "")
            json_data = resp.json() if "application/json" in content_type else f"[{content_type}]"
            test_results[path] = {
                "method": method,
                "status_code": resp.status_code,
                "latency_ms": elapsed_ms,
                "response": json_data,
            }
            print(f"[{method}] {path:<25} -> Status: {resp.status_code} ({elapsed_ms} ms)")
        except Exception as e:
            test_results[path] = {
                "method": method,
                "status_code": 0,
                "error": str(e),
            }
            print(f"[{method}] {path:<25} -> FAILED: {e}")

    # Test Image Analysis Endpoint with real image
    print("\n=== Testing Real Image Inference via POST /analyze ===")
    img_path = ROOT_DIR / "data/raw/UECFoodPixComplete/data/UECFoodPIXCOMPLETE/train/img/1.jpg"
    result_id = None
    if img_path.is_file():
        t0 = time.time()
        with open(img_path, "rb") as f:
            files = {"file": ("1.jpg", f, "image/jpeg")}
            data = {
                "confidence_threshold": 0.015,
                "enable_gradcam": True,
                "enable_nutrition": True,
            }
            resp = requests.post(f"{BASE_URL}/analyze", files=files, data=data, timeout=30.0)
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            print(f"[POST] /analyze                   -> Status: {resp.status_code} ({elapsed_ms} ms)")
            if resp.status_code == 200:
                anal_res = resp.json()
                result_id = anal_res.get("result_id")
                test_results["/analyze"] = {
                    "method": "POST",
                    "status_code": resp.status_code,
                    "latency_ms": elapsed_ms,
                    "result_id": result_id,
                    "pipeline_status": anal_res.get("status"),
                    "total_foods_detected": anal_res.get("total_foods_detected"),
                    "items": anal_res.get("items"),
                    "meal_summary": anal_res.get("meal_summary"),
                    "errors": anal_res.get("errors"),
                }
                print(f"  Result ID: {result_id}")
                print(f"  Pipeline Status: {anal_res.get('status')}")
                print(f"  Foods Detected: {anal_res.get('total_foods_detected')}")
            else:
                test_results["/analyze"] = {
                    "method": "POST",
                    "status_code": resp.status_code,
                    "error": resp.text,
                }
    else:
        print(f"Image {img_path} not found.")

    # Test Retrieving the Result by ID
    if result_id:
        print(f"\n=== Testing GET /results/{result_id} ===")
        r_url = f"{BASE_URL}/results/{result_id}"
        t0 = time.time()
        r_resp = requests.get(r_url, timeout=5.0)
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        print(f"[GET] /results/{{result_id}}       -> Status: {r_resp.status_code} ({elapsed_ms} ms)")
        test_results[f"/results/{result_id}"] = {
            "method": "GET",
            "status_code": r_resp.status_code,
            "latency_ms": elapsed_ms,
            "match": (r_resp.status_code == 200),
        }

    # Clean shutdown
    print("\nShutting down test API server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Test server shutdown complete.")

    out_file = ROOT_DIR / "outputs/step6a_api_endpoint_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nSaved API test results to: {out_file}")

if __name__ == "__main__":
    run_live_api_tests()
