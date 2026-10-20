import time
import random
import requests

API_URL = "http://localhost:8000/evaluate-risk"

def generate_sample_patient_payload():
    return {
        "age": float(random.randint(30, 75)),
        "sex": random.choice([0, 1]),
        "cp": random.randint(1, 4),
        "trestbps": float(random.randint(100, 180)),
        "chol": float(random.randint(150, 350)),
        "fbs": random.choice([0, 1]),
        "restecg": random.choice([0, 1, 2]),
        "thalach": float(random.randint(90, 190)),
        "exang": random.choice([0, 1]),
        "oldpeak": round(random.uniform(0.0, 4.0), 1),
        "slope": random.randint(1, 3),
        "ca": float(random.randint(0, 3)),
        "thal": float(random.choice([3, 6, 7]))
    }

if __name__ == "__main__":
    print("Sending synthetic load requests to API (Ctrl+C to stop)...")
    while True:
        try:
            payload = generate_sample_patient_payload()
            res = requests.post(API_URL, json=payload, timeout=2)
            print(f"Status: {res.status_code} | Response: {res.json()}")
            time.sleep(random.uniform(0.1, 0.5))
        except Exception as e:
            print(f"Request failed: {e}")
            time.sleep(1)