import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Sample wine data features
# Format: [alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity,hue, od280/od315_of_diluted_wines, proline]
wine_samples = [
    # Sample 1
    [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065],
    # Sample 2
    [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050],
    # Sample 3
    [13.16, 2.36, 2.67, 18.6, 101, 2.8, 3.24, 0.3, 2.81, 5.68, 1.03, 3.17, 1185]
]

# Prepare the request payload
payload = {
    "features": wine_samples
}

# Send POST request to the API
print("Sending prediction request for wine classification...")
try:
    response = requests.post(url, json=payload)
    
    # Print the response
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Predictions:")
        print(json.dumps(result, indent=2))
        
        if "Predictions" in result:
            print("\nClass Outputs:")
            for i, pred in enumerate(result["Predictions"]):
                print(f"Sample {i+1}: Class {pred}")
    else:
        print(f"Error: {response.text}")
except requests.exceptions.ConnectionError:
    print("Failed to connect to the server. Make sure the API is running at", url) 