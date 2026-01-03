import requests

res = requests.post("http://localhost:5000/predict", json={
    "features": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
})

print("Status Code:", res.status_code)
print("Response Text:", res.text)

