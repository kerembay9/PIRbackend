import requests
import json

# The URL of the Flask API endpoints
url_generate_dummy_queries = "http://127.0.0.1:5000/generate-dummy-queries"
url_generate_consecutive_queries = "http://127.0.0.1:5000/generate-consecutive-queries"

# Test data for generating dummy queries
data_generate_dummy_queries = {
    "input_query": "how to cook pasta",
    "num_queries": 3
}

# Test data for generating consecutive queries
data_generate_consecutive_queries = {
    "input_query": "how to cook pasta",
    "dummy_queries": [
        {"query": "how to make bread", "category": "cooking"},
        {"query": "how to bake a cake", "category": "cooking"}
    ],
    "input_category": "cooking"
}

# Function to test generate-dummy-queries endpoint
def test_generate_dummy_queries():
    response = requests.post(url_generate_dummy_queries, json=data_generate_dummy_queries)
    if response.status_code == 200:
        print("Dummy Queries Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")

# Function to test generate-consecutive-queries endpoint
def test_generate_consecutive_queries():
    response = requests.post(url_generate_consecutive_queries, json=data_generate_consecutive_queries)
    if response.status_code == 200:
        print("Consecutive Queries Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")

# Run tests
test_generate_dummy_queries()
test_generate_consecutive_queries()
