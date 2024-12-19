
import requests
import csv
import random

def search_serpapi(query, api_key):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "google",  
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("organic_results", [])
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

def save_results_to_csv(results, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Query", "Title", "URL"])  
        for query, query_results in results.items():
            for result in query_results:
                writer.writerow([query, result['title'], result['link']])

def generate_random_queries(word_list, num_queries=5):
    """random queries from a list of words."""
    queries = []
    for _ in range(num_queries):
        query = " ".join(random.sample(word_list, random.randint(2, 3)))
        queries.append(query)
    return queries


word_list = ["robotics", "automation", "shoes", "cheap", "cancer", "treatment", 
             "technology", "food", "health", "education", "travel", "fashion"]

queries = generate_random_queries(word_list, num_queries=5)  

all_results = {}
for query in queries:
    print(f"Results for query: {query}")
    results = search_serpapi(query, api_key)
    all_results[query] = results  
    for result in results:
        print(f"Title: {result['title']}\nURL: {result['link']}\n")

save_results_to_csv(all_results, "search_results.csv")
print("Results saved to search_results.csv")
