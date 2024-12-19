import random
import json
import pandas as pd
import requests

from typing import List, Dict
import random
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from openai import OpenAI
# line 20 openai key line 250 other key
class DummyQueryGenerator:
    def __init__(self):
        self.categories = pd.read_csv('data/google_trends.csv')
        self.categories['embedding'] = self.categories['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), dtype=float, sep=' '))
        self.similarities = None
        self.client = None
        self.style_prompt = ""

    def get_embedding(self,query: str):
        # Define the API endpoint for the POST request
        url_post = "https://kerembayramoglu-cs533pir.hf.space/gradio_api/call/predict"

        # Define the query data
        data = {
            "data": [query]
        }

        # First POST request to trigger the model prediction
        response_post = requests.post(url_post, json=data)

        # Check if the first request was successful
        if response_post.status_code == 200:
            # Extract the event ID from the response
            event_id = response_post.json().get('event_id', None)

            if event_id:
                # Use the event ID to poll for the response
                url_get = f"https://kerembayramoglu-cs533pir.hf.space/gradio_api/call/predict/{event_id}"
                response_get = requests.get(url_get, stream=True)
                try:
                    data = response_get.text.split("data: ")[1]  # Get everything after 'data: '
                    data = json.loads(data)  # Convert the string representation of a list into an actual list
                    # Return the embeddings
                    return data
                except Exception as e:
                    print("Error parsing response:", e)
                    return None
            else:
                print("Event ID not found.")
                return None
        else:
            print(f"Error in POST request: {response_post.status_code}")
            return None
           
    def analyze_query_style(self, query: str) -> Dict:
        style_features = {
            'length': len(query.split()),
            'has_question_mark': '?' in query,
            'starts_with_question_word': any(query.lower().startswith(w) for w in ['how', 'what', 'where', 'when', 'why', 'who']),
            'capitalization': query[0].isupper() if query else False,
            'lowercase_ratio': sum(1 for c in query if c.islower()) / len(query) if query else 0
        }
        return style_features

    def identify_query_category(self, query: str) -> str:
        # Get query embedding
        query_embedding = self.get_embedding(query)
        # Get most similar category with cosine similarity
        similarities = self.categories['embedding'].apply(lambda x: cosine_similarity(np.array(query_embedding).reshape(1, -1), x.reshape(1, -1))[0][0])
        self.similarities = similarities
        most_similar_category = self.categories.iloc[self.similarities.idxmax()]
        return most_similar_category['category']
        
        

    # TODO: Discuss, and implement/change as needed
    def get_distant_categories(self, num_categories: int = 2) -> List[str]:
        # Get all embeddings as numpy array
        embeddings = np.array(self.categories['embedding'].tolist())
        
        # 1. First divide the embedding space into regions using K-means
        from sklearn.cluster import KMeans
        n_clusters = min(num_categories, len(self.categories))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 2. Get cluster centers distances to query
        query_cluster_distances = kmeans.transform(embeddings[self.similarities.idxmax()].reshape(1, -1))[0]
        
        # 3. Sample categories from each cluster
        selected_categories = []
        cluster_assignments = {i: [] for i in range(n_clusters)}
        
        # Group categories by cluster
        for idx, cluster in enumerate(cluster_labels):
            cluster_assignments[cluster].append(idx)
        
        # Calculate samples per cluster to ensure uniform distribution
        samples_per_cluster = num_categories // n_clusters
        remaining_samples = num_categories % n_clusters
        
        # Sample from each cluster
        for cluster_idx in range(n_clusters):
            cluster_indices = cluster_assignments[cluster_idx]
            
            # Calculate number of samples for this cluster
            n_samples = samples_per_cluster + (1 if cluster_idx < remaining_samples else 0)
            
            if cluster_indices:  # If cluster is not empty
                # Random sampling within cluster
                sampled_indices = random.sample(cluster_indices, min(n_samples, len(cluster_indices)))
                selected_categories.extend(self.categories.iloc[sampled_indices]['category'].tolist())
        
        # Shuffle the final selection to avoid any patterns
        random.shuffle(selected_categories)
        
        return selected_categories

    def generate_dummy_queries(self, input_query: str, num_queries: int = 2):
        # Analyze input query style
        query_style = self.analyze_query_style(input_query)
        
        # Identify input category
        input_category = self.identify_query_category(input_query)
        # Get distant categories
        distant_categories = self.get_distant_categories(num_queries)  # Get enough categories for individual queries
        
        # Prepare style prompt
        style_prompt = f"""
        Generate a query that matches these style characteristics:
        - Similar length (around {query_style['length']} words)
        - {'Use' if query_style['has_question_mark'] else 'Avoid'} question marks
        - {'Start with question words' if query_style['starts_with_question_word'] else 'Use declarative form'}
        - {'Capitalize first letter' if query_style['capitalization'] else 'Use lowercase'}
        """

        all_queries = []
        print(distant_categories)
        # Generate one query per category
        for category in distant_categories:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": f"""You are a query generation assistant. Generate a single query that:
                    1. Is completely unrelated to {input_category}
                    2. Is specifically about {category}
                    3. Matches the original query's style
                    {style_prompt}
                    Always respond with valid JSON containing a single query."""},
                    {"role": "user", "content": """Generate one search query.
                    Format your response as JSON with the following structure:
                    {
                        "queries": [
                            {"query": "query text", "category": "category_name"}
                        ]
                    }"""}
                ],
                response_format={ "type": "json_object" },
                temperature=0.7,
                seed=random.randint(0, 1000000)
            )
            
            # Parse and add the generated query
            query_result = parse_dummy_queries(completion.choices[0].message.content)
            all_queries.extend(query_result)
            
            # Break if we have enough queries
            if len(all_queries) >= num_queries:
                break
        all_queries = all_queries[:num_queries]
        all_queries.append({"query": input_query, "category": input_category})

        result = fetch_multiple_query_results(all_queries)
        return all_queries,result,input_category
    
    def generate_consecutive_queries(self,input_query, dummy_queries,input_category):        
        # Analyze input query style
        query_style = self.analyze_query_style(input_query)
        
        # Prepare style prompt
        style_prompt = f"""
        Generate a query that matches these style characteristics:
        - Similar length (around {query_style['length']} words)
        - {'Use' if query_style['has_question_mark'] else 'Avoid'} question marks
        - {'Start with question words' if query_style['starts_with_question_word'] else 'Use declarative form'}
        - {'Capitalize first letter' if query_style['capitalization'] else 'Use lowercase'}
        """
        all_queries = []
        for dummy_query in dummy_queries:
            # Generate a query that will act as a follow up to the dummy query
            follow_up_query = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "system", "content": f"""You are a query generation assistant. Generate a single query that:
                        1. Will act as a follow up to the query: {dummy_query['query']} 
                        2. Matches the following style characteristics:
                        {style_prompt}
                        Always respond with valid JSON containing a single query."""},
                        {"role": "user", "content": """Generate one search query.
                        Format your response as JSON with the following structure:
                        {
                            "queries": [
                                {"query": "query text"}
                            ]
                        }"""}],
                # TODO: No need for json object
                temperature=0.7,
                seed=random.randint(0, 1000000)
            )
            query_result = parse_dummy_queries(follow_up_query.choices[0].message.content)

            all_queries.extend(query_result)

        all_queries.append({"query": input_query, "category": input_category})        
        result = fetch_multiple_query_results(all_queries)
        return all_queries, result
    
def parse_dummy_queries(response):
    return json.loads(response)['queries']

def fetch_single_query_results(query, api_key):
    """
    Fetch search results for a single query using SerpAPI.
    Returns a list of dictionaries with title and URL.
    """
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "google",
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        organic_results = response.json().get("organic_results", [])
        return [
            {"title": result.get("title", ""), "url": result.get("link", "")}
            for result in organic_results
        ]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

def fetch_multiple_query_results(queries):
    """
    Fetch search results for multiple queries using SerpAPI.
    Returns a dictionary where keys are queries and values are lists of dictionaries (title and URL).
    """
    results = {}
    for query in queries:
        query = query["query"]
        results[query] = fetch_single_query_results(query, api_key)
        print("results[query]: ", results[query])
    return results
