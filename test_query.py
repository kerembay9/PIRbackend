import json
from query import DummyQueryGenerator

# Instantiate the DummyQueryGenerator and generate dummy queries
generator = DummyQueryGenerator()
dummy_queries, results,input_category = generator.generate_dummy_queries("cancer treatment")

# Print the initial dummy queries and results
print('Initial dummy queries: ')
print(dummy_queries, "res:", results)

# Save results to a JSON file
results_filename = 'dummy_query_results.json'
with open(results_filename, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f'Results have been saved to {results_filename}')


# consecutive_queries, consec_results = generator.generate_consecutive_queries("cancer symptoms", dummy_queries,input_category)
# print('Consecutive queries: ')
# for query in consecutive_queries:
#     print(query)
    
# consecutive_queries_2 = generator.generate_consecutive_queries("chemotherapy side effects", consecutive_queries)
# print('Consecutive queries 2: ')
# for query in consecutive_queries_2:
#     print(query)