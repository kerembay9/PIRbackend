from query import DummyQueryGenerator

generator = DummyQueryGenerator()
dummy_queries = generator.generate_dummy_queries("cancer treatment")
print('Initial dummy queries: ')
for query in dummy_queries:
    print(query)

consecutive_queries = generator.generate_consecutive_queries("cancer symptoms", dummy_queries)
print('Consecutive queries: ')
for query in consecutive_queries:
    print(query)
    
generator.visualize_query_trajectories("cancer treatment", dummy_queries, consecutive_queries)

consecutive_queries_2 = generator.generate_consecutive_queries("chemotherapy side effects", consecutive_queries)
print('Consecutive queries 2: ')
for query in consecutive_queries_2:
    print(query)