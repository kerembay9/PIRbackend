import json
import random
from flask import Flask, request, jsonify
from query import DummyQueryGenerator  # Assuming you saved your class in this module

app = Flask(__name__)

# Create an instance of DummyQueryGenerator
query_generator = DummyQueryGenerator()

@app.route('/generate-dummy-queries', methods=['POST'])
def generate_dummy_queries():
    try:
        data = request.get_json()
        input_query = data.get('input_query', '')
        num_queries = data.get('num_queries', 2)

        if not input_query:
            return jsonify({"error": "input_query is required"}), 400

        # Generate dummy queries
        all_queries, result, input_category = query_generator.generate_dummy_queries(input_query, num_queries)

        return jsonify({
            "all_queries": all_queries,
            "result": result,
            "input_category": input_category
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate-consecutive-queries', methods=['POST'])
def generate_consecutive_queries():
    try:
        data = request.get_json()
        input_query = data.get('input_query', '')
        dummy_queries = data.get('dummy_queries', [])
        input_category = data.get('input_category', '')

        if not input_query or not dummy_queries or not input_category:
            return jsonify({"error": "input_query, dummy_queries, and input_category are required"}), 400

        # Generate consecutive queries
        all_queries, result = query_generator.generate_consecutive_queries(input_query, dummy_queries, input_category)

        return jsonify({
            "all_queries": all_queries,
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5000")
