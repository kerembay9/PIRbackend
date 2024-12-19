from flask import Flask, jsonify, request
from flask_cors import CORS  # Import the CORS package

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

@app.route('/query', methods=['GET'])
def query():
    search_term = request.args.get('search')

    # For simplicity, let's return some mock results based on the search term
    if search_term:
        results = [
            {"title": f"Result for {search_term} 1", "link": f"https://example.com/{search_term}-1"},
            {"title": f"Result for {search_term} 2", "link": f"https://example.com/{search_term}-2"},
            {"title": f"Result for {search_term} 3", "link": f"https://example.com/{search_term}-3"},
        ]
        return jsonify({"results": results})
    else:
        return jsonify({"results": []})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
