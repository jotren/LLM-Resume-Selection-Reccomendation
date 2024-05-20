from flask import Flask, request, jsonify
import pandas as pd
from src.preprocess import clean_text, preprocess_for_bm25
from src.embeddings import create_embeddings, tokenize
from src.similarity import get_combined_similarity
from rank_bm25 import BM25Okapi
import numpy as np

app = Flask(__name__)

@app.route('/create_embeddings', methods=['POST'])
def return_embeddings():
    data = request.get_json()

    # Check if the data is provided
    if not data or 'resumes' not in data:
        return jsonify({"error": "No data provided"}), 400

    # Convert data to DataFrame
    df = pd.DataFrame(data['resumes'])

    # Preprocess the DataFrame
    df['cleaned_resume'] = df['resume'].apply(clean_text)
    df['tokenized_data'] = df['cleaned_resume'].apply(lambda x: tokenize(x))
    df['embeddings'] = df['tokenized_data'].apply(lambda row: create_embeddings(row))

    # Convert embeddings to list for JSON serialization
    embeddings_list = df['embeddings'].apply(lambda x: x.tolist()).tolist()

    return jsonify({"embeddings": embeddings_list}), 200

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity_score():
    data = request.get_json()

    # Check if the data is provided
    if not data or 'embeddings' not in data or 'query' not in data:
        return jsonify({"error": "No data provided"}), 400

    embeddings_list = data['embeddings']
    query = data['query']

    # Convert list back to numpy array
    embeddings_matrix = np.array(embeddings_list)

    # Ensure embeddings_matrix is 2D
    embeddings_matrix = embeddings_matrix.reshape(len(embeddings_matrix), -1)

    # Create a dummy DataFrame for similarity calculations
    df = pd.DataFrame({'embeddings': embeddings_list})
    df['embeddings'] = df['embeddings'].apply(lambda x: np.array(x))

    # Tokenize the resumes for BM25
    df['tokenized_resume'] = df['embeddings'].apply(lambda x: preprocess_for_bm25(' '.join(map(str, x))))

    # Create a BM25 object
    bm25 = BM25Okapi(df['tokenized_resume'].tolist())

    # Calculate combined similarity
    combined_similarity = get_combined_similarity(df, query, embeddings_matrix, bm25, df['embeddings'])

    return jsonify({"similarity_scores": combined_similarity}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
