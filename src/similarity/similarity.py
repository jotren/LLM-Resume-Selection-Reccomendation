import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
from src.preprocess import preprocess_for_bm25
from src.embeddings import tokenize, create_embeddings

def calculate_cosine_similarity(query_embedding, embeddings_matrix):
    cosine_similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings_matrix)
    return cosine_similarities[0]

def normalize_scores(scores):
    mean_score = scores.mean()
    std_score = scores.std()

    if std_score == 0:
        normalized_scores = scores - mean_score
    else:
        normalized_scores = (scores - mean_score) / std_score

    return normalized_scores

def get_combined_similarity(df, query, embeddings_matrix, bm25, embeddings):
    # Query text processing for BERT
    query_tokenized = tokenize(query)
    query_embedding = create_embeddings(query_tokenized).flatten()

    # Calculate Cosine Similarity using BERT embeddings
    cosine_similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings_matrix)
    normalized_cosine_similarities = normalize_scores(cosine_similarities[0])

    # Query preprocessing for BM25
    tokenized_query = preprocess_for_bm25(query)

    # Get BM25 scores
    bm25_scores = bm25.get_scores(tokenized_query)
    normalized_bm25_scores = normalize_scores(bm25_scores)

    # Adjusted similarity by combining BM25 and BERT-based cosine similarity
    combined_similarity = normalized_cosine_similarities + normalized_bm25_scores

    return combined_similarity.tolist()
