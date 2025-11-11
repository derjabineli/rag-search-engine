from sentence_transformers import SentenceTransformer
import numpy as np
import os

from .search_utils import CACHE_DIR, load_movies

class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, 'movie_embeddings.npy')
    
    def generate_embedding(self, text):
        if not text:
            raise ValueError("text cannot be empty")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        documents_text = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            documents_text.append(f"{doc['title']}: {doc['description']}")
        embeddings = self.model.encode(documents_text, show_progress_bar=True)
        self.embeddings = embeddings
        np.save(self.embeddings_path, embeddings)
        return embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        similarities = []
        for i, e in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, e)
            similarities.append((similarity, self.documents[i]))
        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"]
                }
            )
        return results

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    sem_search = SemanticSearch()
    documents = load_movies()
    embeddings = sem_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query: str, limit: int):
    sem_search = SemanticSearch()
    movies = load_movies()
    sem_search.load_or_create_embeddings(movies)
    results = sem_search.search(query, limit)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()