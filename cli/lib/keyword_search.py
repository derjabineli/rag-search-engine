import os
import pickle
import string
import collections
import math

from nltk.stem import PorterStemmer

from .search_utils import (DEFAULT_SEARCH_LIMIT, CACHE_DIR, BM25_K1, BM25_B, load_movies, load_stopwords)

class InvertedIndex:
    def __init__(self):
        self.index={}
        self.docmap={}
        self.term_frequencies: dict[int, collections.Counter] = {}
        self.doc_lengths = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = collections.Counter()
            self.term_frequencies[doc_id][token] += 1
        doc_length = len(tokens)
        self.doc_lengths[doc_id] = doc_length

    def __get_avg_doc_length(self) -> float:
        docs_count = len(self.doc_lengths)
        if docs_count == 0:
            return 0.0
        total = 0
        for l in self.doc_lengths:
            total += self.doc_lengths[l]
        return total / docs_count

    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise TypeError
        token = tokens[0]
        if token not in self.term_frequencies[doc_id]:
            return 0
        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_tfidf(self, doc_id, term) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        tf_idf = tf * idf
        return tf_idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        N = len(self.docmap)
        df = len(self.index.get(token, []))

        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1 = BM25_K1, b = BM25_B):
        raw_tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit) -> list[dict]:
        tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            for token in tokens:
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += self.bm25(doc_id, token)
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results = []
        for s in sorted_docs[:limit]:
            doc_id = s[0]
            score = s[1]
            doc = self.docmap[doc_id]
            result = {}
            result["id"] = doc_id
            result["score"] = score
            result["title"] = doc["title"]
            result["description"] = doc["description"]
            results.append(result)
        return results

    def get_documents(self, term) -> list:
        term = term.lower()
        if term not in self.index:
            return []
        documents = self.index[term]
        sorted_documents = list(documents)
        sorted_documents.sort()
        return sorted_documents
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie
            
    def save(self):
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)        

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path):
            raise FileNotFoundError
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
            

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    try:
        idx = InvertedIndex()
        idx.load()
    except FileNotFoundError:
        print("Error: File not found.")
    
    res_count = 0
    documents = []
    tokenized_query = tokenize_text(query)
    for token in tokenized_query:
        results = idx.get_documents(token)
        for r in results:
            documents.append(idx.docmap[r])
            res_count += 1
            if res_count >= 5:
                break
    return documents
        
def tf_command(doc_id: int, term: str) -> int:
    try:
        idx = InvertedIndex()
        idx.load()
    except FileNotFoundError:
        print("Error: File not found.")
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(doc_id, term)

def bm25idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25search_command(query: str, limit: int = 5):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)
    
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_tokens = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_tokens.append(word)
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for word in filtered_tokens:
        stemmed_tokens.append(stemmer.stem(word))
    return stemmed_tokens


def match_exists(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
