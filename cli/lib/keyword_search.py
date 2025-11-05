import os
import pickle
import string
import collections
import math

from nltk.stem import PorterStemmer

from .search_utils import (DEFAULT_SEARCH_LIMIT, CACHE_DIR, load_movies, load_stopwords)

class InvertedIndex:
    def __init__(self):
        self.index={}
        self.docmap={}
        self.term_frequencies: dict[int, collections.Counter] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = collections.Counter()
            self.term_frequencies[doc_id][token] += 1

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
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie['id'], text)
            self.docmap[movie["id"]] = movie
            
    def save(self):
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)    

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path):
            raise FileNotFoundError
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
            

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
    
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split(" ")
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
