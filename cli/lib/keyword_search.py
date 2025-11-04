from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    tokenized_query = tokenize_text(query)
    for movie in movies:
        tokenized_title = tokenize_text(movie["title"])
        if match_exists(tokenized_query, tokenized_title):
            results.append(movie)
            if len(results) >= limit:
                break
    return results
    
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