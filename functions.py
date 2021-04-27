from bs4 import BeautifulSoup
from collections import Counter
from itertools import chain
import nltk

nltk.download("punkt")
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
import requests
import string
from typing import Dict, List, NamedTuple

from groupings import deep_learning, machine_learning, analytics, management


class Score(NamedTuple):
    program: str
    deep_learning_score: int
    machine_learning_score: int
    analytics_score: int
    management_score: int


def process_text(raw_text) -> List[str]:
    """Process raw text from html

    @type  raw_text: str
    @param raw_text: text from each programs curriculum page
    """

    stop_words = stopwords.words("english")
    porter = PorterStemmer()

    tokens = []
    text_p = "".join(raw_text)  # make text a string
    text_p = text_p.lower()  # make all lowercase
    text_p = "".join(
        [char for char in text_p if char not in string.punctuation]
    )  # remove punctuation
    text_p = word_tokenize(text_p)  # tokenize words
    text_p = [word for word in text_p if word not in stop_words]  # remove stop words
    text_p = [porter.stem(word) for word in text_p]  # stem all tokens
    bigrms = list(ngrams(text_p, 2))

    return text_p, bigrms


def scrape_curriculums(programs: Dict[str, str]) -> Dict[str, str]:
    """Scrape each program's curriculum page and return the program as the key and processed text as the value in a dictionary

    @type  programs: dictionary 
    @param programs: keys == programs and values == urls 
    """

    processed_texts = {}
    results = {}

    for key, value in programs.items():
        html = requests.get(value)
        soup = BeautifulSoup(html.text, "html.parser")
        text = soup.get_text()
        tokens, bigrms = process_text(text)
        processed_texts[key] = tokens, bigrms

    return processed_texts


def score_programs(processed_texts: Dict):
    """Count how many times a token or bigram in each programs curriculum page matches a token in a list on groupings.py

    @type  processed_texts: dictionary 
    @param programs: keys == programs and values == processed texts 
    """

    scores = []
    for institution, tokens in processed_texts.items():
        deep_learning_count = 0
        machine_learning_count = 0
        analytics_count = 0
        management_count = 0
        counts = dict(Counter(chain(*tokens)))
        for token, count in counts.items():
            if token in deep_learning:
                deep_learning_count += count
            if token in machine_learning:
                machine_learning_count += count
            if token in analytics:
                analytics_count += count
            if token in management:
                management_count += count
        result = Score(
            institution,
            deep_learning_count,
            machine_learning_count,
            analytics_count,
            management_count,
        )
        scores.append(result)

    return scores
