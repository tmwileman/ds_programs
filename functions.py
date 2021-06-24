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
import seaborn as sns
import matplotlib.pyplot as plt
import string
from typing import Dict, List, NamedTuple

from groupings import deep_learning, machine_learning, analytics, management


class Score(NamedTuple):
    university: str
    total_tokens: int
    deep_learning_score: int
    machine_learning_score: int
    analytics_score: int
    technical_score: int
    management_score: int
    technical_per_word: int
    management_per_word: int


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


def scrape_curriculums(sites: Dict[str, str]) -> Dict[str, str]:
    """Scrape each program's curriculum page and return the program as the key and processed text as the value in a dictionary

    @type  sites: dictionary 
    @param sites: keys == programs and values == urls 
    """

    processed_texts = {}
    results = {}

    for key, value in sites.items():
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
    for university, tokens in processed_texts.items():
        deep_learning_count = 0
        machine_learning_count = 0
        analytics_count = 0
        management_count = 0
        counts = dict(Counter(chain(*tokens)))
        total_tokens = sum(counts.values())
        for token, count in counts.items():
            if token in deep_learning:
                deep_learning_count += count
            if token in machine_learning:
                machine_learning_count += count
            if token in analytics:
                analytics_count += count
            if token in management:
                management_count += count
        technical_score = deep_learning_count + machine_learning_count + analytics_count
        technical_per_word = technical_score / total_tokens
        management_per_word = management_count / total_tokens
        result = Score(
            university,
            total_tokens,
            deep_learning_count,
            machine_learning_count,
            analytics_count,
            technical_score,
            management_count,
            technical_per_word,
            management_per_word,
        )
        scores.append(result)

    return scores


def scatter_plot(x, y, text_column, hue, data, title, xlabel, ylabel):
    """
    Scatter plot with country codes on the x y coordinates
    Based on this answer: https://stackoverflow.com/a/54789170/2641825
    """
    # Create the scatter plot
    sns.set(rc={"figure.figsize": (15, 10)})
    p1 = sns.scatterplot(x, y, data=data, hue=hue, size=20, legend=False)
    # Add text besides each point
    for line in range(0, data.shape[0]):
        p1.text(
            data[x][line] + 0.01,
            data[y][line],
            data[text_column][line],
            # horizontalalignment="left",
            size="medium",
            color="black",
        )
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("ds_programs.png")
    return p1
