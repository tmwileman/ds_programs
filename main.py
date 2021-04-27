import pandas as pd

from functions import scrape_curriculums, score_programs
from programs import programs


def main():
    processed_texts = scrape_curriculums(programs)
    scores = score_programs(processed_texts)
    scores = pd.DataFrame(scores)
    scores.to_csv("scores.csv")


if __name__ == "__main__":
    scores = main()
