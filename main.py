import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from functions import scrape_curriculums, score_programs, scatter_plot
from programs import programs
from sites import sites


def main():
    processed_texts = scrape_curriculums(sites)
    scores = score_programs(processed_texts)
    scores = pd.DataFrame(scores)
    program_type = pd.DataFrame(
        programs.items(), columns=["university", "program_type"]
    )
    scores = pd.merge(scores, program_type, how="left", on="university")

    columns_to_normalize = ["technical_per_word", "management_per_word"]
    x = scores[columns_to_normalize].values
    x_scaled = MinMaxScaler().fit_transform(x)
    scores_temp = pd.DataFrame(
        x_scaled, columns=columns_to_normalize, index=scores.index
    )
    scores[columns_to_normalize] = scores_temp

    scatter_plot(
        "management_per_word",
        "technical_per_word",
        "university",
        "program_type",
        data=scores,
        title="test",
        xlabel="management scores",
        ylabel="technical scores",
    )

    scores.to_csv("scores.csv")


if __name__ == "__main__":
    main()
