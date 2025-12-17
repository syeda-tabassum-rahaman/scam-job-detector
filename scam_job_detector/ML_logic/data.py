"""
Utilities for loading, cleaning and saving the scam-job dataset.

This module provides a small preprocessing pipeline used to prepare the
fake job postings dataset for feature extraction and modeling. It contains
functions to clean textual fields, reduce categorical cardinality, and
persist cleaned data to disk.

"""

import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame of raw job postings.
    
    Steps performed:
    - Combine textual fields into a single `job_description` column and apply
      lightweight tokenization, stopword removal and lemmatization.
    - Extract a simplified `country` field from the `location` column.
    - Reduce cardinality of `industry`, `country` and `function` to their
      top-10 values and label the rest as "Other".
    - Drop a set of columns considered irrelevant for modeling.

    Returns:
        A cleaned DataFrame ready for feature extraction.
    """

    def preprocessing(sentence: str) -> str:
        """
        Clean a single text string.
        - Removes punctuation and digits
        - Converts to lowercase
        - Tokenizes, removes English stopwords and lemmatizes verbs

        Returns:
            A cleaned, space-separated token string.
        """

        stop_words = set(stopwords.words('english'))

        # replace punctuation characters with space to avoid joining words
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, ' ')

        # normalize case
        sentence = sentence.lower()

        # remove digits in one pass (safer and faster than repeated replacements)
        sentence = ''.join(ch for ch in sentence if not ch.isdigit())

        # tokenize the cleaned string
        tokens = word_tokenize(sentence)

        # remove common English stop words
        tokens = [word for word in tokens if word not in stop_words]

        # lemmatize using verb pos to get base forms where appropriate
        tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]

        return ' '.join(tokens)

    # columns containing text to be combined for vectorization
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']

    # ensure text columns are strings and missing values become empty strings
    df[text_columns] = df[text_columns].fillna("").astype(str)

    # combine all text fields into a single field used for TF-IDF/embedding input
    df["job_description"] = df[text_columns].agg(" ".join, axis=1).str.strip()

    # Fill any remaining NAs and apply the lightweight text preprocessing
    df['job_description'] = df['job_description'].fillna('missing value')
    df['job_description'] = df['job_description'].apply(preprocessing)

    # extract country part from location (before first comma)
    df['country'] = df['location'].astype(str).apply(lambda x: x.split(',')[0])

    # reduce cardinality to top-10 for selected categorical features
    top10_idx = df["industry"].value_counts().head(10).index
    df["industry"] = df["industry"].where(df["industry"].isin(top10_idx), "Other")

    top10_idx = df["country"].value_counts().head(10).index
    df["country"] = df["country"].where(df["country"].isin(top10_idx), "Other")

    top10_idx = df["function"].value_counts().head(10).index
    df["function"] = df["function"].where(df["function"].isin(top10_idx), "Other")

    # drop columns not used by the downstream model pipeline
    df.drop(columns=['department', 'has_questions', 'required_experience',
                     'salary_range', 'telecommuting', 'required_education',
                     'location', 'title', 'company_profile', 'description',
                     'requirements', 'benefits', 'job_id'], inplace=True)

    print("✅ data cleaned")

    return df

def save_clean_data():
    """
    Load raw CSV, run the `clean_data` pipeline and persist the result.

    This convenience function reads `raw_data/fake_job_postings.csv` 
    relative to the repository root and writes `raw_data/data_cleaned.csv`.
    """
    # build an absolute path to the raw CSV file
    raw_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'raw_data',
        'fake_job_postings.csv'
    )

    # read the raw dataset
    df_raw = pd.read_csv(raw_path)

    # run the cleaning pipeline
    df_clean = clean_data(df_raw)

    # construct the destination path for the cleaned CSV
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'raw_data',
        'data_cleaned.csv'
    )

    # persist the cleaned data without the index column
    df_clean.to_csv(save_path, index=False)

    # inform the user where the cleaned data was written
    print(f"✅ Cleaned data saved at: {save_path}")

if __name__ == "__main__":
    save_clean_data()
