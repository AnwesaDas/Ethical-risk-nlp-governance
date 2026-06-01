"""
preprocessing.py
----------------
Text preprocessing pipeline for corporate governance documents.
Handles tokenisation, stopword removal, and lemmatisation using spaCy.

HACE: Data Changing Child Labour
MSc Data Science - University of Manchester (2024-2025)
"""

import spacy
import pandas as pd
from typing import List

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Custom stopwords to RETAIN — legally significant terms
RETAIN_STOPWORDS = {
    "not", "no", "nor", "against", "without", "never",
    "to", "be", "held", "responsible", "liable"
}


def tokenise(text: str) -> List[str]:
    """
    Tokenise text into individual word/punctuation tokens using spaCy.

    Args:
        text: Raw document text string

    Returns:
        List of token strings
    """
    doc = nlp(text)
    return [token.text for token in doc]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove common stopwords while retaining legally significant terms.

    Args:
        tokens: List of token strings

    Returns:
        Filtered list of tokens
    """
    doc = nlp(" ".join(tokens))
    filtered = []
    for token in doc:
        # Retain if not a stopword OR if it's in our custom retain list
        if not token.is_stop or token.text.lower() in RETAIN_STOPWORDS:
            if not token.is_punct and not token.is_space:
                filtered.append(token.text)
    return filtered


def lemmatise(tokens: List[str]) -> List[str]:
    """
    Lemmatise tokens to their base dictionary form using spaCy POS-aware lemmatisation.
    Applies manual exceptions for frequently misclassified terms.

    Args:
        tokens: List of token strings

    Returns:
        List of lemmatised tokens
    """
    # Manual exceptions for ambiguous terms
    manual_exceptions = {
        "lead": "lead",      # Keep as-is — context classifier handles this
        "leading": "lead",
        "leads": "lead",
    }

    doc = nlp(" ".join(tokens))
    lemmatised = []
    for token in doc:
        text_lower = token.text.lower()
        if text_lower in manual_exceptions:
            lemmatised.append(manual_exceptions[text_lower])
        else:
            lemmatised.append(token.lemma_.lower())
    return lemmatised


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline: tokenise → remove stopwords → lemmatise.

    Args:
        text: Raw document text

    Returns:
        Preprocessed text string (space-joined lemmatised tokens)
    """
    tokens = tokenise(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatise(tokens)
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame column.

    Args:
        df: Input DataFrame with text column
        text_col: Name of the column containing document text

    Returns:
        DataFrame with added 'processed_text' column
    """
    print(f"Preprocessing {len(df)} documents...")
    df = df.copy()
    df["processed_text"] = df[text_col].apply(
        lambda x: preprocess_text(str(x)) if pd.notna(x) else ""
    )
    print("Preprocessing complete.")
    return df


if __name__ == "__main__":
    # Quick test
    sample = "The company is leading efforts to eliminate child labour from gold mining operations."
    print("Original:", sample)
    print("Processed:", preprocess_text(sample))
