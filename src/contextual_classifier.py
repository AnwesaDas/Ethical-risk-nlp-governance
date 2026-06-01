"""
contextual_classifier.py
------------------------
TF-IDF + Logistic Regression classifier to disambiguate ambiguous goods mentions.

Distinguishes contextually relevant uses (e.g. "gold" the metal) from
irrelevant uses (e.g. "gold" the colour or metaphor).

Achieves 94% accuracy on test set after removing "lead" (overwhelmingly verb).

HACE: Data Changing Child Labour
MSc Data Science - University of Manchester (2024-2025)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Tuple, List

from src.goods_dictionary import AMBIGUOUS_GOODS

# Goods to exclude from classifier (overwhelmingly used as verb)
EXCLUDED_FROM_CLASSIFIER = ["lead"]

# Active ambiguous goods (excluding problematic ones)
ACTIVE_AMBIGUOUS_GOODS = [g for g in AMBIGUOUS_GOODS
                           if g not in EXCLUDED_FROM_CLASSIFIER]


def build_classifier_pipeline() -> Pipeline:
    """
    Build sklearn Pipeline with TF-IDF vectoriser and Logistic Regression.

    Returns:
        Configured sklearn Pipeline
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),        # Unigrams and bigrams
            max_features=5000,          # Vocabulary limit
            min_df=2,                   # Minimum document frequency
            sublinear_tf=True,          # Apply log normalisation to TF
            strip_accents="unicode",
            lowercase=True
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight="balanced"    # Handle class imbalance
        ))
    ])
    return pipeline


def train_classifier(snippets_df: pd.DataFrame,
                     label_col: str = "label",
                     snippet_col: str = "snippet",
                     model_path: str = "outputs/classifier.pkl",
                     vectorizer_path: str = "outputs/vectorizer.pkl") -> Pipeline:
    """
    Train the contextual classifier on labelled snippets.

    Labels:
        1 = Relevant (commodity use of ambiguous good)
        0 = Irrelevant (non-commodity use)

    Args:
        snippets_df: DataFrame with snippet and label columns
        label_col: Name of label column
        snippet_col: Name of snippet column
        model_path: Path to save trained model
        vectorizer_path: Path to save vectorizer

    Returns:
        Trained sklearn Pipeline
    """
    # Filter to ambiguous goods only, exclude "lead"
    ambiguous_df = snippets_df[
        snippets_df["canonical_good"].isin(ACTIVE_AMBIGUOUS_GOODS)
    ].copy()

    print(f"Training on {len(ambiguous_df)} labelled snippets")
    print(f"Active ambiguous goods: {ACTIVE_AMBIGUOUS_GOODS}")
    print(f"Excluded: {EXCLUDED_FROM_CLASSIFIER} (overwhelmingly used as verb)")

    X = ambiguous_df[snippet_col].values
    y = ambiguous_df[label_col].values

    # Train/test split — stratified to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build and train pipeline
    pipeline = build_classifier_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\n" + "="*50)
    print("CONTEXTUAL CLASSIFIER PERFORMANCE")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Irrelevant", "Relevant"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")
    print(f"\n5-Fold CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

    return pipeline


def load_classifier(model_path: str = "outputs/classifier.pkl") -> Pipeline:
    """
    Load a previously trained classifier pipeline.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded sklearn Pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Please train the classifier first."
        )
    return joblib.load(model_path)


def filter_relevant_snippets(snippets_df: pd.DataFrame,
                              pipeline: Pipeline,
                              snippet_col: str = "snippet") -> pd.DataFrame:
    """
    Apply classifier to filter out irrelevant mentions of ambiguous goods.
    Non-ambiguous goods are kept as-is (always relevant).

    Args:
        snippets_df: DataFrame of all detected goods mentions
        pipeline: Trained classifier pipeline
        snippet_col: Name of snippet column

    Returns:
        Filtered DataFrame with only contextually relevant mentions
    """
    df = snippets_df.copy()

    # Separate ambiguous and non-ambiguous goods
    is_ambiguous = df["canonical_good"].isin(ACTIVE_AMBIGUOUS_GOODS)
    is_excluded = df["canonical_good"].isin(EXCLUDED_FROM_CLASSIFIER)

    ambiguous_df = df[is_ambiguous].copy()
    non_ambiguous_df = df[~is_ambiguous & ~is_excluded].copy()
    excluded_df = df[is_excluded].copy()

    print(f"Non-ambiguous mentions (kept): {len(non_ambiguous_df)}")
    print(f"Ambiguous mentions (to classify): {len(ambiguous_df)}")
    print(f"Excluded mentions ('{EXCLUDED_FROM_CLASSIFIER}'): {len(excluded_df)}")

    # Classify ambiguous mentions
    if len(ambiguous_df) > 0:
        snippets = ambiguous_df[snippet_col].values
        predictions = pipeline.predict(snippets)
        ambiguous_df["classifier_label"] = predictions

        # Count filtered
        n_filtered = (predictions == 0).sum()
        print(f"Irrelevant mentions filtered out: {n_filtered}")
        print(f"Relevant ambiguous mentions kept: {(predictions == 1).sum()}")

        # Keep only relevant
        relevant_ambiguous = ambiguous_df[ambiguous_df["classifier_label"] == 1]
    else:
        relevant_ambiguous = ambiguous_df

    # Combine non-ambiguous (all relevant) + relevant ambiguous
    non_ambiguous_df["classifier_label"] = 1
    filtered_df = pd.concat([non_ambiguous_df, relevant_ambiguous], ignore_index=True)

    print(f"\nTotal mentions after filtering: {len(filtered_df)}")
    return filtered_df


def create_labelling_template(snippets_df: pd.DataFrame,
                               output_path: str = "outputs/snippets_to_label.csv") -> None:
    """
    Export ambiguous snippets for manual labelling.
    Creates a CSV with snippet and empty label column.

    Args:
        snippets_df: DataFrame of detected mentions
        output_path: Path to save labelling template
    """
    ambiguous = snippets_df[
        snippets_df["canonical_good"].isin(ACTIVE_AMBIGUOUS_GOODS)
    ].copy()

    ambiguous["label"] = ""  # Empty for manual filling
    ambiguous["label_guide"] = "1=Relevant (commodity), 0=Irrelevant (other meaning)"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ambiguous[["doc_id", "company_id", "canonical_good",
               "matched_text", "snippet", "label", "label_guide"]].to_csv(
        output_path, index=False
    )
    print(f"Labelling template saved to: {output_path}")
    print(f"Please label {len(ambiguous)} snippets manually before training.")


if __name__ == "__main__":
    # Demo with synthetic labelled data
    print("Contextual Classifier Demo")
    print("="*50)

    # Synthetic labelled snippets for demonstration
    sample_snippets = [
        # Gold — Relevant (commodity)
        {"canonical_good": "gold", "snippet": "artisanal gold mining operations in Congo DRC extraction", "label": 1},
        {"canonical_good": "gold", "snippet": "gold ore processing facility workers supply chain audit", "label": 1},
        {"canonical_good": "gold", "snippet": "gold mining child labour risk West Africa cobalt", "label": 1},
        {"canonical_good": "gold", "snippet": "sourcing gold from certified responsible mining operations", "label": 1},
        {"canonical_good": "gold", "snippet": "gold procurement supply chain due diligence minerals", "label": 1},
        {"canonical_good": "gold", "snippet": "artisanal small scale gold ASGM mining communities", "label": 1},
        # Gold — Irrelevant (colour/metaphor)
        {"canonical_good": "gold", "snippet": "gold standard practice corporate governance excellence award", "label": 0},
        {"canonical_good": "gold", "snippet": "golden opportunity investors shareholders return value", "label": 0},
        {"canonical_good": "gold", "snippet": "gold medal performance employee recognition programme", "label": 0},
        {"canonical_good": "gold", "snippet": "as good as gold compliance record outstanding achievement", "label": 0},
        # Silver — Relevant
        {"canonical_good": "silver", "snippet": "silver mining operations artisanal extraction Peru Bolivia", "label": 1},
        {"canonical_good": "silver", "snippet": "silver ore supply chain responsible sourcing minerals", "label": 1},
        # Silver — Irrelevant
        {"canonical_good": "silver", "snippet": "silver bullet solution problem solved efficiently quickly", "label": 0},
        {"canonical_good": "silver", "snippet": "silver lining positive outcome difficult challenging year", "label": 0},
        # Rubber — Relevant
        {"canonical_good": "rubber", "snippet": "natural rubber plantation tapping Malaysia Indonesia workers", "label": 1},
        {"canonical_good": "rubber", "snippet": "rubber supply chain child labour risk monitoring audit", "label": 1},
        # Rubber — Irrelevant
        {"canonical_good": "rubber", "snippet": "rubber stamp approval process administrative regulatory", "label": 0},
        # Tin — Relevant
        {"canonical_good": "tin", "snippet": "tin mining cassiterite extraction DRC Congo supply chain", "label": 1},
        {"canonical_good": "tin", "snippet": "tin ore processing conflict minerals responsible sourcing", "label": 1},
        # Tin — Irrelevant
        {"canonical_good": "tin", "snippet": "tin ear feedback poor listening communication skills", "label": 0},
        # Diamonds — Relevant
        {"canonical_good": "diamonds", "snippet": "rough diamond mining Sierra Leone conflict minerals Kimberley", "label": 1},
        {"canonical_good": "diamonds", "snippet": "diamond supply chain child labour artisanal mining Africa", "label": 1},
        # Diamonds — Irrelevant
        {"canonical_good": "diamonds", "snippet": "diamond shaped logo brand corporate identity design", "label": 0},
    ]

    snippets_df = pd.DataFrame(sample_snippets)

    # Train classifier
    pipeline = train_classifier(
        snippets_df=snippets_df,
        model_path="outputs/classifier.pkl"
    )

    # Test filtering
    print("\n\nTesting filter on new snippets:")
    test_snippets = pd.DataFrame([
        {"doc_id": 1, "company_id": "ABC", "canonical_good": "gold",
         "matched_text": "gold", "snippet": "gold mining supply chain workers Africa extraction"},
        {"doc_id": 1, "company_id": "ABC", "canonical_good": "gold",
         "matched_text": "gold", "snippet": "gold standard excellent performance record award"},
        {"doc_id": 1, "company_id": "ABC", "canonical_good": "cocoa",
         "matched_text": "cocoa", "snippet": "cocoa farming child labour Ivory Coast supply chain"},
    ])

    filtered = filter_relevant_snippets(test_snippets, pipeline)
    print("\nFiltered results:")
    print(filtered[["company_id", "canonical_good", "snippet"]].to_string())
