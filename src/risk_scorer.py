"""
risk_scorer.py
--------------
Company risk scoring based on unique high-risk goods mentions in governance documents.

Assigns each document a normalised risk score (0-100) based on the number of
UNIQUE relevant goods mentioned — avoiding bias from repetitive boilerplate text.

HACE: Data Changing Child Labour
MSc Data Science - University of Manchester (2024-2025)
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List


def compute_document_risk_scores(filtered_snippets_df: pd.DataFrame,
                                  gov_docs_df: pd.DataFrame,
                                  doc_id_col: str = "doc_id",
                                  company_col: str = "company_id",
                                  good_col: str = "canonical_good") -> pd.DataFrame:
    """
    Compute risk scores for each governance document.

    Scoring logic:
    - Count UNIQUE relevant goods per document (not total mentions)
    - Using unique goods avoids inflating scores from repetitive boilerplate
    - Normalise to 0-100 scale using linear scaling

    Args:
        filtered_snippets_df: DataFrame of contextually relevant goods mentions
        gov_docs_df: Original governance documents DataFrame
        doc_id_col: Column name for document ID
        company_col: Column name for company ID
        good_col: Column name for canonical good

    Returns:
        DataFrame with risk scores merged into governance documents
    """
    print("Computing document-level risk scores...")

    # Aggregate: unique goods per document
    doc_aggregates = (
        filtered_snippets_df
        .groupby([doc_id_col, company_col])[good_col]
        .agg(
            total_mentions="count",
            unique_goods=lambda x: x.nunique(),
            detected_goods=lambda x: list(x.unique())
        )
        .reset_index()
    )

    print(f"Documents with detections: {len(doc_aggregates)}")
    print(f"Max unique goods in any document: {doc_aggregates['unique_goods'].max()}")

    # Normalise unique goods count to 0-100 scale (linear scaling)
    max_unique = doc_aggregates["unique_goods"].max()

    if max_unique > 0:
        doc_aggregates["risk_score"] = (
            doc_aggregates["unique_goods"] / max_unique * 100
        ).round(2)
    else:
        doc_aggregates["risk_score"] = 0.0

    # Merge scores back into governance documents
    scored_df = gov_docs_df.merge(
        doc_aggregates[[doc_id_col, company_col, "total_mentions",
                        "unique_goods", "detected_goods", "risk_score"]],
        left_index=True,
        right_on=doc_id_col,
        how="left"
    )

    # Documents with no detections get score of 0
    scored_df["risk_score"] = scored_df["risk_score"].fillna(0.0)
    scored_df["unique_goods"] = scored_df["unique_goods"].fillna(0).astype(int)
    scored_df["total_mentions"] = scored_df["total_mentions"].fillna(0).astype(int)
    scored_df["detected_goods"] = scored_df["detected_goods"].fillna("").apply(
        lambda x: x if isinstance(x, list) else []
    )

    print(f"Documents scored: {len(scored_df)}")
    print(f"Documents scoring 0: {(scored_df['risk_score'] == 0).sum()}")
    print(f"Documents scoring 100: {(scored_df['risk_score'] == 100).sum()}")

    return scored_df


def get_top_companies(scored_df: pd.DataFrame,
                      company_col: str = "company_id",
                      company_name_col: str = "company_name",
                      n: int = 20) -> pd.DataFrame:
    """
    Return top N companies by risk score.

    Note: Multiple entries per company are kept (intentional design decision):
    - Companies may evolve operations over time
    - Aggregating would favour companies with fewer documents
    - Multiple appearances signal multiple high-risk disclosures

    Args:
        scored_df: DataFrame with risk scores
        company_col: Column name for company ID
        company_name_col: Column name for company display name
        n: Number of top companies to return

    Returns:
        Top N DataFrame sorted by risk score descending
    """
    cols = [company_col, "risk_score", "unique_goods",
            "total_mentions", "detected_goods"]
    if company_name_col in scored_df.columns:
        cols = [company_name_col] + cols

    top_df = (
        scored_df[cols]
        .sort_values("risk_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    top_df.index += 1  # 1-based ranking

    return top_df


def generate_risk_report(scored_df: pd.DataFrame,
                          output_path: str = "outputs/company_risk_scores.csv") -> None:
    """
    Export full risk scores to CSV for HACE analyst use.

    Args:
        scored_df: DataFrame with risk scores
        output_path: Path to save output CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    export_cols = []
    for col in ["company_id", "company_name", "sector", "sub_industry",
                "company_size", "risk_score", "unique_goods",
                "total_mentions", "detected_goods", "doc_length"]:
        if col in scored_df.columns:
            export_cols.append(col)

    export_df = (
        scored_df[export_cols]
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
    )
    export_df.index += 1
    export_df.index.name = "rank"

    export_df.to_csv(output_path)
    print(f"\nRisk scores exported to: {output_path}")
    print(f"Total documents exported: {len(export_df)}")


def get_score_distribution_stats(scored_df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics of risk score distribution.

    Args:
        scored_df: DataFrame with risk scores

    Returns:
        Dict of distribution statistics
    """
    scores = scored_df["risk_score"]
    stats = {
        "total_documents": len(scores),
        "scoring_zero": int((scores == 0).sum()),
        "scoring_above_60": int((scores > 60).sum()),
        "scoring_100": int((scores == 100).sum()),
        "mean_score": round(float(scores.mean()), 2),
        "median_score": round(float(scores.median()), 2),
        "std_score": round(float(scores.std()), 2),
        "max_score": round(float(scores.max()), 2),
    }
    return stats


if __name__ == "__main__":
    # Demo with synthetic data
    print("Risk Scorer Demo")
    print("="*50)

    # Synthetic governance documents
    gov_docs = pd.DataFrame({
        "company_id": ["CO001", "CO002", "CO003", "CO004", "CO005"],
        "company_name": [
            "Watches of Switzerland Group PLC",
            "Greencore Group",
            "Vesuvius PLC",
            "Hormel Foods Corp",
            "Small Retail Co"
        ],
        "sector": ["Consumer Discretionary", "Consumer Staples",
                   "Industrials", "Consumer Staples", "Consumer Discretionary"],
        "text": ["doc1", "doc2", "doc3", "doc4", "doc5"]
    })

    # Synthetic filtered snippets (post-classifier)
    filtered_snippets = pd.DataFrame({
        "doc_id": [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4],
        "company_id": ["CO001"]*4 + ["CO002"]*3 + ["CO003"]*2 + ["CO004"] + ["CO005"],
        "canonical_good": [
            "gold", "diamonds", "cobalt", "tin",           # CO001: 4 unique goods → score 100
            "cocoa", "palm oil", "cotton",                   # CO002: 3 unique goods
            "coal", "rubber",                                # CO003: 2 unique goods
            "sugarcane",                                     # CO004: 1 unique good
            "cocoa",                                         # CO005: 1 unique good (same score as CO004)
        ],
        "snippet": ["context"] * 11
    })

    # Score documents
    scored = compute_document_risk_scores(filtered_snippets, gov_docs)

    # Top companies
    print("\nTop Companies by Risk Score:")
    top = get_top_companies(scored, n=5)
    print(top[["company_name", "risk_score", "unique_goods", "detected_goods"]].to_string())

    # Distribution stats
    stats = get_score_distribution_stats(scored)
    print("\nScore Distribution:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Export
    generate_risk_report(scored)
