"""
main.py
-------
Main pipeline script for the HACE Ethical Risk Scoring project.

Runs the full pipeline:
1. Load data
2. Preprocess governance documents
3. Detect high-risk goods using PhraseMatcher
4. Classify ambiguous mentions using TF-IDF + Logistic Regression
5. Compute company risk scores
6. Export results

HACE: Data Changing Child Labour
MSc Data Science - University of Manchester (2024-2025)
"""

import pandas as pd
import os
import argparse
from src.preprocessing import preprocess_dataframe
from src.goods_detector import detect_goods_in_corpus, build_phrase_matcher
from src.contextual_classifier import (
    train_classifier, load_classifier,
    filter_relevant_snippets, create_labelling_template
)
from src.risk_scorer import (
    compute_document_risk_scores, get_top_companies,
    generate_risk_report, get_score_distribution_stats
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_PATH = "outputs/classifier.pkl"

GOV_DOCS_PATH = os.path.join(DATA_DIR, "governance_documents.csv")
GOODS_PATH = os.path.join(DATA_DIR, "goods_dataset.csv")
DOWNSTREAM_PATH = os.path.join(DATA_DIR, "downstream_goods.csv")
LABELLED_SNIPPETS_PATH = os.path.join(DATA_DIR, "labelled_snippets.csv")


def load_data():
    """Load all three datasets."""
    print("Loading data...")

    if not os.path.exists(GOV_DOCS_PATH):
        raise FileNotFoundError(
            f"Governance documents not found at {GOV_DOCS_PATH}\n"
            "Please place your dataset files in the data/ directory."
        )

    gov_docs = pd.read_csv(GOV_DOCS_PATH)
    print(f"Governance documents loaded: {len(gov_docs)} records")

    goods_df = pd.read_csv(GOODS_PATH) if os.path.exists(GOODS_PATH) else None
    downstream_df = pd.read_csv(DOWNSTREAM_PATH) if os.path.exists(DOWNSTREAM_PATH) else None

    if goods_df is not None:
        print(f"Goods dataset loaded: {len(goods_df)} goods")
    if downstream_df is not None:
        print(f"Downstream goods loaded: {len(downstream_df)} records")

    return gov_docs, goods_df, downstream_df


def run_full_pipeline(retrain: bool = False):
    """
    Execute the complete HACE risk scoring pipeline.

    Args:
        retrain: If True, retrain classifier even if model exists
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("HACE ETHICAL RISK SCORING PIPELINE")
    print("MSc Data Science - University of Manchester")
    print("="*60 + "\n")

    # ── Step 1: Load Data ────────────────────────────────────────────────────
    print("STEP 1: Loading Data")
    print("-"*40)
    gov_docs, goods_df, downstream_df = load_data()

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    print("\nSTEP 2: Preprocessing Governance Documents")
    print("-"*40)
    gov_docs = preprocess_dataframe(gov_docs, text_col="text")

    # ── Step 3: Detect Goods ─────────────────────────────────────────────────
    print("\nSTEP 3: Detecting High-Risk Goods (PhraseMatcher)")
    print("-"*40)
    snippets_df = detect_goods_in_corpus(
        gov_docs,
        text_col="processed_text",
        company_col="company_id"
    )

    print(f"\nTotal mentions detected: {len(snippets_df)}")
    if len(snippets_df) > 0:
        print(f"Unique goods detected: {snippets_df['canonical_good'].nunique()}")
        print(f"Top detected goods:\n{snippets_df['canonical_good'].value_counts().head(10)}")

    # Save raw snippets
    snippets_df.to_csv(os.path.join(OUTPUT_DIR, "raw_snippets.csv"), index=False)

    # ── Step 4: Contextual Classification ────────────────────────────────────
    print("\nSTEP 4: Contextual Classification of Ambiguous Goods")
    print("-"*40)

    if not os.path.exists(MODEL_PATH) or retrain:
        # Check if labelled data exists
        if os.path.exists(LABELLED_SNIPPETS_PATH):
            print(f"Loading labelled snippets from: {LABELLED_SNIPPETS_PATH}")
            labelled_df = pd.read_csv(LABELLED_SNIPPETS_PATH)
            labelled_df = labelled_df[labelled_df["label"].notna()]
            labelled_df["label"] = labelled_df["label"].astype(int)
            classifier = train_classifier(labelled_df, model_path=MODEL_PATH)
        else:
            print("No labelled snippets found.")
            print("Creating labelling template...")
            create_labelling_template(snippets_df,
                                      output_path=os.path.join(OUTPUT_DIR, "snippets_to_label.csv"))
            print(f"\nPlease label the snippets in: outputs/snippets_to_label.csv")
            print("Then re-run with: python main.py --retrain")
            print("\nContinuing with unfiltered mentions (all treated as relevant)...")
            filtered_snippets = snippets_df.copy()
            filtered_snippets["classifier_label"] = 1
            classifier = None
    else:
        print(f"Loading existing classifier from: {MODEL_PATH}")
        classifier = load_classifier(MODEL_PATH)

    if classifier is not None:
        filtered_snippets = filter_relevant_snippets(snippets_df, classifier)
    else:
        filtered_snippets = snippets_df.copy()

    filtered_snippets.to_csv(
        os.path.join(OUTPUT_DIR, "filtered_snippets.csv"), index=False
    )

    # ── Step 5: Risk Scoring ─────────────────────────────────────────────────
    print("\nSTEP 5: Computing Company Risk Scores")
    print("-"*40)
    scored_df = compute_document_risk_scores(filtered_snippets, gov_docs)

    # ── Step 6: Results ──────────────────────────────────────────────────────
    print("\nSTEP 6: Results Summary")
    print("-"*40)

    # Top 20 companies
    top_companies = get_top_companies(scored_df, n=20)
    print("\nTop 20 Companies by Child Labour Risk Score:")
    print(top_companies.to_string())

    # Distribution stats
    stats = get_score_distribution_stats(scored_df)
    print("\nRisk Score Distribution:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Export
    generate_risk_report(scored_df)
    top_companies.to_csv(os.path.join(OUTPUT_DIR, "top_20_companies.csv"))
    print(f"\nTop 20 companies saved to: outputs/top_20_companies.csv")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("="*60)

    return scored_df, filtered_snippets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HACE Ethical Risk Scoring Pipeline"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the contextual classifier"
    )
    args = parser.parse_args()

    scored_df, filtered_snippets = run_full_pipeline(retrain=args.retrain)
