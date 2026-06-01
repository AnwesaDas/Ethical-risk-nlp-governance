"""
goods_detector.py
-----------------
Detects mentions of child-labour-linked goods in corporate governance documents
using spaCy PhraseMatcher with the goods synonym dictionary.

Returns canonical good form + surrounding 5-token context window per mention.

HACE: Data Changing Child Labour
MSc Data Science - University of Manchester (2024-2025)
"""

import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
from typing import List, Dict, Tuple
from src.goods_dictionary import GOODS_DICTIONARY, get_reverse_mapping

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Build reverse mapping once
REVERSE_MAP = get_reverse_mapping()

# Window size for context extraction (tokens either side of mention)
WINDOW_SIZE = 5


def build_phrase_matcher() -> PhraseMatcher:
    """
    Build a spaCy PhraseMatcher from the goods synonym dictionary.
    Case-insensitive matching using LOWER attribute.

    Returns:
        Configured PhraseMatcher object
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    for canonical, synonyms in GOODS_DICTIONARY.items():
        patterns = [nlp.make_doc(syn) for syn in synonyms]
        matcher.add(canonical, patterns)

    print(f"PhraseMatcher built with {len(GOODS_DICTIONARY)} goods categories.")
    return matcher


def extract_context_window(doc: spacy.tokens.Doc, start: int, end: int,
                            window: int = WINDOW_SIZE) -> str:
    """
    Extract surrounding context window around a matched span.

    Args:
        doc: spaCy Doc object
        start: Token index of match start
        end: Token index of match end
        window: Number of tokens to include either side

    Returns:
        Context window string
    """
    context_start = max(0, start - window)
    context_end = min(len(doc), end + window)
    return doc[context_start:context_end].text


def detect_goods_in_document(text: str, matcher: PhraseMatcher,
                              doc_id: int = None,
                              company_id: str = None) -> List[Dict]:
    """
    Run PhraseMatcher on a single document and return all detected goods
    with their context windows.

    Args:
        text: Document text string
        matcher: Configured PhraseMatcher
        doc_id: Optional document index
        company_id: Optional company identifier

    Returns:
        List of dicts with keys: doc_id, company_id, canonical_good, snippet
    """
    doc = nlp(text[:1000000])  # spaCy limit safety
    matches = matcher(doc)

    results = []
    for match_id, start, end in matches:
        canonical = nlp.vocab.strings[match_id]
        snippet = extract_context_window(doc, start, end)
        span_text = doc[start:end].text

        results.append({
            "doc_id": doc_id,
            "company_id": company_id,
            "canonical_good": canonical,
            "matched_text": span_text,
            "snippet": snippet,
            "token_start": start,
            "token_end": end
        })

    return results


def detect_goods_in_corpus(df: pd.DataFrame,
                            text_col: str = "text",
                            company_col: str = "company_id") -> pd.DataFrame:
    """
    Run goods detection across entire corpus of governance documents.

    Args:
        df: DataFrame with document texts
        text_col: Column name containing document text
        company_col: Column name containing company identifier

    Returns:
        DataFrame of all detected goods mentions with context snippets
    """
    matcher = build_phrase_matcher()
    all_results = []

    print(f"Detecting goods across {len(df)} documents...")

    for idx, row in df.iterrows():
        text = str(row.get(text_col, ""))
        company_id = row.get(company_col, f"company_{idx}")

        if not text.strip():
            continue

        results = detect_goods_in_document(
            text=text,
            matcher=matcher,
            doc_id=idx,
            company_id=company_id
        )
        all_results.extend(results)

    snippets_df = pd.DataFrame(all_results)
    print(f"Total goods mentions detected: {len(snippets_df)}")
    return snippets_df


if __name__ == "__main__":
    # Test on sample text
    sample_texts = [
        {
            "company_id": "TestCo_001",
            "text": """
            Our supply chain involves sourcing gold from certified mines in West Africa.
            We also procure cocoa from Ivory Coast and cotton from India.
            The company uses rubber in several of its manufacturing processes.
            Our gold supply chain is rigorously audited for human rights compliance.
            We remain committed to eliminating child labour from our operations.
            """
        },
        {
            "company_id": "TestCo_002",
            "text": """
            Cobalt is a critical material in our battery supply chain.
            We source timber responsibly from certified forests.
            Our products include smartphones and electric vehicles.
            """
        }
    ]

    df = pd.DataFrame(sample_texts)
    matcher = build_phrase_matcher()

    snippets_df = detect_goods_in_corpus(df)
    print("\nSample detections:")
    print(snippets_df[["company_id", "canonical_good", "matched_text", "snippet"]].head(10))
