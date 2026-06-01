"""
eda.py
------
Exploratory Data Analysis for the HACE child labour risk project.
Generates visualisations for goods, countries, sectors, and company distributions.

HACE: Data Changing Child Labour
MSc Data Science - University of Manchester (2024-2025)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from wordcloud import WordCloud
import os
from typing import Optional

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
FIGURE_SIZE = (14, 7)
COLOR_PALETTE = "YlOrRd"
OUTPUT_DIR = "outputs/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_goods_heatmap(goods_df: pd.DataFrame,
                       good_col: str = "good",
                       child_labour_col: str = "child_labour",
                       forced_col: str = "forced_child_labour",
                       top_n: int = 20,
                       save: bool = True) -> None:
    """
    Heatmap of top N goods by child labour violation frequency.
    Mirrors Figure 1 from the project report.
    """
    goods_df = goods_df.copy()
    goods_df["total"] = goods_df[child_labour_col] + goods_df[forced_col]
    top_goods = goods_df.nlargest(top_n, "total")

    heatmap_data = top_goods.set_index(good_col)[[child_labour_col, forced_col]]
    heatmap_data.columns = ["Child Labour", "Forced Child Labour"]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap=COLOR_PALETTE,
                linewidths=0.5, ax=ax)
    ax.set_title(f"Top {top_n} Goods Most Frequently Associated with Child Labour",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Good", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig1_goods_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_countries_heatmap(countries_df: pd.DataFrame,
                            country_col: str = "country",
                            child_labour_col: str = "child_labour",
                            forced_col: str = "forced_child_labour",
                            top_n: int = 20,
                            save: bool = True) -> None:
    """
    Heatmap of top N countries by child labour violation frequency.
    Mirrors Figure 2 from the project report.
    """
    countries_df = countries_df.copy()
    countries_df["total"] = countries_df[child_labour_col] + countries_df[forced_col]
    top_countries = countries_df.nlargest(top_n, "total")

    heatmap_data = top_countries.set_index(country_col)[[child_labour_col, forced_col]]
    heatmap_data.columns = ["Child Labour", "Forced Child Labour"]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="Blues",
                linewidths=0.5, ax=ax)
    ax.set_title(f"Top {top_n} Countries Associated with Child Labour Violations",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig2_countries_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_downstream_goods(downstream_df: pd.DataFrame,
                           good_col: str = "downstream_good",
                           input_col: str = "input_good",
                           top_n: int = 20,
                           save: bool = True) -> None:
    """
    Bar chart of child-labour-linked downstream goods.
    Mirrors Figure 3 from the project report.
    """
    top_downstream = downstream_df[good_col].value_counts().head(top_n)
    colours = cm.tab20(np.linspace(0, 1, len(top_downstream)))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(range(len(top_downstream)), top_downstream.values,
                  color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(top_downstream)))
    ax.set_xticklabels(top_downstream.index, rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Top {top_n} Downstream Goods Linked to Child Labour",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Associations", fontsize=11)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig3_downstream_goods.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_word_frequency(corpus_text: str,
                         top_n: int = 20,
                         save: bool = True) -> None:
    """
    Bar chart of most frequent lemmatised words in governance documents.
    Mirrors Figure 6 from the project report.
    """
    from collections import Counter
    words = corpus_text.lower().split()
    word_counts = Counter(words).most_common(top_n)
    words_list = [w for w, _ in word_counts]
    counts = [c for _, c in word_counts]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.barh(words_list[::-1], counts[::-1],
            color=sns.color_palette("Blues_r", top_n))
    ax.set_title(f"Top {top_n} Most Frequent Words in Governance Documents",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Frequency", fontsize=11)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig6_word_frequency.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_wordcloud(corpus_text: str, save: bool = True) -> None:
    """
    Word cloud of governance document corpus.
    Mirrors Figure 7 from the project report.
    """
    wordcloud = WordCloud(
        width=1400, height=700,
        background_color="white",
        colormap="YlOrRd",
        max_words=200,
        collocations=False
    ).generate(corpus_text)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — Governance Document Corpus",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig7_wordcloud.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_risk_score_distribution(scored_df: pd.DataFrame,
                                  score_col: str = "risk_score",
                                  save: bool = True) -> None:
    """
    Distribution of child labour risk scores across governance documents.
    Mirrors Figure 12 from the project report.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.hist(scored_df[score_col], bins=20, color="#E74C3C",
            edgecolor="white", linewidth=0.8, alpha=0.85)
    ax.axvline(scored_df[score_col].mean(), color="navy",
               linestyle="--", linewidth=1.5, label=f"Mean: {scored_df[score_col].mean():.1f}")
    ax.axvline(scored_df[score_col].median(), color="darkgreen",
               linestyle="-.", linewidth=1.5, label=f"Median: {scored_df[score_col].median():.1f}")
    ax.set_title("Distribution of Child Labour Risk Scores",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Risk Score (0-100)", fontsize=11)
    ax.set_ylabel("Number of Documents", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig12_risk_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_sector_mentions(gov_docs_df: pd.DataFrame,
                          sector_col: str = "sector",
                          save: bool = True) -> None:
    """
    Bar chart of child labour mentions by sector.
    Mirrors Figure 9 from the project report.
    """
    sector_counts = gov_docs_df.groupby(sector_col)["risk_score"].agg(
        ["count", "mean"]
    ).sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colours = sns.color_palette("Set2", len(sector_counts))
    ax.bar(range(len(sector_counts)), sector_counts["mean"],
           color=colours, edgecolor="white")
    ax.set_xticks(range(len(sector_counts)))
    ax.set_xticklabels(sector_counts.index, rotation=45, ha="right", fontsize=9)
    ax.set_title("Average Child Labour Risk Score by Sector",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Risk Score", fontsize=11)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig9_sector_mentions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_top_companies(scored_df: pd.DataFrame,
                        company_col: str = "company_name",
                        score_col: str = "risk_score",
                        top_n: int = 20,
                        save: bool = True) -> None:
    """
    Bar chart of top N highest scoring companies.
    Mirrors Table 3 visualisation from the project report.
    """
    top = scored_df.nlargest(top_n, score_col)

    fig, ax = plt.subplots(figsize=(12, 8))
    colours = cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top)))
    bars = ax.barh(range(len(top)), top[score_col].values,
                   color=colours, edgecolor="white")

    labels = top[company_col].values if company_col in top.columns else top.index
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Companies by Child Labour Risk Score",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Risk Score (0-100)", fontsize=11)
    ax.axvline(60, color="red", linestyle="--", alpha=0.5, label="Score = 60")
    ax.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "top_companies_risk.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("EDA module loaded. Call individual plot functions with your data.")
    print("Available plots:")
    print("  - plot_goods_heatmap(goods_df)")
    print("  - plot_countries_heatmap(countries_df)")
    print("  - plot_downstream_goods(downstream_df)")
    print("  - plot_word_frequency(corpus_text)")
    print("  - plot_wordcloud(corpus_text)")
    print("  - plot_risk_score_distribution(scored_df)")
    print("  - plot_sector_mentions(gov_docs_df)")
    print("  - plot_top_companies(scored_df)")
