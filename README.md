# 🔍 Ethical Risk Scoring in Corporate Governance Documents

An NLP pipeline developed in collaboration with **HACE: Data Changing Child Labour** to automatically detect and score child labour risk exposure in corporate governance documents using Named Entity Recognition, contextual classification, and interpretable risk scoring.

---

## 🎯 Project Overview

Corporate governance documents are a rich but underutilised source for detecting supply chain risks related to child labour. This project builds a scalable, interpretable NLP pipeline that:

- Detects mentions of high-risk goods (e.g. gold, cobalt, cotton) in corporate disclosures
- Disambiguates contextually ambiguous terms using a supervised classifier
- Scores companies based on unique high-risk goods mentioned in their governance documents
- Produces ranked, exportable results for use by HACE analysts

> **Industry Partner:** HACE: Data Changing Child Labour  
> **Academic Context:** MSc Data Science — University of Manchester (2024–2025)  
> **Module:** DATA70202 Applying Data Science

---

## 🏗️ Pipeline Architecture

```
Corporate Governance Documents
        ↓
Text Preprocessing
(Tokenisation → Stopword Removal → Lemmatisation)
        ↓
Goods Detection
(spaCy PhraseMatcher + Synonym Dictionary)
        ↓
Contextual Classifier
(TF-IDF + Logistic Regression — disambiguates ambiguous goods)
        ↓
Risk Signal Derivation
(Unique goods per document → Normalised 0–100 score)
        ↓
Company Risk Rankings (CSV Export)
```

---

## ✨ Key Features

- **Domain-specific goods dictionary** — maps UK/US synonyms to canonical forms (e.g. Aubergine ↔ Eggplant)
- **spaCy PhraseMatcher** — fast, token-level phrase matching at scale across hundreds of documents
- **Contextual disambiguation** — TF-IDF + Logistic Regression classifier achieves **94% accuracy** in resolving ambiguous goods (e.g. "gold" the metal vs "gold" the colour)
- **5-token window analysis** — surrounding context extracted per mention for classification
- **Unique goods scoring** — normalised 0–100 risk score based on distinct high-risk goods per document (not raw frequency — avoids boilerplate bias)
- **Downstream goods integration** — maps raw materials to consumer products (e.g. cobalt → smartphones)
- **Exploratory Data Analysis** — visualisations across goods, countries, sectors, and sub-industries
- **CSV export** — ranked company risk scores ready for analyst use

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| NLP Pipeline | spaCy (en_core_web_sm), PhraseMatcher |
| Classification | Scikit-learn (TF-IDF + Logistic Regression) |
| Text Processing | NLTK, Lemmatisation, Stopword Removal |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Language | Python 3.10+ |

---

## 📁 Dataset Description

Three datasets were used:

| Dataset | Description |
|---|---|
| **Goods Dataset** | 169 goods linked to child/forced labour from U.S. Department of Labor |
| **Downstream Goods Dataset** | Maps raw materials (e.g. palm oil) to consumer products (e.g. cosmetics) |
| **Governance Documents** | Corporate disclosure reports from 189 companies provided by HACE |

---

## 🔬 Methodology

### 1. Text Preprocessing
- **Tokenisation** — spaCy sentence and word tokenisation
- **Stopword Removal** — customised list retaining legally significant stopwords
- **Lemmatisation** — spaCy POS-aware lemmatisation with manual exceptions for ambiguous terms like "lead"

### 2. Goods Detection
- Built comprehensive synonym dictionary mapping 169+ goods to canonical UK-English forms
- Used spaCy `PhraseMatcher` (case-insensitive) for efficient multi-token phrase detection
- Returned canonical good + surrounding 5-token context window per mention

### 3. Contextual Classification
- **Problem:** Ambiguous goods like "gold", "silver", "rubber" have non-commodity meanings
- **Solution:** Trained Logistic Regression on TF-IDF vectors of 5-token context windows
- **Labels:** Manually labelled snippets as Relevant (commodity) or Irrelevant (non-commodity)
- **Result:** 94% accuracy, 0.94 precision/recall/F1 after removing "lead" (overwhelmingly used as verb)
- Filtered out 216 irrelevant mentions from 3,708 total detections

### 4. Risk Scoring
- Score = linear scaling of **unique relevant goods** per document (0–100)
- Unique goods used instead of total mentions to avoid boilerplate repetition bias
- Scores act as **screening signals** — not definitive indictments — for further human investigation

---

## 📊 Key Results

- **3,708** total risky goods mentions detected across 329 governance documents
- **216** irrelevant mentions filtered by contextual classifier
- **3,492** contextually relevant mentions used for scoring
- Top-scoring companies: Watches of Switzerland Group PLC, Greencore Group (score: 100/100)
- Most represented sector: **Consumer Staples** (packaged foods, beverages)
- Highest risk goods: **Gold, Bricks, Sugarcane, Cotton, Cocoa**
- Highest risk countries: **India, Paraguay, Brazil, Pakistan, Burma**

---

## 🧠 Model Performance

### Contextual Classifier (excl. "Lead")

| Metric | Score |
|---|---|
| Accuracy | **0.94** |
| Precision | 0.94 |
| Recall | 0.94 |
| F1-Score | 0.94 |

Significantly outperforms manual keyword filtering baseline.

---

## 💡 Key Design Decisions

- **Unique vs total mentions** — Avoids inflating scores for longer documents with repetitive boilerplate language
- **Dropping "lead"** — 670 mentions, only 6 referred to the metal. Removing it dramatically improved classifier performance
- **Human-in-the-loop** — High-confidence detections are automated; borderline cases flagged for analyst review
- **Modular pipeline** — Designed to extend to forced labour, environmental violations, or other ESG risks

---

## 🔮 Future Work

- [ ] Fine-tune transformer-based NER model (e.g. BERT) for higher entity recall
- [ ] Integrate LLM-based contextual analysis for more nuanced disambiguation
- [ ] Add temporal analysis to track risk evolution across disclosure years
- [ ] Build interactive dashboard for HACE analysts
- [ ] Extend to forced labour and environmental risk signals

---

## 👥 Team

Collaborative MSc project — University of Manchester (2024–2025):
- Anwesa Das
- Ben Zubier
- Esteban G. Russi
- Zhen Fu
- Dena Shirzad

**Industry Partner:** HACE: Data Changing Child Labour

---

## 👩‍💻 Author

**Anwesa Das** — MLOps & AI Engineer  
[GitHub](https://github.com/AnwesaDas) | [LinkedIn](https://linkedin.com/in/anwesa-das)

---

## 📄 License

MIT License
