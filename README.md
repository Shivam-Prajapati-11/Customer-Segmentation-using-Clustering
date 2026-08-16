# Customer Segmentation with RFM & K-Means

A portfolio data-science project that turns 540,000+ e-commerce transaction rows
into clear, named customer segments (with a marketing action for each) using
**RFM analysis** and **K-Means clustering**.

---

## What this project does

Analyses a retailer's raw purchase history and automatically groups the customers
into segments that behave differently. Concretely it:

- Cleans messy transaction data (cancelled orders, missing IDs, invalid rows)
- Summarises every customer into **RFM** features — **Recency** (days since last
  purchase), **Frequency** (how often they buy), and **Monetary** (total spent)
- Uses the **Elbow method** and **Silhouette score** to decide how many natural
  groups exist
- Groups customers with **K-Means clustering** and verifies that the grouping is
  stable across runs (Adjusted Rand Index)
- Labels each group with a human persona — **Champions, Loyal, At-Risk,
  New/Low-Value** — and suggests a marketing action for each

---

## Why this project is useful

When you only have raw purchase data, it's impossible to tell who your best
customers are, who is drifting away, or who should get a reactivation offer.

This project makes that visible: instead of treating all customers alike, you get
a clear picture of **4 distinct segments** that behave very differently — and a
recommended action for each one:

| Segment | Share | What it means / action |
|---|---|---|
| **Champions** | ~3% | Top, frequent spenders → reward & retain |
| **Loyal** | ~13% | Steady repeat buyers → nurture into Champions |
| **At-Risk** | ~24% | Went quiet → reactivation campaign |
| **New/Low-Value** | ~60% | Low spend / newer → onboarding & intro offers |

For a data-science learner it's also a complete, ready-to-read example of an
unsupervised **customer segmentation** pipeline — from raw data to business value.

---

## Getting started

### Prerequisites
- **Python 3.12+**
- The packages below (install with one command)

### Install
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

### Run it
```bash
python customer_segmentation.py
```
The **Online Retail** dataset (UCI / Kaggle) is downloaded automatically on the
first run — no manual download needed, just make sure you're online that first time.

### Repository layout
```
customer_segmentation.py     # the full project: code, comments and results
README.md                    # this file
```

---

## Getting help

- Read the comments in `customer_segmentation.py` — every step is explained in
  plain language as you follow the script top to bottom.
- Open an **issue** on this repository if something isn't working or is unclear.
- If the dataset fails to download, check your internet connection / Kaggle
  credentials; the script can be pointed at a locally saved `data.csv` instead.

---

## Maintainers & contributing

This project is maintained by **Shiva** (3rd-year B.Tech Computer Engineering
student) as a portfolio project.

Contributions, suggestions, and improvements are welcome:
- **Reporting bugs** → open an issue
- **Improving the project** → fork the repo, make your changes, and open a pull
  request describing what you changed and why

If you found this useful, a ⭐ is always appreciated!
