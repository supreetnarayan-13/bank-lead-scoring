# Bank Lead Scoring App

**Live app:** https://bank-lead-scoring-m8q9xae9w2vrzfpf7lhgdh.streamlit.app/

A machine learning app that ranks bank customers by how likely they are to say yes to a term deposit offer, so the bank can call the most promising customers first.

## Problem
A bank phones customers to offer a term deposit (a savings product). About 88% say no, and every call costs time and money. Which customers should the bank call first, and how many calls are worth making?

## Data
UCI Bank Marketing dataset (`bank-full.csv`), about 45,000 phone calls.
Source: Moro, Cortez and Rita (2014), UCI Machine Learning Repository. Used with attribution.

## Approach
- **Removed `duration`** (call length). It is only known after the call ends, so using it would be data leakage and give unrealistically high scores.
- **Time-based split.** Trained on the older 80% of rows and tested on the newer 20%, instead of a random split. The file has no date column, so the order is taken from row position.
- **Model:** gradient boosting (scikit-learn `HistGradientBoostingClassifier`) inside a Pipeline with one-hot encoding.
- **Metrics:** ROC-AUC, PR-AUC, gain and lift. Accuracy is not used, because always predicting "no" already gives about 88%.
- **Cost-based call planner:** chooses the number of calls that maximises profit, from adjustable cost and profit values.
- **Ask page:** Python calculates conversion rates, and Claude (Anthropic API) answers questions using only those numbers.

## Results (tested on the newer 20% of calls)
| Metric | Value |
|---|---|
| ROC-AUC | 0.676 |
| PR-AUC | 0.467 |
| Calling the top 10% of ranked customers | finds 16% of all yes (1.6x lift) |
| Calling the top 20% of ranked customers | finds 33% of all yes (1.6x lift) |
| Calling the top 30% of ranked customers | finds 48% of all yes (1.6x lift) |

Lift of 1.6x means the ranking finds 1.6 times more buyers than calling customers at random.

## App pages
| Page | What it does |
|---|---|
| Dashboard | Conversion rate by job, month, education and more |
| Call planner | Sliders for cost per call and profit per sale; shows the best number of calls |
| Score a lead | Enter one customer's details and get a conversion score |
| Ask | Ask questions in plain English and get answers from Claude |

## Limitations
- **The data changes over time.** The share of customers who said yes was 6.7% in the training data and 31.6% in the test data. The model learned from older calls and is tested on newer calls that behave very differently, so performance is moderate (ROC-AUC 0.68, 1.6x lift). A random split would give higher scores, but those scores would be less realistic.
- The data has no date column, so time order is assumed from row position.
- Cost per call and profit per sale in the call planner are adjustable assumptions, not real bank figures.
- `month` can reflect seasonality that may not hold in other years.
- The Score a lead page keeps some fields fixed (such as day and campaign).
- The Ask page uses API credit, so it may show an error if credit runs out.

## Tech
Python, pandas, NumPy, scikit-learn, joblib, Streamlit, Anthropic Claude API, Git/GitHub, Streamlit Community Cloud.