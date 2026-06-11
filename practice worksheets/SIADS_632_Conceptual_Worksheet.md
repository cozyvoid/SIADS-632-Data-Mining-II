---
title: "SIADS 632 Data Mining II - Conceptual Worksheet"
subtitle: "Recall, Explanation, and Short-Answer Practice"
author: "Prepared for Sam"
date: "Summer 2026"
geometry: margin=0.75in
fontsize: 11pt
papersize: letter
---

# Instructions

Use this worksheet for active recall. Try to answer without looking at the study guide. Then check your answers and rewrite weak areas in your own words.

# Unit 1: Sequences, N-Grams, Markov Chains, and HMMs

## 1. Define the following terms.

Sequence:

\vspace{0.6in}

N-gram:

\vspace{0.6in}

Vocabulary:

\vspace{0.6in}

Sentence boundary token:

\vspace{0.6in}

First-order Markov assumption:

\vspace{0.8in}

Hidden state:

\vspace{0.6in}

Emission probability:

\vspace{0.6in}

Transition probability:

\vspace{0.6in}

## 2. Explain why a bigram model is a first-order Markov model.

\vspace{1.0in}

## 3. For the formula below, explain the numerator and denominator in words.

\[
P(w_i \mid w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}
\]

\vspace{1.1in}

## 4. What problem do `<s>` and `</s>` solve in an N-gram model?

\vspace{1.0in}

## 5. Compare Markov chains and Hidden Markov Models.

| Feature | Markov chain | Hidden Markov Model |
|---|---|---|
| State visibility | | |
| Example | | |
| Main probabilities | | |
| Typical task | | |

\newpage

# Unit 2: Time Series Patterns and Similarity

## 6. Define the following time series components.

Trend:

\vspace{0.7in}

Seasonality:

\vspace{0.7in}

Cycle:

\vspace{0.7in}

Irregular component:

\vspace{0.7in}

Outlier:

\vspace{0.7in}

## 7. Explain the difference between an additive and multiplicative decomposition.

\vspace{1.2in}

## 8. Compare SMA, WMA, and EMA.

| Method | How weights work | Strength | Weakness |
|---|---|---|---|
| Simple moving average | | | |
| Weighted moving average | | | |
| Exponential moving average | | | |

## 9. When would DTW be better than Euclidean distance for time series comparison?

\vspace{1.1in}

## 10. Fill in the comparison table.

| Similarity/distance | Best for | Main limitation |
|---|---|---|
| Euclidean distance | | |
| Cosine similarity | | |
| DTW | | |
| Edit distance | | |

\newpage

# Unit 3: Forecasting

## 11. Define stationarity.

\vspace{1.0in}

## 12. Why do we difference a time series?

\vspace{1.0in}

## 13. Explain log returns in words.

\vspace{1.0in}

## 14. Compare ACF and PACF.

| Tool | What it measures | Why it is useful |
|---|---|---|
| ACF | | |
| PACF | | |

## 15. Explain each ARIMA parameter.

| Parameter | Meaning |
|---|---|
| p | |
| d | |
| q | |

## 16. Compare AR, ARMA, ARIMA, and VAR.

| Model | Uses past values? | Uses past errors? | Handles multiple series? |
|---|---|---|---|
| AR | | | |
| ARMA | | | |
| ARIMA | | | |
| VAR | | | |

## 17. Why is Granger causality not the same as true causal inference?

\vspace{1.2in}

\newpage

# Unit 4: Data Streams

## 18. List four restrictions of data streams.

1. \vspace{0.25in}
2. \vspace{0.25in}
3. \vspace{0.25in}
4. \vspace{0.25in}

## 19. What is a synopsis structure?

\vspace{0.9in}

## 20. Compare stream algorithms.

| Method | Main purpose | Output | Main tradeoff |
|---|---|---|---|
| Random sampling | | | |
| Reservoir sampling | | | |
| Bloom filter | | | |
| Lossy counting | | | |

## 21. Explain why reservoir sampling treats early and late items fairly.

\vspace{1.2in}

## 22. Bloom filter interpretation.

If a Bloom filter says an item is not present, what does that mean?

\vspace{0.6in}

If a Bloom filter says an item is present, what does that mean?

\vspace{0.6in}

Which error type is possible?

\vspace{0.4in}

Which error type is not expected in a standard insertion-only Bloom filter?

\vspace{0.4in}

## 23. Explain lossy counting in one paragraph.

\vspace{1.4in}

\newpage

# Mixed Oral Exam Practice

Answer each in 2-4 sentences.

1. Why does an unseen N-gram receive probability zero under MLE?

\vspace{0.9in}

2. Why should time series train/test splits be chronological?

\vspace{0.9in}

3. What does a high lag-1 autocorrelation suggest?

\vspace{0.9in}

4. Why might VAR outperform independent AR models?

\vspace{0.9in}

5. Why are stream mining algorithms often approximate?

\vspace{0.9in}

