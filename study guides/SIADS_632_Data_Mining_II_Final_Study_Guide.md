---
title: "SIADS 632 Data Mining II - Final Study Guide"
subtitle: "Course- and Assignment-Aligned Review for Concepts and Python Implementation"
author: "Prepared for Sam"
date: "Summer 2026"
geometry: margin=0.7in
fontsize: 10pt
papersize: letter
---

# How to Use This Guide

This guide is aligned to the SIADS 632 lecture sequence and the uploaded assignments. It emphasizes both conceptual understanding and Python implementation because the course work is implementation-heavy. The most important exam skill is not just knowing definitions, but recognizing which representation, formula, library function, or data structure fits the problem.

Recommended use:

1. Read the unit summary.
2. Review the formulas and code patterns.
3. Practice explaining the topic out loud in one or two sentences.
4. Use the worksheets separately for recall and implementation practice.

Core packages that appear repeatedly:

```python
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
```

Frequently used course libraries:

```python
import nltk
from nltk.util import ngrams, pad_both_ends
from nltk.lm import MLE
from nltk.tag.hmm import HiddenMarkovModelTagger

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
```

# Course Map

| Unit | Main representation | Main tasks | Key methods |
|---|---|---|---|
| 1 | Sequences | next-token prediction, POS tagging | N-grams, Markov chains, HMMs |
| 2 | Time series patterns | decomposition, smoothing, similarity | trend, seasonality, WMA, EMA, DTW |
| 3 | Time series forecasting | predict future values | stationarity, ACF/PACF, ARMA/ARIMA, VAR, Granger |
| 4 | Data streams | mine continuous data with limited memory | random sampling, reservoir sampling, Bloom filters, lossy counting |

# Unit 1: Sequences, N-Grams, Markov Chains, and HMMs

## 1.1 Sequences

A sequence is an ordered list of symbols, tokens, or events. Order matters.

Examples:

```python
["I", "love", "data", "mining"]
["home", "search", "product", "cart", "checkout"]
["A", "C", "G", "T"]
```

A sequence mining problem usually asks one of the following:

- What token/event is likely to come next?
- How likely is this sequence under a model?
- Which hidden structure produced this observed sequence?
- How similar are two ordered sequences?

## 1.2 Assignment 1 Part 1: Loading Shakespeare Sonnet Lines

In the assignment, a sentence is defined as one line of a sonnet, not a grammatical sentence. The sonnet boundaries are ignored. The expected number of sentences is 2155.

Important requirements:

- Return a `list` of tokenized sentences.
- Each sentence should be a `list` of tokens.
- Lowercase alphabetic tokens.
- Drop newline characters.
- Do not include raw `"\n"` as a token.

Canonical pattern:

```python
def load_data():
    sentences = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.lower().split()
            sentences.append(tokens)

    return sentences
```

Hidden-test risk: if a file includes title text, Roman numerals, or blank lines, you may need to filter them according to the assignment's data format. The autograder expects exactly 2155 sentence lines.

## 1.3 Vocabulary Construction

The vocabulary is the set of unique tokens plus sentence boundary symbols.

Required special tokens:

```python
"<s>"
"</s>"
```

Pattern:

```python
def build_vocab(sentences):
    vocab = set()

    for sent in sentences:
        vocab.update(sent)

    vocab.add("<s>")
    vocab.add("</s>")

    return list(vocab)
```

Conceptual explanation:

The vocabulary defines the possible tokens the language model can generate or assign probability to. Boundary tokens are included because N-gram models need to learn how sentences start and end.

## 1.4 N-Grams

An N-gram is a contiguous subsequence of length `n`.

Example:

```python
tokens = ["I", "love", "data", "mining"]

unigrams = list(zip(tokens))
bigrams = list(zip(tokens, tokens[1:]))
trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
```

Output:

```python
[("I",), ("love",), ("data",), ("mining",)]
[("I", "love"), ("love", "data"), ("data", "mining")]
[("I", "love", "data"), ("love", "data", "mining")]
```

Using NLTK with padding:

```python
from nltk.util import ngrams, pad_both_ends

def build_ngrams(n, sentences):
    all_ngrams = []

    for sent in sentences:
        if n >= 2:
            padded = list(pad_both_ends(sent, n=n))
        else:
            padded = sent

        all_ngrams.append(list(ngrams(padded, n)))

    return all_ngrams
```

For a sentence of length `l`, padding with `n - 1` tokens on both sides produces `l + n - 1` N-grams.

## 1.5 N-Gram Probabilities

Unigram probability:

\[
P(w_i) = \frac{count(w_i)}{N}
\]

Bigram probability:

\[
P(w_i \mid w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}
\]

Trigram probability:

\[
P(w_i \mid w_{i-2}, w_{i-1}) = \frac{count(w_{i-2}, w_{i-1}, w_i)}{count(w_{i-2}, w_{i-1})}
\]

The denominator is the count of the context.

Example explanation:

If computing `P(like | I)`, the denominator is `count(I)`, not `count(like)`, because the question is: among all times `I` appeared, how often was it followed by `like`?

## 1.6 Assignment 1 Part 2: Guessing the Next Token

The assignment asks for the most likely token after a starting context such as `("<s>", "<s>", "<s>")` under a bigram model.

A bigram model is a first-order Markov chain: the next token depends only on the current token, not the full history. Therefore, with bigrams, only the final token of the context is relevant.

Pattern:

```python
from collections import Counter

def bigram_next_token(start_tokens=("<s>",) * 3):
    sentences = load_data()
    all_bigrams = build_ngrams(2, sentences)

    bigram_counts = Counter()
    context_counts = Counter()

    for sent_grams in all_bigrams:
        for w1, w2 in sent_grams:
            bigram_counts[(w1, w2)] += 1
            context_counts[w1] += 1

    context = start_tokens[-1]
    candidates = []

    for (w1, w2), count in bigram_counts.items():
        if w1 == context:
            prob = count / context_counts[w1]
            candidates.append((w2, prob))

    return max(candidates, key=lambda x: x[1])
```

## 1.7 Training an N-Gram Language Model with NLTK

The assignment uses `nltk.lm.MLE`.

Pattern:

```python
from nltk.lm import MLE

def train_ngram_lm(n):
    sentences = load_data()
    vocab = build_vocab(sentences)
    train_data = build_ngrams(n, sentences)

    lm = MLE(n)
    lm.fit(train_data, vocabulary_text=vocab)

    return lm
```

Key idea:

MLE estimates probabilities directly from observed frequencies. Without smoothing, unseen N-grams receive probability zero.

## 1.8 Markov Chains

A Markov chain models transitions among observable states.

Markov assumption:

\[
P(X_t \mid X_{t-1}, X_{t-2}, ..., X_1) = P(X_t \mid X_{t-1})
\]

This means the next state depends only on the current state.

Transition matrix example:

```python
import numpy as np

P = np.array([
    [0.8, 0.2],
    [0.4, 0.6]
])

current = np.array([1.0, 0.0])
next_dist = current @ P
three_steps = current @ np.linalg.matrix_power(P, 3)
```

Rows usually represent the current state and columns represent the next state. Each row should sum to 1.

## 1.9 Hidden Markov Models

A Hidden Markov Model, or HMM, is a sequence model where the true state is hidden and the observations are visible.

Components:

| Component | Meaning |
|---|---|
| hidden states | unobserved labels or states |
| observations | visible tokens or emissions |
| initial probabilities | probability of starting in each state |
| transition probabilities | probability of moving from one hidden state to another |
| emission probabilities | probability of observing a token given a hidden state |

POS tagging example:

- Observed sequence: words
- Hidden sequence: part-of-speech tags

Classic HMM questions:

| Task | Question | Common algorithm |
|---|---|---|
| Evaluation | How likely is this observation sequence? | forward algorithm |
| Decoding | What hidden state sequence is most likely? | Viterbi algorithm |
| Learning | How do we estimate model parameters? | Baum-Welch / EM |

## 1.10 Assignment 1 Part 3: CoNLL Data and HMM Tagger

The assignment loads CoNLL-2003 data.

Important requirements:

- Return a dictionary with train/dev/test sets.
- Train and dev contain labeled `(token, POS)` tuples.
- Test contains token strings if labels are not included.
- Remove document markers like `-DOCSTART-`.
- Split sentences on blank lines.
- Ignore sentences shorter than the minimum length required by the assignment.

Training pattern:

```python
from nltk.tag.hmm import HiddenMarkovModelTagger

def train_hmm(train, dev):
    tagger = HiddenMarkovModelTagger.train(train, test_sequence=dev)
    return tagger
```

Conceptual answer:

An HMM POS tagger treats words as observed emissions and POS tags as hidden states. The model learns which tags tend to follow other tags and which words tend to be emitted by each tag.

# Unit 2: Time Series Patterns and Similarity

## 2.1 Time Series Data

A time series is a sequence of observations indexed by time.

In assignments 2 and 3, the central dataset is COVID-19 confirmed cases. The original data is cumulative, but the assignment asks for daily new cases.

Loading pattern:

```python
def load_data():
    df = pd.read_csv(path)

    date_cols = df.columns[4:]
    cumulative = df[date_cols].sum(axis=0)
    cumulative.index = pd.to_datetime(cumulative.index)

    daily_new = cumulative.diff().dropna().astype(float)

    return daily_new
```

Assignment expectations for the global series:

- Return `pd.Series`.
- Length 212.
- Index is `pd.DatetimeIndex`.
- Time range is 2020-01-23 through 2020-08-21.
- No missing values.
- Float dtype.

For top-five-country data:

- Return `pd.DataFrame`.
- Shape `(212, 5)`.
- Columns are the top five countries by cumulative cases as of 2020-08-21.
- Columns sorted descending by cumulative cases.
- Index is `pd.DatetimeIndex`.
- Values are daily new cases.

## 2.2 Discrete vs Continuous Time Series

A discrete time series is observed at separate time points, such as daily cases or monthly sales.

A continuous time series is conceptually evolving continuously, such as a physiological signal or movement trajectory, even if sampled digitally.

Most assignment data is discrete-time because cases are reported by date.

## 2.3 Time Series Mining Tasks

Common tasks:

| Task | Purpose |
|---|---|
| decomposition | separate trend, seasonal, residual components |
| smoothing | reduce noise and reveal pattern |
| similarity search | compare series |
| alignment | match patterns shifted in time |
| forecasting | predict future values |
| anomaly detection | find unusual points or subsequences |

## 2.4 Seasonal Decomposition

Seasonal decomposition separates a series into components.

Additive model:

\[
Y_t = Trend_t + Seasonal_t + Residual_t
\]

Multiplicative model:

\[
Y_t = Trend_t \times Seasonal_t \times Residual_t
\]

Assignment pattern:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

def sea_decomp(ser, model="additive"):
    return seasonal_decompose(ser, model=model)
```

For a DataFrame, apply decomposition column by column and collect seasonal components:

```python
def sea_decomp(df, model="additive"):
    seasonal = pd.DataFrame(index=df.index)

    for col in df.columns:
        result = seasonal_decompose(df[col], model=model)
        seasonal[col] = result.seasonal

    return seasonal.astype(float)
```

## 2.5 Trend Curves

A trend is the long-term direction of a series. A polynomial trend curve can model nonlinear trend.

Pattern:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def fit_trend(ser, degree):
    x = np.arange(len(ser)).reshape(-1, 1)
    y = ser.values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(X_poly, y)

    return model.predict(X_poly)
```

Assignment expectation: return a one-dimensional `np.ndarray` with the same length as the input series.

## 2.6 Moving Averages

A simple moving average smooths a series by averaging values within a rolling window.

\[
MA_t = \frac{Y_t + Y_{t-1} + ... + Y_{t-k+1}}{k}
\]

Python:

```python
ser.rolling(window=7).mean()
```

Interpretation:

A larger window produces a smoother curve but responds more slowly to changes. A smaller window is more responsive but noisier.

## 2.7 Weighted Moving Average

A weighted moving average gives different weights to values in a window.

Assignment pattern:

```python
def calc_wma(ser, wd_size, weights=1):
    if np.isscalar(weights):
        weights = np.ones(wd_size)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    ans = ser.rolling(wd_size).apply(
        lambda x: np.sum(x * weights),
        raw=True
    )

    return ans.to_numpy(dtype=float)
```

Potential hidden-test issue: this returns NaN for the first `wd_size - 1` positions unless the assignment expects a custom partial-window method. Check assignment instructions and tests carefully.

## 2.8 Exponential Moving Average

An exponential moving average gives more weight to recent observations.

One common form:

\[
EMA_t = \alpha Y_t + (1 - \alpha) EMA_{t-1}
\]

Course assignment uses a lambda-style time EMA where `lambda = 0` should return the cumulative moving average.

Standard pandas EMA:

```python
ser.ewm(alpha=alpha, adjust=False).mean()
```

Cumulative moving average case:

```python
np.cumsum(ser) / np.arange(1, len(ser) + 1)
```

Conceptual comparison:

| Method | Weighting | Behavior |
|---|---|---|
| simple moving average | equal weights in fixed window | smooths local noise |
| weighted moving average | chosen weights in fixed window | emphasizes selected positions |
| exponential moving average | exponentially decaying weights | emphasizes recent observations |

## 2.9 Seasonality, Cycles, and Irregularity

Trend: long-term direction.

Seasonality: repeated pattern with fixed period.

Cycle: rise and fall without a fixed period.

Irregular component: noise, shocks, or unexplained movement.

Example distinction:

| Pattern | Fixed period? | Example |
|---|---|---|
| seasonality | yes | weekly case-reporting pattern |
| cycle | no | business cycle |
| irregular | no | data reporting backlog or one-time event |

## 2.10 Time Series Similarity

Assignments compute pairwise similarity/distance matrices for five country time series.

A pairwise output matrix should usually:

- be a `pd.DataFrame`,
- have the same row and column labels,
- be square,
- preserve the country order,
- have float dtype.

Matrix pattern:

```python
def pairwise_template(df, metric):
    labels = df.columns
    out = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for c1 in labels:
        for c2 in labels:
            out.loc[c1, c2] = metric(df[c1].values, df[c2].values)

    return out
```

## 2.11 Euclidean Distance

\[
d(x,y) = \sqrt{\sum_i (x_i - y_i)^2}
\]

Python:

```python
def euclidean(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sqrt(np.sum((x - y) ** 2))
```

Best when two series are aligned point-by-point and scale differences matter.

## 2.12 Cosine Similarity

\[
cos(x,y) = \frac{x \cdot y}{||x|| ||y||}
\]

Python:

```python
def cosine(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
```

Cosine similarity focuses more on direction or shape than absolute magnitude.

## 2.13 Dynamic Time Warping

Dynamic Time Warping, or DTW, compares time series by allowing flexible alignment.

Use DTW when similar patterns happen at different speeds or slightly different times.

Assignment pattern:

```python
def calc_pairwise_dtw_cost(x, y, ret_matrix=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1]
            )

    if ret_matrix:
        return dtw[1:, 1:]

    return float(dtw[n, m])
```

DTW versus Euclidean:

| Method | Alignment assumption |
|---|---|
| Euclidean | point `i` only matches point `i` |
| DTW | points can be flexibly aligned while preserving order |

## 2.14 DTW and Edit Distance

DTW and edit distance are both dynamic programming methods for ordered data.

| Method | Usually used for | Cost idea |
|---|---|---|
| DTW | numeric time series | alignment cost between values |
| edit distance | symbolic sequences | insertion, deletion, substitution operations |

## 2.15 Symbolic Representations

Symbolic time series representations convert numeric values into discrete symbols.

Example:

```python
df["symbol"] = pd.cut(
    df["value"],
    bins=3,
    labels=["low", "medium", "high"]
)
```

Reasons to use symbolic representations:

- reduce dimensionality,
- reduce noise,
- make sequence mining methods applicable,
- simplify pattern matching.

# Unit 3: Time Series Forecasting

## 3.1 Forecasting Overview

Forecasting predicts future values using past observations.

Important rule: do not randomly split time series data. Use chronological splits because the prediction task is about using the past to predict the future.

## 3.2 Rolling Statistics

Assignment function:

```python
def calc_rolling_stats(ser, wd_size=7):
    rolling_mean = ser.rolling(wd_size).mean()
    rolling_std = ser.rolling(wd_size).std()
    return rolling_mean, rolling_std
```

Expected output:

- tuple of length 2,
- each element is a `pd.Series`,
- same length as input,
- same `DatetimeIndex`,
- float dtype.

## 3.3 Stationarity

A stationary time series has stable statistical properties over time:

- stable mean,
- stable variance,
- stable autocorrelation structure.

Why it matters:

Many forecasting models assume that relationships over time are stable. Trend and changing variance can make raw data non-stationary.

Augmented Dickey-Fuller example:

```python
from statsmodels.tsa.stattools import adfuller

stat, pval, *_ = adfuller(ser.dropna())
```

Interpretation:

- Small p-value: evidence against a unit root; series is more likely stationary.
- Large p-value: insufficient evidence of stationarity.

## 3.4 Log Returns

Assignment function:

```python
def calc_log_ret(ser):
    return np.log(ser).diff().dropna()
```

Log returns model relative change rather than level.

\[
r_t = \log(Y_t) - \log(Y_{t-1})
\]

Useful because log returns often stabilize variance and make growth-like data easier to model.

## 3.5 Differencing

First difference:

\[
Y'_t = Y_t - Y_{t-1}
\]

Python:

```python
diff = ser.diff().dropna()
```

Purpose:

Differencing removes trend by modeling changes instead of raw levels.

To convert differenced forecasts back to original scale:

```python
forecast = ser.iloc[-1] + diff_forecast.cumsum()
```

## 3.6 Autocorrelation and ACF

Autocorrelation measures the relationship between a time series and lagged versions of itself.

Assignment pattern:

```python
from statsmodels.tsa.stattools import acf

def calc_acf(ser, max_lag):
    return acf(ser.dropna(), nlags=max_lag)
```

Expected length is `max_lag + 1` because lag 0 is included.

Lag 0 autocorrelation is always 1.

## 3.7 PACF

PACF measures the direct relationship between a series and a lag after accounting for shorter lags.

Assignment pattern:

```python
from statsmodels.tsa.stattools import pacf

def calc_pacf(ser, max_lag):
    return pacf(ser.dropna(), nlags=max_lag)
```

ACF versus PACF:

| Tool | Interpretation |
|---|---|
| ACF | total correlation with lagged values |
| PACF | direct correlation with a lag after shorter lags are controlled |

## 3.8 AR, MA, ARMA, and ARIMA

AR model:

\[
Y_t = c + \phi_1Y_{t-1} + ... + \phi_pY_{t-p} + \epsilon_t
\]

MA model:

\[
Y_t = c + \epsilon_t + \theta_1\epsilon_{t-1} + ... + \theta_q\epsilon_{t-q}
\]

ARMA combines AR and MA on a stationary series.

ARIMA is written as `ARIMA(p, d, q)`:

| Parameter | Meaning |
|---|---|
| p | autoregressive order |
| d | differencing order |
| q | moving-average order |

In the assignment, ARMA is implemented using `ARIMA(..., order=(p, 0, q))` on transformed data.

## 3.9 ARMA on Log Returns

Pattern:

```python
from statsmodels.tsa.arima.model import ARIMA


def arma_log_ret(ser, p, q, num_forecasts):
    log_ret = np.log(ser).diff().dropna()

    model = ARIMA(log_ret, order=(p, 0, q))
    result = model.fit()

    ret_forecast = result.forecast(steps=num_forecasts)

    future_index = pd.date_range(
        ser.index[-1] + pd.Timedelta(days=1),
        periods=num_forecasts,
        freq="D"
    )

    # Convert forecasted log returns back to original scale.
    forecast_values = ser.iloc[-1] * np.exp(np.cumsum(ret_forecast.values))

    return pd.Series(forecast_values, index=future_index, dtype=float)
```

Key idea:

If log returns are forecasted, use cumulative predicted log returns and exponentiate to return to the original data scale.

## 3.10 ARMA on First Differences

Pattern:

```python
def arma_first_diff(ser, p, q, num_forecasts):
    diff = ser.diff().dropna()

    model = ARIMA(diff, order=(p, 0, q))
    result = model.fit()

    diff_forecast = result.forecast(steps=num_forecasts)

    future_index = pd.date_range(
        ser.index[-1] + pd.Timedelta(days=1),
        periods=num_forecasts,
        freq="D"
    )

    forecast_values = ser.iloc[-1] + np.cumsum(diff_forecast.values)

    return pd.Series(forecast_values, index=future_index, dtype=float)
```

## 3.11 VAR: Vector Autoregression

VAR models multiple time series together.

Instead of predicting each series from only its own past, VAR predicts each series from the lagged values of all series.

Pattern:

```python
from statsmodels.tsa.api import VAR


def var_first_diff(df, p, num_forecasts):
    diff_df = df.diff().dropna()

    model = VAR(diff_df)
    result = model.fit(p)

    lagged_values = diff_df.values[-p:]
    diff_forecast = result.forecast(lagged_values, steps=num_forecasts)

    future_index = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        periods=num_forecasts,
        freq="D"
    )

    diff_forecast_df = pd.DataFrame(
        diff_forecast,
        index=future_index,
        columns=df.columns
    )

    forecast_df = df.iloc[-1] + diff_forecast_df.cumsum()

    return result, forecast_df.astype(float)
```

VAR versus independent AR:

| Model | Uses other series? | Use case |
|---|---|---|
| independent AR | no | each series is modeled separately |
| VAR | yes | series may influence each other over time |

## 3.12 Independent AR Models for Multiple Series

Pattern:

```python
def ar_first_diff(df, p, num_forecasts):
    forecasts = pd.DataFrame()

    for col in df.columns:
        forecasts[col] = arma_first_diff(
            df[col],
            p=p,
            q=0,
            num_forecasts=num_forecasts
        )

    return forecasts.astype(float)
```

## 3.13 Granger Causality

Granger causality asks whether past values of one series improve prediction of another series after accounting for the target's own past.

Definition:

X Granger-causes Y if lagged values of X improve forecasts of Y beyond lagged values of Y alone.

Critical warning:

Granger causality is predictive causality. It does not prove true causal mechanism or intervention-based causality.

Pattern for pairwise testing:

```python
from statsmodels.tsa.stattools import grangercausalitytests


def test_granger(df, p):
    labels = df.columns
    out = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for target in labels:
        for predictor in labels:
            if target == predictor:
                out.loc[target, predictor] = np.nan
            else:
                tests = grangercausalitytests(
                    df[[target, predictor]],
                    maxlag=p,
                    verbose=False
                )
                # one common choice: p-value from ssr F-test at lag p
                out.loc[target, predictor] = tests[p][0]["ssr_ftest"][1]

    return out.astype(float)
```

Important orientation:

In `df[[target, predictor]]`, many `statsmodels` examples test whether the second column Granger-causes the first column. Always confirm the assignment's expected orientation.

## 3.14 Outlier Detection

Point outlier: one unusual value.

Contextual outlier: unusual given time context.

Subsequence outlier: unusual pattern over an interval.

Global z-score:

```python
z = (ser - ser.mean()) / ser.std()
outliers = ser[z.abs() > 3]
```

Rolling z-score:

```python
rolling_mean = ser.rolling(20).mean()
rolling_std = ser.rolling(20).std()
rolling_z = (ser - rolling_mean) / rolling_std
outliers = ser[rolling_z.abs() > 3]
```

A rolling method can be better when the series has local trend or seasonality.

# Unit 4: Data Streams

## 4.1 Data Streams

A data stream is a potentially infinite sequence of data items arriving over time.

Examples:

- tweets,
- web clicks,
- transactions,
- sensor readings,
- network packets,
- clinical monitoring events.

Stream restrictions:

- cannot store all data,
- stream length may be unknown,
- each item may be seen only once,
- processing must be fast,
- distribution may change over time.

## 4.2 Synopsis Structures

A synopsis is a compact summary of a stream.

Examples:

| Synopsis | Purpose |
|---|---|
| sample | approximate stream distribution |
| reservoir | fixed-size uniform sample |
| Bloom filter | approximate membership |
| lossy counter | approximate frequent items |
| sketch | approximate counts or aggregates |

Core tradeoff:

Synopsis structures save memory by accepting approximation.

## 4.3 Emoji Extraction in Assignment 4

The stream assignments use Twitter data and emoji counts.

Pattern:

```python
def extract_emojis(text):
    return [char for char in text if char in UNICODE_EMOJI]
```

Counts are usually stored in a `defaultdict(int)`:

```python
from collections import defaultdict

counts = defaultdict(int)
```

Add emoji counts:

```python
for emoji in extract_emojis(tweet):
    counts[emoji] += 1
```

Remove emoji counts:

```python
for emoji in extract_emojis(tweet):
    counts[emoji] -= 1
    if counts[emoji] == 0:
        del counts[emoji]
```

## 4.4 Random Sampling from a Stream

Random sampling in the assignment adds each item independently with probability `in_sample_prob`.

Pattern:

```python
class RandomSampler:
    def __init__(self, in_sample_prob, seed=None):
        self.in_sample_prob = in_sample_prob
        self.random = HistPresvRandom(seed)
        self.sample = []
        self.counts = defaultdict(int)

    def _process_new_item(self, item):
        if self.random.random() < self.in_sample_prob:
            self.sample.append(item)
            for emoji in extract_emojis(item):
                self.counts[emoji] += 1

    def do_sampling(self, stream):
        for item in stream:
            self._process_new_item(item)
            yield self.sample.copy(), self.counts.copy()
```

Conceptual issue:

The final sample size is random. It is not guaranteed to equal a fixed size.

## 4.5 Reservoir Sampling

Reservoir sampling maintains a fixed-size uniform sample from a stream of unknown length.

Algorithm for sample size `k`:

1. Keep the first `k` items.
2. For the item at stream index `i`, where indexing starts at 0:
   - choose a random item from the range `0` through `i`,
   - if the chosen index is less than `k`, remove an existing reservoir item and add the new item.

Assignment-specific detail: append the new item to the end of `self.sample` instead of replacing the old item in place.

Pattern:

```python
class ReservoirSampler:
    def __init__(self, sample_size, seed=None):
        self.sample_size = sample_size
        self.random = HistPresvRandom(seed)
        self.sample = []
        self.counts = defaultdict(int)

    def _add_item(self, item):
        self.sample.append(item)
        for emoji in extract_emojis(item):
            self.counts[emoji] += 1

    def _remove_item(self, item):
        self.sample.remove(item)
        for emoji in extract_emojis(item):
            self.counts[emoji] -= 1
            if self.counts[emoji] == 0:
                del self.counts[emoji]

    def _process_new_item(self, item, index):
        if len(self.sample) < self.sample_size:
            self._add_item(item)
        else:
            selected = self.random.sample(range(index + 1))
            if selected < self.sample_size:
                old_item = self.sample[selected]
                self._remove_item(old_item)
                self._add_item(item)

    def do_sampling(self, stream):
        for index, item in enumerate(stream):
            self._process_new_item(item, index)
            yield self.sample.copy(), self.counts.copy()
```

Conceptual explanation:

After processing `n` items, each item has probability `k / n` of being in the reservoir. The algorithm treats early and late items fairly without knowing the stream length in advance.

## 4.6 Bloom Filters

A Bloom filter is a memory-efficient probabilistic data structure for membership testing.

It uses:

- a bit array or slot array,
- multiple hash functions.

Insert item:

1. Hash the item using each hash function.
2. Set the corresponding slots to 1.

Check item:

1. Hash the item using each hash function.
2. If any corresponding slot is 0, the item has definitely not appeared.
3. If all corresponding slots are 1, the item has possibly appeared.

Assignment pattern:

```python
class BloomFilter:
    def __init__(self, num_slots, num_hash_fns):
        self.slots = np.zeros(num_slots, dtype=int)
        self.hash_fns = [HashFunction(num_slots) for _ in range(num_hash_fns)]

    def check_appearance(self, item):
        return all(self.slots[h(item)] == 1 for h in self.hash_fns)

    def do_filtering(self, stream):
        for tweet in stream:
            for emoji in extract_emojis(tweet):
                for h in self.hash_fns:
                    self.slots[h(emoji)] = 1
            yield self.slots.copy()
```

False positives are possible. False negatives are not expected in a standard insertion-only Bloom filter.

## 4.7 Lossy Counting

The assignment uses a bucket-based lossy counter for emojis.

Main idea:

- collect tweets into buckets,
- when a full bucket arrives, update counts,
- increment counts for emojis in the new bucket,
- decrement existing counts to gradually remove less recent emojis,
- delete emojis whose count reaches zero,
- yield a copy of counts at every stream iteration.

Conceptual skeleton:

```python
class LossyCounter:
    def __init__(self, bucket_size):
        self.bucket_size = bucket_size
        self.counts = defaultdict(int)

    def do_counting(self, stream):
        bucket = []

        for tweet in stream:
            bucket.append(tweet)

            if len(bucket) == self.bucket_size:
                # 1. decrement existing counts
                for emoji in list(self.counts.keys()):
                    self.counts[emoji] -= 1
                    if self.counts[emoji] == 0:
                        del self.counts[emoji]

                # 2. increment counts from new bucket
                for item in bucket:
                    for emoji in extract_emojis(item):
                        self.counts[emoji] += 1

                bucket = []

            yield self.counts.copy()
```

Important assignment detail:

The autograder checks immediately after a full bucket arrives, so increments, decrements, and deletion of zero-count items must all be complete before the `yield` for that iteration.

# Cross-Cutting Implementation Pitfalls

## Return Types

Always match the expected return type exactly.

| Function family | Expected type |
|---|---|
| `load_data` for sonnets | `list` |
| `build_vocab` | `list` |
| `build_ngrams` | `list` |
| `bigram_next_token` | `tuple(str, float)` |
| `train_ngram_lm` | `nltk.lm.MLE` |
| `train_hmm` | `HiddenMarkovModelTagger` |
| global COVID load | `pd.Series` |
| top-country COVID load | `pd.DataFrame` |
| trend/WMA/EMA | often `np.ndarray` |
| pairwise distance/similarity | `pd.DataFrame` |
| rolling stats | tuple of `pd.Series` |
| ACF/PACF | `np.ndarray` |
| forecasts | `pd.Series` or `pd.DataFrame` |

## Index and Shape Discipline

For time series:

- keep `pd.DatetimeIndex`,
- sort by time,
- avoid random splits,
- drop NaNs after `diff()` or transformations when appropriate,
- create future indexes with `pd.date_range`,
- preserve column order in DataFrames.

## Copy Discipline in Stream Assignments

When yielding sample or count states, use `.copy()`.

```python
yield self.sample.copy(), self.counts.copy()
```

This prevents later mutations from changing previously yielded objects.

## Do Not Mutate Inputs Unless Asked

Especially in N-gram and time series functions, hidden tests may hash inputs before and after function calls.

Use copies where needed:

```python
sent = list(sentence)
ser = ser.copy()
df = df.copy()
```

# Oral Exam Explanation Bank

## Bigram model

A bigram model predicts the next token using only the immediately previous token. It estimates conditional probabilities from counts: the count of the observed bigram divided by the count of the previous token.

## First-order Markov chain

A first-order Markov chain assumes the next state depends only on the current state, not the full history. This simplifies sequence modeling by reducing the amount of context needed.

## HMM

An HMM models sequences where the state is hidden but produces observable outputs. In POS tagging, the words are observed and the part-of-speech tags are hidden states inferred from transition and emission probabilities.

## Moving average

A moving average smooths a time series by averaging nearby observations. It reduces short-term noise and reveals broader trend, but it can lag behind sudden changes.

## EMA

An exponential moving average is a weighted average where recent observations receive more weight. It is more responsive than a simple moving average.

## DTW

DTW compares two time series by finding a low-cost alignment between them. It is useful when similar patterns occur at slightly different times or speeds.

## Stationarity

A stationary series has stable mean, variance, and autocorrelation over time. Many forecasting models work better on stationary data because the past relationships are more likely to remain valid.

## ACF vs PACF

ACF measures the correlation between a series and lagged versions of itself. PACF measures the direct correlation at a given lag after accounting for shorter lags.

## ARIMA

ARIMA combines autoregression, differencing, and moving-average error correction. The `p` parameter controls autoregressive lags, `d` controls differencing, and `q` controls moving-average error terms.

## VAR

VAR extends autoregression to multiple time series. Each variable is predicted using lagged values of itself and lagged values of the other variables.

## Granger causality

Granger causality tests whether past values of one variable improve prediction of another variable. It is predictive causality, not proof of true causal mechanism.

## Reservoir sampling

Reservoir sampling keeps a fixed-size random sample from a stream of unknown length. After processing `n` items, each item has the same probability of being included in the reservoir.

## Bloom filter

A Bloom filter is a memory-efficient membership structure. It can say an item is definitely absent or possibly present. False positives are possible, but false negatives are not expected in an insertion-only Bloom filter.

## Lossy counting

Lossy counting approximates frequent items in a stream using limited memory. It updates counts in buckets and removes items whose counts are too small or stale.

# Final Exam Checklist

## Must-code skills

- tokenize text lines,
- build vocabulary with boundary tokens,
- generate padded N-grams,
- compute bigram/trigram probabilities,
- train `nltk.lm.MLE`,
- parse CoNLL-style labeled data,
- train an HMM tagger,
- convert cumulative time series to daily new values,
- preserve `DatetimeIndex`,
- run seasonal decomposition,
- fit polynomial trend curves,
- calculate WMA and EMA,
- create pairwise Euclidean/cosine/DTW matrices,
- calculate rolling mean/std,
- calculate log returns,
- calculate ACF/PACF,
- fit ARIMA on transformed series,
- invert forecasts back to original scale,
- fit VAR on differenced multivariate data,
- run Granger causality tests,
- implement random sampling,
- implement reservoir sampling,
- implement Bloom filters,
- implement lossy counting.

## Must-explain skills

- why N-grams are Markov-style models,
- why sentence padding matters,
- why unseen N-grams get zero probability under MLE,
- difference between Markov chain and HMM,
- trend vs seasonality vs cycles,
- SMA vs WMA vs EMA,
- Euclidean vs cosine vs DTW,
- stationarity and differencing,
- ACF vs PACF,
- AR vs MA vs ARMA vs ARIMA,
- VAR vs independent AR,
- Granger causality limitations,
- why stream mining uses approximation,
- reservoir sampling fairness,
- Bloom filter false positives,
- lossy counting memory tradeoff.

