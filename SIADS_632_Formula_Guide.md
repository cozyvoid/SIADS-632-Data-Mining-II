---
title: "SIADS 632 Data Mining II - Detailed Formula Guide"
subtitle: "Sequences, Time Series, Forecasting, and Data Streams"
author: "Study Guide"
date: "2026"
geometry: margin=0.75in
fontsize: 10pt
colorlinks: true
linkcolor: blue
urlcolor: cyan
header-includes:
  - \usepackage{xcolor}
  - \usepackage{amsmath,amssymb}
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{array}
  - \usepackage{enumitem}
  - \setlist{nosep}
---

# How to Use This Formula Guide

This guide collects the formulas most likely to matter in SIADS 632: Data Mining II. It is organized by course unit and by assignment-tested skill. Each formula is labeled so you can reference it during review.

For each formula, focus on four things:

1. **What the formula computes.**
2. **What each symbol means.**
3. **What assumptions are being made.**
4. **How it connects to Python implementation.**

A recurring rule throughout the course is that formulas usually compare an event to a relevant reference set. For example, in N-gram probabilities, the denominator is the count of the context because the context represents the number of chances for the next token to occur.

\newpage

# 1. Sequences and N-Gram Language Models

## 1.1 Basic Sequence Notation

**Formula S1. Sequence of observed tokens**

\[
W = (w_1, w_2, \ldots, w_T)
\]

Where:

- \(W\) is a sequence.
- \(w_t\) is the token/event at position \(t\).
- \(T\) is the sequence length.

Use this for word sequences, click sequences, DNA sequences, or any ordered list of discrete symbols.

---

## 1.2 N-Grams

**Formula S2. N-gram definition**

\[
(w_i, w_{i+1}, \ldots, w_{i+n-1})
\]

Where:

- \(n\) is the length of the N-gram.
- A unigram has \(n=1\).
- A bigram has \(n=2\).
- A trigram has \(n=3\).

Python pattern:

```python
list(zip(tokens, tokens[1:]))          # bigrams
list(zip(tokens, tokens[1:], tokens[2:]))  # trigrams
```

---

## 1.3 Unigram Probability

**Formula S3. Unigram maximum likelihood estimate**

\[
P(w) = \frac{count(w)}{N}
\]

Where:

- \(count(w)\) is the number of times token \(w\) appears.
- \(N\) is the total number of tokens.

Interpretation:

> Out of all tokens in the corpus, what fraction are token \(w\)?

Python pattern:

```python
Counter(tokens)[w] / len(tokens)
```

---

## 1.4 Bigram Probability

**Formula S4. Bigram conditional probability**

\[
P(w_i \mid w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}
\]

Where:

- \(count(w_{i-1}, w_i)\) is the number of times the full bigram appears.
- \(count(w_{i-1})\) is the number of times the previous token appears as a context.

Easy denominator rule:

> The denominator is the number of chances the next token had to appear. For \(P(B \mid A)\), count how many times \(A\) appeared.

Python pattern:

```python
bigram_counts[(A, B)] / unigram_counts[A]
```

---

## 1.5 Trigram Probability

**Formula S5. Trigram conditional probability**

\[
P(w_i \mid w_{i-2}, w_{i-1}) =
\frac{count(w_{i-2}, w_{i-1}, w_i)}{count(w_{i-2}, w_{i-1})}
\]

Where:

- The numerator counts the full triple.
- The denominator counts the two-token context.

Easy denominator rule:

> For \(P(C \mid A,B)\), count how many times \((A,B)\) appeared. Those are the chances for \(C\) to appear next.

Python pattern:

```python
trigram_counts[(A, B, C)] / bigram_counts[(A, B)]
```

---

## 1.6 General N-Gram Probability

**Formula S6. General N-gram conditional probability**

\[
P(w_i \mid w_{i-n+1}, \ldots, w_{i-1}) =
\frac{count(w_{i-n+1}, \ldots, w_{i-1}, w_i)}
{count(w_{i-n+1}, \ldots, w_{i-1})}
\]

Where:

- The numerator is the full N-gram count.
- The denominator is the \((n-1)\)-gram context count.

Memory shortcut:

> Full N-gram on top. Context on bottom.

---

## 1.7 Chain Rule for a Sequence

**Formula S7. Exact chain rule**

\[
P(w_1, w_2, \ldots, w_T)
= \prod_{i=1}^{T} P(w_i \mid w_1, \ldots, w_{i-1})
\]

This is exact but usually impractical because it requires conditioning on the entire previous history.

---

## 1.8 Bigram Approximation to Sequence Probability

**Formula S8. Bigram approximation**

\[
P(w_1, \ldots, w_T) \approx
P(w_1) \prod_{i=2}^{T} P(w_i \mid w_{i-1})
\]

With sentence boundary tokens:

\[
P(W) \approx \prod_{i=1}^{T+1} P(w_i \mid w_{i-1})
\]

where \(w_0 = \langle s \rangle\) and \(w_{T+1}=\langle /s \rangle\).

---

## 1.9 Trigram Approximation to Sequence Probability

**Formula S9. Trigram approximation**

\[
P(w_1, \ldots, w_T) \approx
\prod_{i=1}^{T+1} P(w_i \mid w_{i-2}, w_{i-1})
\]

This assumes the next token depends only on the previous two tokens.

---

## 1.10 Add-One / Laplace Smoothing

**Formula S10. Add-one smoothed bigram probability**

\[
P_{add1}(w_i \mid w_{i-1}) =
\frac{count(w_{i-1}, w_i) + 1}{count(w_{i-1}) + V}
\]

Where:

- \(V\) is the vocabulary size.
- The \(+1\) prevents zero probabilities.
- The denominator adds \(V\) because one count is added for every possible next token.

Interpretation:

> Smoothing reserves probability mass for unseen but possible events.

---

## 1.11 Add-\(k\) Smoothing

**Formula S11. Add-k smoothed probability**

\[
P_{addk}(w_i \mid c) =
\frac{count(c,w_i) + k}{count(c) + kV}
\]

Where:

- \(c\) is the context.
- \(k\) is the smoothing constant.
- \(V\) is vocabulary size.

Special case:

\[
k=1 \Rightarrow \text{Laplace smoothing}
\]

---

## 1.12 Log Probability of a Sequence

**Formula S12. Log sequence probability**

\[
\log P(W) = \sum_{i=1}^{T} \log P(w_i \mid context_i)
\]

Why use logs?

> Multiplying many small probabilities can underflow numerically. Logs convert products into sums.

Python pattern:

```python
sum(np.log(probabilities))
```

---

## 1.13 Perplexity

**Formula S13. Perplexity**

\[
PP(W) = P(w_1, w_2, \ldots, w_T)^{-1/T}
\]

Equivalent log form:

\[
PP(W) = \exp\left(-\frac{1}{T}\sum_{i=1}^{T}\log P(w_i \mid context_i)\right)
\]

Interpretation:

> Lower perplexity means the model is less surprised by the sequence.

---

# 2. Markov Chains

## 2.1 Markov Assumption

**Formula M1. First-order Markov assumption**

\[
P(X_t \mid X_{t-1}, X_{t-2}, \ldots, X_1) = P(X_t \mid X_{t-1})
\]

Interpretation:

> The next state depends only on the current state, not the full past.

Connection to bigrams:

> A bigram language model is a first-order Markov model over tokens.

---

## 2.2 Transition Probability

**Formula M2. Transition probability from state \(i\) to state \(j\)**

\[
P_{ij} = P(X_{t+1}=j \mid X_t=i)
\]

Where:

- \(i\) is the current state.
- \(j\) is the next state.
- \(P_{ij}\) is an entry in the transition matrix.

---

## 2.3 Transition Matrix

**Formula M3. Markov transition matrix**

\[
P =
\begin{bmatrix}
P_{11} & P_{12} & \cdots & P_{1K} \\
P_{21} & P_{22} & \cdots & P_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
P_{K1} & P_{K2} & \cdots & P_{KK}
\end{bmatrix}
\]

Each row sums to 1:

\[
\sum_{j=1}^{K} P_{ij} = 1
\]

Python pattern:

```python
P = np.array([[0.8, 0.2],
              [0.4, 0.6]])
```

---

## 2.4 One-Step State Distribution

**Formula M4. One-step distribution update**

\[
\pi_{t+1} = \pi_t P
\]

Where:

- \(\pi_t\) is the distribution over states at time \(t\).
- \(P\) is the transition matrix.

Python pattern:

```python
next_dist = current_dist @ P
```

---

## 2.5 Multi-Step Transition

**Formula M5. k-step state distribution**

\[
\pi_{t+k} = \pi_t P^k
\]

Python pattern:

```python
future_dist = current_dist @ np.linalg.matrix_power(P, k)
```

---

## 2.6 Stationary Distribution

**Formula M6. Stationary distribution**

\[
\pi = \pi P
\]

with:

\[
\sum_i \pi_i = 1
\]

Interpretation:

> A stationary distribution is unchanged after applying the transition matrix.

---

# 3. Hidden Markov Models

## 3.1 HMM Components

An HMM has:

- Hidden states \(S_t\)
- Observations \(O_t\)
- Initial distribution \(\pi\)
- Transition probabilities \(A\)
- Emission probabilities \(B\)

---

## 3.2 Initial State Probability

**Formula H1. Initial probability**

\[
\pi_i = P(S_1 = i)
\]

Where \(S_1\) is the first hidden state.

---

## 3.3 Transition Probability

**Formula H2. HMM transition probability**

\[
A_{ij} = P(S_t=j \mid S_{t-1}=i)
\]

This models movement between hidden states.

---

## 3.4 Emission Probability

**Formula H3. Emission probability**

\[
B_j(o_t) = P(O_t=o_t \mid S_t=j)
\]

This models how likely a hidden state is to emit a visible observation.

Example for POS tagging:

\[
P(\text{word}=\text{``run''} \mid \text{tag}=\text{VERB})
\]

---

## 3.5 HMM Joint Probability

**Formula H4. Joint probability of hidden states and observations**

\[
P(S,O) = \pi_{s_1} B_{s_1}(o_1)
\prod_{t=2}^{T} A_{s_{t-1},s_t} B_{s_t}(o_t)
\]

Interpretation:

> Probability of a hidden path and the observations generated along that path.

---

## 3.6 Forward Recursion

**Formula H5. Forward probability**

\[
\alpha_t(j) = P(o_1, o_2, \ldots, o_t, S_t=j)
\]

Initialization:

\[
\alpha_1(j)=\pi_j B_j(o_1)
\]

Recursion:

\[
\alpha_t(j) = B_j(o_t) \sum_i \alpha_{t-1}(i) A_{ij}
\]

Use:

> Evaluation: how likely is an observation sequence under the model?

---

## 3.7 Viterbi Recursion

**Formula H6. Viterbi best-path recursion**

\[
\delta_t(j) = B_j(o_t) \max_i \left[\delta_{t-1}(i) A_{ij}\right]
\]

Use:

> Decoding: what is the most likely hidden state sequence?

Key difference from forward algorithm:

| Algorithm | Operation over previous states |
|---|---|
| Forward | sum |
| Viterbi | max |

---

# 4. Time Series Basics and Pattern Formulas

## 4.1 Time Series Notation

**Formula T1. Time series**

\[
Y = (Y_1, Y_2, \ldots, Y_T)
\]

Where \(Y_t\) is the observed value at time \(t\).

---

## 4.2 Additive Decomposition

**Formula T2. Additive time series decomposition**

\[
Y_t = T_t + S_t + R_t
\]

Where:

- \(T_t\) is trend.
- \(S_t\) is seasonal component.
- \(R_t\) is residual or irregular component.

Use when seasonal variation is roughly constant over time.

---

## 4.3 Multiplicative Decomposition

**Formula T3. Multiplicative decomposition**

\[
Y_t = T_t \times S_t \times R_t
\]

Use when seasonal variation grows or shrinks with the level of the series.

---

## 4.4 Seasonal Strength Concept

**Formula T4. Seasonal effect by period**

\[
S_t = S_{t-m}
\]

Where \(m\) is the seasonal period.

Examples:

- Daily data with weekly seasonality: \(m=7\)
- Monthly data with yearly seasonality: \(m=12\)
- Quarterly data with yearly seasonality: \(m=4\)

---

## 4.5 First Difference

**Formula T5. First-order differencing**

\[
\nabla Y_t = Y_t - Y_{t-1}
\]

Python pattern:

```python
df["diff_1"] = df["value"].diff()
```

Interpretation:

> Differencing turns levels into changes and can remove trend.

---

## 4.6 Seasonal Difference

**Formula T6. Seasonal differencing**

\[
\nabla_m Y_t = Y_t - Y_{t-m}
\]

Where \(m\) is the seasonal period.

Python pattern:

```python
df["diff_12"] = df["value"].diff(12)
```

---

## 4.7 Simple Return

**Formula T7. Simple return**

\[
r_t = \frac{Y_t - Y_{t-1}}{Y_{t-1}}
\]

Equivalent:

\[
r_t = \frac{Y_t}{Y_{t-1}} - 1
\]

Use for relative change.

---

## 4.8 Log Return

**Formula T8. Log return**

\[
r_t = \log(Y_t) - \log(Y_{t-1})
\]

Equivalent:

\[
r_t = \log\left(\frac{Y_t}{Y_{t-1}}\right)
\]

Python pattern:

```python
log_returns = np.log(ser).diff().dropna()
```

Why useful?

> Log returns often stabilize variance and convert multiplicative changes into additive changes.

---

# 5. Moving Averages and Smoothing

## 5.1 Simple Moving Average

**Formula MA1. Simple moving average**

\[
SMA_t = \frac{1}{k}\sum_{i=0}^{k-1}Y_{t-i}
\]

Where \(k\) is the window size.

Python pattern:

```python
ser.rolling(window=k).mean()
```

Larger \(k\): smoother but more lag.

Smaller \(k\): more responsive but noisier.

---

## 5.2 Weighted Moving Average

**Formula MA2. Weighted moving average**

\[
WMA_t = \sum_{i=0}^{k-1} a_i Y_{t-i}
\]

with:

\[
\sum_{i=0}^{k-1} a_i = 1
\]

Where \(a_i\) is the weight for each lagged observation.

Python pattern:

```python
weights = np.array(weights) / np.sum(weights)
ser.rolling(k).apply(lambda x: np.sum(x * weights), raw=True)
```

---

## 5.3 Exponential Moving Average

**Formula MA3. Exponential moving average**

\[
EMA_t = \alpha Y_t + (1-\alpha)EMA_{t-1}
\]

Where:

- \(0 < \alpha < 1\)
- Large \(\alpha\): more responsive
- Small \(\alpha\): smoother

Python pattern:

```python
ser.ewm(alpha=alpha, adjust=False).mean()
```

---

## 5.4 EMA Weight Decay

**Formula MA4. EMA weight for observation \(k\) periods ago**

\[
w_k = \alpha(1-\alpha)^k
\]

Interpretation:

> Older observations receive exponentially smaller weights.

---

# 6. Similarity and Distance Formulas

## 6.1 Euclidean Distance

**Formula D1. Euclidean distance**

\[
d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

Python pattern:

```python
np.sqrt(np.sum((x - y) ** 2))
```

Use when:

- series have equal length
- time points are aligned
- magnitude differences matter

---

## 6.2 Squared Euclidean Distance

**Formula D2. Squared Euclidean distance**

\[
d^2(x,y) = \sum_{i=1}^{n}(x_i-y_i)^2
\]

Useful because it avoids the square root when only relative comparison is needed.

---

## 6.3 Manhattan Distance

**Formula D3. Manhattan distance**

\[
d(x,y) = \sum_{i=1}^{n}|x_i-y_i|
\]

Python pattern:

```python
np.sum(np.abs(x - y))
```

---

## 6.4 Minkowski Distance

**Formula D4. Minkowski distance**

\[
d_p(x,y) = \left(\sum_{i=1}^{n}|x_i-y_i|^p\right)^{1/p}
\]

Special cases:

- \(p=1\): Manhattan distance
- \(p=2\): Euclidean distance

---

## 6.5 Cosine Similarity

**Formula D5. Cosine similarity**

\[
\cos(x,y) = \frac{x \cdot y}{\|x\|\|y\|}
\]

Expanded:

\[
\cos(x,y) = \frac{\sum_i x_i y_i}
{\sqrt{\sum_i x_i^2}\sqrt{\sum_i y_i^2}}
\]

Python pattern:

```python
np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
```

Interpretation:

> Compares direction or shape more than magnitude.

---

## 6.6 Correlation Similarity

**Formula D6. Pearson correlation**

\[
r_{xy} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum_i (x_i - \bar{x})^2}\sqrt{\sum_i (y_i - \bar{y})^2}}
\]

Python pattern:

```python
np.corrcoef(x, y)[0, 1]
```

Interpretation:

> Measures whether two series move together after centering.

---

## 6.7 Dynamic Time Warping Recurrence

**Formula D7. DTW local cost**

\[
c(i,j)=|x_i-y_j|
\]

or sometimes:

\[
c(i,j)=(x_i-y_j)^2
\]

---

**Formula D8. DTW dynamic programming recurrence**

\[
DTW(i,j)=c(i,j)+\min
\begin{cases}
DTW(i-1,j) \\
DTW(i,j-1) \\
DTW(i-1,j-1)
\end{cases}
\]

Base case:

\[
DTW(0,0)=0
\]

Interpretation:

> DTW finds a minimum-cost alignment path between two ordered series.

---

## 6.8 Edit Distance Recurrence

**Formula D9. Edit distance recurrence**

\[
ED(i,j)=\min
\begin{cases}
ED(i-1,j)+1 & \text{deletion} \\
ED(i,j-1)+1 & \text{insertion} \\
ED(i-1,j-1)+\mathbb{1}(x_i \ne y_j) & \text{substitution/match}
\end{cases}
\]

Use for symbolic sequences rather than continuous numeric time series.

---

# 7. Stationarity, ACF, and PACF

## 7.1 Weak Stationarity

**Formula ST1. Constant mean**

\[
E[Y_t] = \mu
\]

**Formula ST2. Constant variance**

\[
Var(Y_t)=\sigma^2
\]

**Formula ST3. Lag-dependent covariance**

\[
Cov(Y_t,Y_{t-k}) = \gamma_k
\]

Interpretation:

> In a weakly stationary series, mean and variance are stable, and covariance depends on lag rather than absolute time.

---

## 7.2 Autocovariance

**Formula ST4. Autocovariance at lag \(k\)**

\[
\gamma_k = Cov(Y_t,Y_{t-k})
\]

Sample version:

\[
\hat{\gamma}_k = \frac{1}{T}\sum_{t=k+1}^{T}(Y_t-\bar{Y})(Y_{t-k}-\bar{Y})
\]

---

## 7.3 Autocorrelation Function

**Formula ST5. Autocorrelation at lag \(k\)**

\[
\rho_k = \frac{\gamma_k}{\gamma_0}
\]

Where:

- \(\gamma_k\) is lag-\(k\) autocovariance.
- \(\gamma_0\) is variance.

Python pattern:

```python
ser.autocorr(lag=k)
```

Interpretation:

> ACF measures how strongly a series is related to its own past values.

---

## 7.4 Partial Autocorrelation

**Formula ST6. Partial autocorrelation concept**

\[
PACF(k) = Corr(Y_t, Y_{t-k} \mid Y_{t-1}, \ldots, Y_{t-k+1})
\]

Interpretation:

> PACF measures the direct relationship with lag \(k\) after removing shorter-lag effects.

---

# 8. Forecasting Error Metrics

## 8.1 Forecast Error

**Formula F1. Forecast error**

\[
e_t = Y_t - \hat{Y}_t
\]

Where:

- \(Y_t\) is the actual value.
- \(\hat{Y}_t\) is the forecast.

---

## 8.2 Mean Error

**Formula F2. Mean error**

\[
ME = \frac{1}{n}\sum_{t=1}^{n} e_t
\]

Interpretation:

> Measures forecast bias. Positive means underprediction if \(e_t=Y_t-\hat{Y}_t\).

---

## 8.3 Mean Absolute Error

**Formula F3. MAE**

\[
MAE = \frac{1}{n}\sum_{t=1}^{n}|e_t|
\]

Interpretation:

> Average absolute forecast miss in the original unit.

---

## 8.4 Mean Squared Error

**Formula F4. MSE**

\[
MSE = \frac{1}{n}\sum_{t=1}^{n}e_t^2
\]

Interpretation:

> Penalizes large errors more strongly than MAE.

---

## 8.5 Root Mean Squared Error

**Formula F5. RMSE**

\[
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}e_t^2}
\]

Interpretation:

> Square root of MSE; reported in the original unit.

---

## 8.6 Mean Absolute Percentage Error

**Formula F6. MAPE**

\[
MAPE = \frac{100\%}{n}\sum_{t=1}^{n}\left|\frac{e_t}{Y_t}\right|
\]

Caution:

> MAPE behaves poorly when actual values are zero or close to zero.

---

## 8.7 Naive Forecast

**Formula F7. Naive forecast**

\[
\hat{Y}_{t+1}=Y_t
\]

Interpretation:

> The next value is predicted to equal the most recent value.

---

## 8.8 Seasonal Naive Forecast

**Formula F8. Seasonal naive forecast**

\[
\hat{Y}_{t}=Y_{t-m}
\]

Where \(m\) is the seasonal period.

Example:

- Daily data with weekly seasonality: \(m=7\)

---

# 9. AR, MA, ARMA, and ARIMA

## 9.1 Autoregressive Model

**Formula A1. AR(1)**

\[
Y_t = c + \phi_1Y_{t-1}+\epsilon_t
\]

---

**Formula A2. AR(p)**

\[
Y_t = c + \phi_1Y_{t-1}+\phi_2Y_{t-2}+\cdots+\phi_pY_{t-p}+\epsilon_t
\]

Interpretation:

> The present value is predicted using previous values of the same series.

---

## 9.2 Moving Average Model

**Formula A3. MA(1)**

\[
Y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1}
\]

---

**Formula A4. MA(q)**

\[
Y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1}+\cdots+\theta_q\epsilon_{t-q}
\]

Interpretation:

> The model uses current and past forecast shocks/errors.

---

## 9.3 ARMA Model

**Formula A5. ARMA(p,q)**

\[
Y_t = c + \sum_{i=1}^{p}\phi_iY_{t-i} + \epsilon_t + \sum_{j=1}^{q}\theta_j\epsilon_{t-j}
\]

Interpretation:

> ARMA combines past values and past errors.

Use when the series is stationary.

---

## 9.4 ARIMA Model

**Formula A6. ARIMA(p,d,q)**

\[
\nabla^d Y_t \sim ARMA(p,q)
\]

Where:

- \(p\): number of autoregressive lags
- \(d\): number of differences
- \(q\): number of moving-average error lags

Interpretation:

> ARIMA applies differencing to make a series more stationary, then fits ARMA to the differenced series.

Python pattern:

```python
ARIMA(ser, order=(p, d, q)).fit()
```

---

## 9.5 Forecast Reconstruction from First Differences

**Formula A7. Reconstruct level forecast from differenced forecast**

If:

\[
\hat{\Delta Y}_{T+h}
\]

is the forecasted difference, then:

\[
\hat{Y}_{T+h} = Y_T + \sum_{i=1}^{h}\hat{\Delta Y}_{T+i}
\]

Python pattern:

```python
forecast = last_value + diff_forecast.cumsum()
```

---

# 10. Vector Autoregression and Granger Causality

## 10.1 Vector Autoregression

**Formula V1. VAR(p)**

\[
\mathbf{Y}_t = \mathbf{c} + A_1\mathbf{Y}_{t-1} + A_2\mathbf{Y}_{t-2} + \cdots + A_p\mathbf{Y}_{t-p} + \boldsymbol{\epsilon}_t
\]

Where:

- \(\mathbf{Y}_t\) is a vector of multiple time series.
- \(A_i\) are coefficient matrices.
- \(p\) is the lag order.

Interpretation:

> Each variable is modeled using lagged values of itself and the other variables.

---

## 10.2 VAR(1) with Two Variables

**Formula V2. Two-variable VAR(1)**

\[
\begin{bmatrix}
Y_t \\
X_t
\end{bmatrix}
=
\begin{bmatrix}
c_y \\
c_x
\end{bmatrix}
+
\begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
\begin{bmatrix}
Y_{t-1} \\
X_{t-1}
\end{bmatrix}
+
\begin{bmatrix}
\epsilon_{y,t} \\
\epsilon_{x,t}
\end{bmatrix}
\]

Interpretation:

- \(a_{11}\): effect of past \(Y\) on current \(Y\)
- \(a_{12}\): effect of past \(X\) on current \(Y\)
- \(a_{21}\): effect of past \(Y\) on current \(X\)
- \(a_{22}\): effect of past \(X\) on current \(X\)

---

## 10.3 Granger Causality Concept

**Formula V3. Restricted model for Y**

\[
Y_t = c + \sum_{i=1}^{p}a_iY_{t-i}+\epsilon_t
\]

This uses only the past of \(Y\).

---

**Formula V4. Unrestricted model for Y using X**

\[
Y_t = c + \sum_{i=1}^{p}a_iY_{t-i} + \sum_{i=1}^{p}b_iX_{t-i}+\epsilon_t
\]

This uses past \(Y\) and past \(X\).

---

**Formula V5. Granger causality null hypothesis**

\[
H_0: b_1=b_2=\cdots=b_p=0
\]

Interpretation:

> If the null is rejected, past \(X\) improves prediction of \(Y\) after controlling for past \(Y\).

Important caution:

> Granger causality is predictive causality, not proof of true causal mechanism.

---

# 11. Outlier Detection

## 11.1 Z-Score

**Formula O1. Z-score**

\[
z_t = \frac{Y_t - \mu}{\sigma}
\]

Sample version:

\[
z_t = \frac{Y_t - \bar{Y}}{s}
\]

Common rule:

\[
|z_t| > 3 \Rightarrow \text{possible outlier}
\]

---

## 11.2 Rolling Z-Score

**Formula O2. Rolling z-score**

\[
z_t = \frac{Y_t - \mu_{t,k}}{\sigma_{t,k}}
\]

Where:

- \(\mu_{t,k}\) is the rolling mean over a window of size \(k\).
- \(\sigma_{t,k}\) is the rolling standard deviation.

Interpretation:

> Rolling z-scores compare a point to its local context rather than the entire series.

---

## 11.3 IQR Outlier Rule

**Formula O3. Interquartile range**

\[
IQR = Q_3 - Q_1
\]

**Formula O4. IQR fences**

\[
Lower = Q_1 - 1.5(IQR)
\]

\[
Upper = Q_3 + 1.5(IQR)
\]

Values below the lower fence or above the upper fence are possible outliers.

---

# 12. Data Streams

## 12.1 Stream Notation

**Formula DS1. Data stream**

\[
S = (x_1, x_2, x_3, \ldots)
\]

A stream may be too large or too fast to store completely.

---

## 12.2 Landmark Window

**Formula DS2. Landmark window**

\[
W_t = \{x_i : i \geq L, i \leq t\}
\]

Where \(L\) is a fixed starting point.

Example:

> All events since midnight.

---

## 12.3 Sliding Window

**Formula DS3. Count-based sliding window**

\[
W_t = \{x_{t-k+1}, \ldots, x_t\}
\]

Where \(k\) is the window size.

Example:

> The most recent 1,000 events.

---

## 12.4 Exponential Decay / Damped Window

**Formula DS4. Exponential decay weight**

\[
w(\Delta t)=\lambda^{\Delta t}
\]

where:

\[
0 < \lambda < 1
\]

Interpretation:

> Older observations receive less weight.

---

# 13. Reservoir Sampling

## 13.1 Reservoir Sampling Selection Probability

**Formula R1. Probability an item remains in reservoir after n items**

\[
P(\text{item selected after } n \text{ items}) = \frac{k}{n}
\]

Where:

- \(k\) is reservoir size.
- \(n\) is the number of stream items observed.

---

## 13.2 Acceptance Probability for New Item

**Formula R2. Probability of accepting item \(i\)**

\[
P(\text{accept item } i) = \frac{k}{i}
\]

When item \(i\) arrives after the reservoir is full, accept it with probability \(k/i\). If accepted, replace a random reservoir item.

---

## 13.3 Replacement Probability

**Formula R3. Probability of replacing a specific reservoir item at step i**

\[
P(\text{specific item replaced}) = \frac{k}{i}\cdot\frac{1}{k}=\frac{1}{i}
\]

Interpretation:

> Each existing reservoir item has equal replacement risk when a new item arrives.

---

# 14. Bloom Filters

## 14.1 Bloom Filter Bit Positions

**Formula B1. Hash-based bit positions**

\[
h_1(x), h_2(x), \ldots, h_k(x) \in \{0,1,\ldots,m-1\}
\]

Where:

- \(m\) is the number of bit slots.
- \(k\) is the number of hash functions.

Insertion sets all positions to 1.

---

## 14.2 Bloom Filter Membership Rule

**Formula B2. Membership test**

\[
x \in BF \quad \text{if all bits } h_1(x),\ldots,h_k(x) \text{ are } 1
\]

Interpretation:

- If any bit is 0: definitely not present.
- If all bits are 1: possibly present.

---

## 14.3 Probability a Bit Remains Zero

**Formula B3. Approximate probability a bit is still zero**

\[
P(bit=0) \approx \left(1-\frac{1}{m}\right)^{kn}
\]

Approximation:

\[
P(bit=0) \approx e^{-kn/m}
\]

Where:

- \(n\) is the number of inserted items.
- \(k\) is the number of hash functions.
- \(m\) is the number of bit slots.

---

## 14.4 Bloom Filter False Positive Rate

**Formula B4. Approximate false positive probability**

\[
P(FP) \approx \left(1-e^{-kn/m}\right)^k
\]

Interpretation:

> False positives increase as more bits become set to 1.

---

## 14.5 Approximate Optimal Number of Hash Functions

**Formula B5. Optimal number of hash functions**

\[
k^* \approx \frac{m}{n}\ln 2
\]

This balances setting enough bits to reduce collisions without making the bit array too full.

---

# 15. Lossy Counting

## 15.1 Bucket Width

**Formula LC1. Lossy counting bucket width**

\[
w = \left\lceil\frac{1}{\epsilon}\right\rceil
\]

Where \(\epsilon\) is the error tolerance.

Interpretation:

- Smaller \(\epsilon\): more accurate, more memory.
- Larger \(\epsilon\): less accurate, less memory.

---

## 15.2 Bucket ID

**Formula LC2. Current bucket number**

\[
b = \left\lceil\frac{N}{w}\right\rceil
\]

Where:

- \(N\) is the number of items processed.
- \(w\) is the bucket width.

---

## 15.3 Stored Count and Error

**Formula LC3. Stored lossy counting record**

\[
(item, f, \Delta)
\]

Where:

- \(f\) is the estimated frequency.
- \(\Delta\) is the maximum possible error.

For a new item in bucket \(b\):

\[
\Delta = b - 1
\]

---

## 15.4 Pruning Rule

**Formula LC4. Lossy counting pruning condition**

\[
f + \Delta \leq b
\]

If this condition holds, the item can be removed.

Interpretation:

> The item is too infrequent to remain a plausible frequent item candidate.

---

## 15.5 Frequent Item Reporting Rule

**Formula LC5. Approximate frequent item threshold**

\[
f \geq (s - \epsilon)N
\]

Where:

- \(s\) is the support threshold.
- \(\epsilon\) is the allowed error.
- \(N\) is the number of processed stream items.

---

# 16. High-Yield Formula Comparison Tables

## 16.1 Probability Models

| Topic | Formula | Main denominator |
|---|---|---|
| Unigram | \(P(w)=count(w)/N\) | all tokens |
| Bigram | \(P(B\mid A)=count(A,B)/count(A)\) | first token/context |
| Trigram | \(P(C\mid A,B)=count(A,B,C)/count(A,B)\) | two-token context |
| Add-one | \((count(c,w)+1)/(count(c)+V)\) | context count plus vocabulary size |

---

## 16.2 Time Series Transformations

| Transformation | Formula | Purpose |
|---|---|---|
| First difference | \(Y_t-Y_{t-1}\) | remove trend |
| Seasonal difference | \(Y_t-Y_{t-m}\) | remove seasonality |
| Simple return | \((Y_t-Y_{t-1})/Y_{t-1}\) | relative change |
| Log return | \(\log(Y_t/Y_{t-1})\) | stabilized relative change |

---

## 16.3 Smoothing Methods

| Method | Formula | Key behavior |
|---|---|---|
| SMA | \(\frac{1}{k}\sum_{i=0}^{k-1}Y_{t-i}\) | equal weights |
| WMA | \(\sum a_iY_{t-i}\) | custom weights |
| EMA | \(\alpha Y_t+(1-\alpha)EMA_{t-1}\) | recent values weighted more |

---

## 16.4 Similarity Measures

| Measure | Formula | Best when |
|---|---|---|
| Euclidean | \(\sqrt{\sum(x_i-y_i)^2}\) | aligned values |
| Manhattan | \(\sum|x_i-y_i|\) | absolute deviations |
| Cosine | \(x\cdot y/(\|x\|\|y\|)\) | direction/shape |
| Correlation | centered cosine | co-movement |
| DTW | dynamic programming recurrence | shifted/stretched patterns |
| Edit distance | insert/delete/substitute recurrence | symbolic sequences |

---

## 16.5 Forecasting Models

| Model | Formula idea | Uses |
|---|---|---|
| AR | past values | momentum/autocorrelation |
| MA | past errors | shock correction |
| ARMA | past values + past errors | stationary series |
| ARIMA | differencing + ARMA | non-stationary series |
| VAR | multivariate lagged values | interacting time series |

---

## 16.6 Stream Methods

| Method | Main formula | Purpose |
|---|---|---|
| Reservoir sampling | \(P=k/n\) | uniform fixed-size sample |
| Bloom filter | \(P(FP)\approx(1-e^{-kn/m})^k\) | approximate membership |
| Lossy counting | \(w=\lceil 1/\epsilon\rceil\) | approximate frequent items |

---

# 17. Formula-Based Self-Test Questions

1. For \(P(\text{dogs}\mid \text{like})\), what goes in the denominator?
2. For \(P(C\mid A,B)\), why is \(count(A,B)\) the denominator?
3. What does \(\alpha\) control in an exponential moving average?
4. Why does \(d\) matter in ARIMA\((p,d,q)\)?
5. What is the difference between Euclidean distance and DTW?
6. In a Bloom filter, why are false positives possible?
7. Why does reservoir sampling give each item probability \(k/n\)?
8. Why is Granger causality not the same as causal inference?
9. What does the PACF measure that the ACF does not isolate?
10. What does \(f+\Delta\leq b\) mean in lossy counting?

---

# 18. Ultra-Condensed Exam Formula Sheet

## N-Grams

\[
P(B\mid A)=\frac{count(A,B)}{count(A)}
\]

\[
P(C\mid A,B)=\frac{count(A,B,C)}{count(A,B)}
\]

\[
P_{add1}(w\mid c)=\frac{count(c,w)+1}{count(c)+V}
\]

## Markov / HMM

\[
P(X_t\mid X_{t-1},\ldots,X_1)=P(X_t\mid X_{t-1})
\]

\[
\pi_{t+k}=\pi_tP^k
\]

\[
P(S,O)=\pi_{s_1}B_{s_1}(o_1)\prod_{t=2}^{T}A_{s_{t-1},s_t}B_{s_t}(o_t)
\]

## Time Series

\[
Y_t=T_t+S_t+R_t
\]

\[
\nabla Y_t=Y_t-Y_{t-1}
\]

\[
r_t=\log(Y_t/Y_{t-1})
\]

\[
EMA_t=\alpha Y_t+(1-\alpha)EMA_{t-1}
\]

## Similarity

\[
d(x,y)=\sqrt{\sum(x_i-y_i)^2}
\]

\[
\cos(x,y)=\frac{x\cdot y}{\|x\|\|y\|}
\]

\[
DTW(i,j)=c(i,j)+\min\{DTW(i-1,j),DTW(i,j-1),DTW(i-1,j-1)\}
\]

## Forecasting

\[
AR(p):Y_t=c+\sum_{i=1}^{p}\phi_iY_{t-i}+\epsilon_t
\]

\[
ARIMA(p,d,q):\nabla^dY_t\sim ARMA(p,q)
\]

\[
VAR(p):\mathbf{Y}_t=\mathbf{c}+A_1\mathbf{Y}_{t-1}+\cdots+A_p\mathbf{Y}_{t-p}+\boldsymbol{\epsilon}_t
\]

## Streams

\[
P(\text{reservoir selected})=\frac{k}{n}
\]

\[
P(FP)\approx(1-e^{-kn/m})^k
\]

\[
w=\left\lceil\frac{1}{\epsilon}\right\rceil
\]

\[
f+\Delta\leq b \Rightarrow \text{prune item}
\]

