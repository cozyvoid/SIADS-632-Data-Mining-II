---
title: "SIADS 632 Data Mining II: Conceptual Deep-Dive Study Guide"
author: "Prepared for SIADS 632 review"
date: "Summer 2026"
geometry: margin=0.75in
fontsize: 10.5pt
---

# How to Use This Guide

This guide is a concept-first companion to the implementation-heavy SIADS 632 assignments. The goal is not only to memorize definitions, but to understand why each method exists, what assumptions it makes, what problem it solves, and how it appears in Python code.

A useful study rhythm is:

1. Read the concept explanation.
2. Say the oral-exam explanation out loud.
3. Write the formula from memory.
4. Sketch the Python pattern without looking.
5. Explain one real-world application.

The course has four broad arcs:

| Unit | Main data representation | Central question |
|---|---|---|
| 1 | Sequences | What comes next in an ordered symbolic sequence? |
| 2 | Time series patterns and similarity | What structure exists over time, and which series are similar? |
| 3 | Forecasting | How can past observations predict future observations? |
| 4 | Data streams | How can we mine data that is too fast or too large to store? |

# 1. Data Representations: The Big Conceptual Map

Data Mining II is largely about choosing the right representation for ordered or arriving data. The same real-world phenomenon can sometimes be represented several ways.

For example, user behavior on a website could be represented as:

- a **sequence** of page visits: `home -> search -> product -> cart`;
- a **time series** of hourly click counts;
- a **data stream** of click events arriving continuously.

The representation determines the method. If the object is an ordered list of symbols, N-grams and Markov chains make sense. If it is numeric and indexed by time, trend extraction, similarity measures, and forecasting become relevant. If it arrives continuously and cannot be fully stored, sampling and synopsis structures become necessary.

## 1.1 Sequence vs. Time Series vs. Data Stream

A **sequence** is an ordered collection of items. The items may be words, clicks, DNA bases, tags, or events. Time may be implicit, but the main fact is order.

A **time series** is a sequence indexed by time, usually numeric. It has timestamps or regular time positions and often exhibits temporal dependence.

A **data stream** is data that arrives continuously, possibly forever. The defining issue is not just order, but constraints: limited memory, limited processing time, and usually one-pass access.

| Representation | Typical item type | Order matters? | Usually finite? | Common methods |
|---|---:|---:|---:|---|
| Sequence | symbolic/discrete | yes | usually yes | N-grams, Markov chains, HMMs |
| Time series | numeric | yes | yes or growing | decomposition, similarity, ARIMA, VAR |
| Data stream | any arriving item | yes | often unbounded | reservoir sampling, Bloom filters, lossy counting |

**Oral-exam explanation:** A sequence is ordered symbolic data; a time series is ordered numeric data indexed by time; a stream is data arriving continuously under memory and time constraints. The representation changes which mining tasks are feasible.

# 2. Sequences and N-Grams

## 2.1 What Makes Sequence Mining Different?

In a standard tabular dataset, rows are often treated as independent observations. In sequence mining, the position of each item matters. The same items in a different order may imply a different meaning.

For example:

```text
"dog bites man" != "man bites dog"
```

The tokens are identical, but the order changes the meaning. This is why sequence models need transition or conditional probability structure.

## 2.2 Tokens, Sentences, and Vocabulary

In the Shakespeare assignment, a sentence is defined operationally as **one line of a sonnet**, not necessarily a grammatical sentence ending in punctuation. That matters because N-grams are generated inside those line boundaries.

A **token** is the unit modeled by the language model. In this course, tokens are often whitespace-separated lowercase strings.

A **vocabulary** is the set of all possible tokens the model knows about. The vocabulary usually includes special boundary symbols:

```text
<s>   sentence start
</s>  sentence end
```

These tokens let a model learn which words tend to start or end a sentence.

## 2.3 Why Padding Exists

For an N-gram model of order `n`, each predicted token depends on `n - 1` previous tokens. Padding gives the first real word enough context.

For a trigram model, the sentence:

```text
shall I compare thee
```

may become:

```text
<s> <s> shall I compare thee </s> </s>
```

The model can now learn probabilities like:

$
P(shall \mid <s>, <s>)
$

and

$
P(</s> \mid thee, </s>)
$

A sentence with length $\ell$, padded with $n-1$ start tokens and $n-1$ end tokens, produces $\ell+n-1$ N-grams. This follows because the padded sequence has length $\ell+2(n-1)$, and a length-$L$ sequence contains $L-n+1$ N-grams:

$$
(\ell+2(n-1))-n+1 = \ell+n-1
$$

## 2.4 N-Gram Models

An N-gram model approximates a full sequence probability by limiting context to the previous $n-1$ tokens.

The exact chain rule is:

$$
P(w_1,w_2,\ldots,w_T)=\prod_{t=1}^{T}P(w_t \mid w_1,\ldots,w_{t-1})
$$

This is usually impossible to estimate directly because each full history may appear rarely or never. The N-gram approximation is:

$$
P(w_t \mid w_1,\ldots,w_{t-1}) \approx P(w_t \mid w_{t-n+1},\ldots,w_{t-1})
$$

So a bigram model uses one previous token:

$$
P(w_t \mid w_{t-1})
$$

A trigram model uses two previous tokens:

$$
P(w_t \mid w_{t-2},w_{t-1})
$$

## 2.5 The Denominator Rule

The probability of a predicted token is always counted relative to its context.

For a bigram:

$$
P(w_i \mid w_{i-1}) = \frac{count(w_{i-1},w_i)}{count(w_{i-1})}
$$

For a trigram:

$$
P(w_i \mid w_{i-2},w_{i-1}) = \frac{count(w_{i-2},w_{i-1},w_i)}{count(w_{i-2},w_{i-1})}
$$

The denominator is not the count of the predicted word. It is the count of the context.

**Why:** The conditional probability asks, "Among all cases where this context occurred, what fraction were followed by this token?"

## 2.6 MLE Interpretation

The course uses `nltk.lm.MLE`, where MLE means **maximum likelihood estimation**. In the N-gram setting, MLE estimates probabilities by relative frequencies.

If the context `I love` appears 10 times and is followed by `thee` 4 times, the MLE estimate is:

$$
P(thee \mid I,love)=\frac{4}{10}=0.4
$$

MLE is simple and intuitive, but it has a major problem: unseen events get probability zero.

## 2.7 Unseen N-Grams and Zero Probability

If a test sentence contains one N-gram not seen in training, an unsmoothed model assigns it probability zero. Then the entire sequence probability becomes zero because sequence probability is a product of conditional probabilities.

This is not the same as saying the sentence is impossible in the real world. It only means the training data did not contain that specific context-token pair.

This is why smoothing exists, even if the assignment focuses on MLE.

**Oral-exam explanation:** N-gram models estimate next-token probabilities from relative counts. The numerator is the count of the full N-gram, and the denominator is the count of the context. MLE is easy to compute but assigns zero probability to unseen N-grams, which can be too harsh for language.

# 3. Markov Chains

## 3.1 Markov Chains as Observable State Models

A Markov chain models a sequence of observable states where the probability of the next state depends only on the current state.

The first-order Markov property is:

$$
P(X_t \mid X_{t-1},X_{t-2},\ldots,X_1)=P(X_t \mid X_{t-1})
$$

This assumption reduces complexity. Instead of needing to estimate probabilities for every possible full history, we only estimate transitions from one state to the next.

## 3.2 Transition Matrices

A Markov chain can be represented by a transition matrix. Each row corresponds to the current state, and each column corresponds to the next state.

Example:

| From / To | Sunny | Rainy |
|---|---:|---:|
| Sunny | 0.8 | 0.2 |
| Rainy | 0.4 | 0.6 |

Each row sums to 1 because the next state must be one of the possible states.

If the current probability distribution over states is a row vector $\pi_t$, the next distribution is:

$$
\pi_{t+1}=\pi_tP
$$

After $k$ steps:

$$
\pi_{t+k}=\pi_tP^k
$$

## 3.3 Bigram Models as Markov Chains

A bigram language model is essentially a first-order Markov chain over tokens. Each word is a state, and transition probabilities are estimated from word-to-word counts.

For example:

$$
P(love \mid I)
$$

is the transition probability from state `I` to state `love`.

A trigram model is not first-order over individual tokens, but it can be viewed as first-order over contexts. For example, the state could be a two-token context such as `(I, love)`.

## 3.4 Strengths and Weaknesses of the Markov Assumption

The Markov assumption makes modeling feasible, but it discards long-range context. In language, this means the model may miss dependencies across many words. In behavior sequences, it may ignore earlier events that still matter.

| Benefit | Cost |
|---|---|
| easy to estimate | limited memory of past |
| computationally simple | may miss long-term dependencies |
| interpretable transitions | context can be too short |

**Oral-exam explanation:** A Markov chain models transitions among observable states. The first-order assumption says the next state depends only on the current state. A bigram language model is a Markov chain because each next word is predicted from only the previous word.

# 4. Hidden Markov Models

## 4.1 Why Hidden States?

A Markov chain assumes the states are observed. But many real processes have hidden states that generate visible observations.

In POS tagging, the visible observation is the word. The hidden state is the part-of-speech tag.

```text
Observed words:  The    dog    runs
Hidden tags:     DT     NN     VBZ
```

The model observes the words and infers the most likely tag sequence.

## 4.2 HMM Components

An HMM has five conceptual components:

| Component | Symbol | Meaning |
|---|---:|---|
| Hidden states | $S$ | possible latent states, such as POS tags |
| Observations | $O$ | visible tokens, such as words |
| Initial probabilities | $\pi$ | probability of starting in each hidden state |
| Transition probabilities | $A$ | probability of moving from one hidden state to another |
| Emission probabilities | $B$ | probability of observing a token from a hidden state |

The transition probability is:

$$
P(s_t \mid s_{t-1})
$$

The emission probability is:

$$
P(o_t \mid s_t)
$$

## 4.3 Two Independence Assumptions

The HMM has two major assumptions.

First, the hidden state sequence is Markov:

$$
P(s_t \mid s_{t-1},s_{t-2},\ldots)=P(s_t \mid s_{t-1})
$$

Second, the observation depends only on the current hidden state:

$$
P(o_t \mid s_t,s_{t-1},o_{t-1},\ldots)=P(o_t \mid s_t)
$$

These assumptions make the model tractable.

## 4.4 POS Tagging with HMMs

In the assignment, the CoNLL-style data contains rows like:

```text
scientists NNS I-NP O
```

For POS tagging, the relevant pieces are usually:

```text
token = scientists
POS tag = NNS
```

The HMM learns:

- which tags are likely to start sentences;
- which tags tend to follow other tags;
- which words are likely to be emitted by each tag.

For example, a determiner tag `DT` may often emit words like `the`, `a`, and `an`.

## 4.5 HMM Problems

Classic HMM tasks:

| Task | Question | Common algorithm |
|---|---|---|
| Evaluation | How likely is this observation sequence? | Forward algorithm |
| Decoding | What hidden states most likely produced this sequence? | Viterbi algorithm |
| Learning | How do we estimate the model? | Baum-Welch / supervised counts |

In the assignment, training is supervised because the POS tags are provided. That means the model can estimate transition and emission probabilities from labeled data.

**Oral-exam explanation:** An HMM models a sequence where the state is not directly observed. In POS tagging, the words are observed and the tags are hidden. The model combines tag transition probabilities with word emission probabilities to infer the most likely tag sequence.

# 5. Time Series Foundations

## 5.1 What Is a Time Series?

A time series is a set of observations ordered by time:

$$
y_1,y_2,\ldots,y_T
$$

Unlike ordinary tabular data, observations are usually dependent. Today's value may depend on yesterday's value, last week's value, or last year's value.

## 5.2 Discrete vs. Continuous Time

A discrete-time series is observed at separate time points:

```text
daily cases, monthly sales, hourly temperature
```

A continuous-time process evolves continuously, even if we sample it at discrete moments:

```text
heart rate signal, audio waveform, physical movement
```

In practice, most data is stored discretely, but it may represent an underlying continuous process.

## 5.3 Time Series Mining Tasks

A time series can support many tasks:

| Task | Question |
|---|---|
| Pattern extraction | Is there trend, seasonality, or outlier behavior? |
| Similarity search | Which series resemble a query series? |
| Clustering | Which series form groups? |
| Classification | What label should this series receive? |
| Forecasting | What future values are likely? |
| Anomaly detection | Which points or subsequences are unusual? |

The COVID assignments focus on pattern extraction, similarity, and forecasting.

# 6. Time Series Patterns

## 6.1 Trend

A trend is a long-term direction in the series. It can be increasing, decreasing, or nonlinear.

In the COVID assignment, cumulative cases were transformed into daily new cases, and then trend patterns were studied. The trend in daily new cases can be nonlinear because pandemic spread is affected by growth, interventions, reporting, and saturation.

A polynomial trend fit approximates the long-term curve:

$$
\hat{y}_t = \beta_0 + \beta_1t + \beta_2t^2 + \cdots + \beta_dt^d
$$

Higher-degree polynomials are more flexible but can overfit.

## 6.2 Seasonality

Seasonality is a regular pattern with a fixed period. Examples include weekly patterns in daily data or yearly patterns in monthly data.

In public health reporting, seasonality can appear because reporting behavior differs by weekday. For example, weekend reporting delays may create lower counts on some days and higher catch-up counts later.

Seasonal decomposition separates a series into components.

Additive decomposition:

$$
y_t = T_t + S_t + R_t
$$

Multiplicative decomposition:

$$
y_t = T_t \times S_t \times R_t
$$

Use additive decomposition when seasonal variation has roughly constant magnitude. Use multiplicative decomposition when seasonal variation grows with the level of the series.

## 6.3 Cyclic and Irregular Patterns

A cycle is a repeated rise and fall without a fixed calendar period. Business cycles are classic examples. Irregular variation is residual noise or unexplained fluctuation.

| Component | Fixed period? | Example |
|---|---:|---|
| Seasonality | yes | every weekday/weekend pattern |
| Cycle | no | economic expansion and recession |
| Irregular | no | random shocks, data glitches |

## 6.4 Outliers

A time series outlier is unusual relative to context. A high value may not be an outlier during a rising trend, but it might be an outlier in a stable local window.

Common types:

| Type | Description |
|---|---|
| Point outlier | one abnormal value |
| Contextual outlier | abnormal given the time context |
| Subsequence outlier | abnormal run of values |

A rolling z-score is often more meaningful than a global z-score because it compares a point to recent local behavior.

**Oral-exam explanation:** Time series patterns include trend, seasonality, cycles, irregular variation, and outliers. Trend is long-term direction; seasonality repeats with fixed period; cycles repeat irregularly; outliers are unusual relative to a global or local context.

# 7. Smoothing: Moving Averages and EMA

## 7.1 Why Smooth a Time Series?

Smoothing reduces high-frequency noise so lower-frequency structure becomes easier to see. It is not primarily a forecasting method in this course; it is also a descriptive tool for pattern extraction.

The tradeoff is responsiveness versus smoothness. A smoother curve is easier to interpret but reacts more slowly to real changes.

## 7.2 Simple Moving Average

A simple moving average of window size \(k\) is:

$$
MA_t = \frac{1}{k}\sum_{i=0}^{k-1}y_{t-i}
$$

Each observation in the window gets equal weight.

A larger window gives more smoothing but more lag. A smaller window gives less smoothing but responds more quickly.

## 7.3 Weighted Moving Average

A weighted moving average assigns different weights within the window:

$$
WMA_t = \sum_{i=0}^{k-1}w_iy_{t-i}
$$

where the weights usually sum to 1:

$$
\sum_{i=0}^{k-1}w_i=1
$$

The assignment's weighted moving average requires careful alignment: the weights must correspond to the values in each rolling window in the expected order.

## 7.4 Exponential Moving Average

The exponential moving average recursively combines the current value with the previous EMA:

$$
EMA_t = \alpha y_t + (1-\alpha)EMA_{t-1}
$$

The parameter $\alpha$ controls responsiveness.

| Alpha | Behavior |
|---:|---|
| large | reacts quickly, less smooth |
| small | reacts slowly, smoother |

The EMA uses all previous observations indirectly, but older observations receive exponentially decreasing influence.

**Oral-exam explanation:** Moving averages smooth a series by averaging nearby values. A simple moving average weights all observations in the window equally, while an exponential moving average weights recent observations more heavily and therefore reacts faster to changes.

# 8. Time Series Similarity and Alignment

## 8.1 Why Similarity Is Hard for Time Series

Two time series can be similar in shape but misaligned in time. For example, two countries may show similar epidemic waves, but one wave may occur later. Point-by-point distance may treat them as dissimilar even when the pattern is similar.

## 8.2 Euclidean Distance

Euclidean distance compares aligned values directly:

$$
d(x,y)=\sqrt{\sum_{t=1}^{T}(x_t-y_t)^2}
$$

It works best when:

- both series have equal length;
- observations are aligned in time;
- magnitude differences matter.

It is sensitive to scale and temporal shifts.

## 8.3 Cosine Similarity

Cosine similarity compares the angle between two vectors:

$$
cos(x,y)=\frac{x\cdot y}{\|x\|\|y\|}
$$

It is often interpreted as shape or direction similarity rather than magnitude similarity. Two series can have high cosine similarity even if one has larger absolute values, as long as their patterns move in similar proportions.

## 8.4 Dynamic Time Warping

Dynamic Time Warping, or DTW, allows flexible alignment. Instead of forcing $x_t$ to match $y_t$, DTW finds a low-cost path through an alignment matrix.

At each matrix cell, DTW considers three possible previous moves:

- insertion;
- deletion;
- match/diagonal.

The recurrence is commonly:

$$
D(i,j)=cost(x_i,y_j)+\min\{D(i-1,j),D(i,j-1),D(i-1,j-1)\}
$$

DTW is useful when two series have similar shapes but the timing differs.

## 8.5 DTW vs. Edit Distance

DTW and edit distance are both dynamic programming methods for ordered data.

| Method | Typical data | Operation idea |
|---|---|---|
| DTW | numeric time series | align values with minimum warping cost |
| Edit distance | symbolic sequences | insert, delete, substitute symbols |

The assignment connects DTW and edit distance because both solve alignment problems.

**Oral-exam explanation:** Euclidean distance compares time series point by point, so it assumes alignment. DTW relaxes this assumption by allowing one point in one series to align with one or more nearby points in another series. This helps when patterns occur at different speeds or times.

# 9. Symbolic Time Series Representations

Symbolic representations convert numeric time series into discrete symbols. This lets methods from sequence mining be applied to time series.

For example:

```text
Raw values:  3.1, 3.3, 8.7, 8.9, 4.2
Symbols:    low, low, high, high, medium
```

Why symbolic representation helps:

- reduces dimensionality;
- reduces sensitivity to small numeric noise;
- supports sequence comparison and pattern mining;
- can make long series easier to index or search.

A simple method is binning. More sophisticated symbolic methods can use segment averages and breakpoints. The conceptual tradeoff is information loss versus simplicity.

**Oral-exam explanation:** Symbolic representation discretizes a numeric time series into symbols. This can reduce noise and dimensionality, but it also loses numeric detail. It is useful when we want to apply sequence mining ideas to time series.

# 10. Stationarity and Predictability

## 10.1 What Is Stationarity?

A stationary time series has stable statistical properties over time. Informally, its mean, variance, and autocorrelation structure do not systematically change.

Many forecasting models assume or work better with stationarity because the relationship learned from the past should remain useful in the future.

A non-stationary series may have:

- trend;
- seasonality;
- changing variance;
- structural breaks.

## 10.2 Differencing

First-order differencing transforms levels into changes:

$$
\Delta y_t = y_t-y_{t-1}
$$

This can remove trend and make the series more stationary.

Seasonal differencing uses a seasonal lag \(s\):

$$
\Delta_s y_t = y_t-y_{t-s}
$$

In the forecasting assignments, first-order differences are used before ARMA/VAR modeling, and forecasts are converted back to the original data space by cumulative summation from the last observed value.

## 10.3 Log Returns

A log return is:

$$
r_t=\log(y_t)-\log(y_{t-1})
$$

Log returns approximate percentage changes and can stabilize variance when values grow multiplicatively.

## 10.4 Measuring Predictability

A series is predictable when past values contain information about future values. Autocorrelation is a key measure.

Lag-$k$ autocorrelation compares $y_t$ with $y_{t-k}$. High autocorrelation means past values may help predict future values.

**Oral-exam explanation:** Stationarity means the series has stable statistical behavior over time. Differencing and log returns are transformations that often make a series easier to model. Predictability is related to whether past observations contain information about future observations, often measured through autocorrelation.

# 11. ACF and PACF

## 11.1 Autocorrelation Function

The ACF measures correlation between a series and its lagged values:

$$
Corr(y_t,y_{t-k})
$$

If the ACF decays slowly, the series may be non-stationary or strongly persistent. If it has spikes at seasonal lags, the series may have seasonality.

## 11.2 Partial Autocorrelation Function

The PACF measures the direct relationship between $y_t$ and $y_{t-k}$ after controlling for intermediate lags.

Conceptually:

- ACF asks: "How correlated are values k steps apart?"
- PACF asks: "How much direct correlation remains at lag k after accounting for shorter lags?"

## 11.3 Why ACF and PACF Matter

They help diagnose autoregressive and moving-average structure.

A common heuristic is:

| Pattern | Possible implication |
|---|---|
| PACF cuts off after lag p | AR(p) may be appropriate |
| ACF cuts off after lag q | MA(q) may be appropriate |
| both decay | ARMA may be appropriate |

These are heuristics, not guarantees.

**Oral-exam explanation:** ACF measures overall correlation with past values, while PACF measures the direct correlation at a lag after controlling for shorter lags. They help identify whether past values or past errors may be useful in a forecasting model.

# 12. Forecasting: AR, MA, ARMA, and ARIMA

## 12.1 Autoregression

An AR(p) model predicts the current value from previous values:

$$
y_t = c + \phi_1y_{t-1}+\phi_2y_{t-2}+\cdots+\phi_py_{t-p}+\epsilon_t
$$

The parameter $p$ is the number of lagged values used.

## 12.2 Moving-Average Error Models

An MA(q) model predicts the current value from past forecast errors:

$$
y_t = c + \epsilon_t + \theta_1\epsilon_{t-1}+\cdots+\theta_q\epsilon_{t-q}
$$

This is different from a moving average smoother. In ARIMA terminology, MA refers to modeling error dependence, not simply rolling averages.

## 12.3 ARMA

ARMA combines autoregressive terms and moving-average error terms:

$$
y_t = c + \sum_{i=1}^{p}\phi_iy_{t-i}+\epsilon_t+\sum_{j=1}^{q}\theta_j\epsilon_{t-j}
$$

ARMA assumes the modeled series is stationary.

## 12.4 ARIMA

ARIMA adds differencing:

$$
ARIMA(p,d,q)
$$

| Parameter | Meaning |
|---|---|
| p | autoregressive lag order |
| d | differencing order |
| q | moving-average error order |

In the assignments, the code often fits `ARIMA(..., order=(p, 0, q))` to an already transformed series such as first differences or log returns. In that case, the differencing has been done manually, so `d=0` inside the model.

## 12.5 Forecasting Back to Original Scale

If you model first differences, the model forecasts future changes. To recover forecasted levels:

$$
\hat{y}_{T+h}=y_T+\sum_{i=1}^{h}\widehat{\Delta y}_{T+i}
$$

This is why the assignment uses cumulative sums of forecasted differences added to the last observed value.

**Oral-exam explanation:** AR uses past values; MA uses past forecast errors; ARMA combines both for stationary series; ARIMA adds differencing for non-stationary series. If we forecast differences, we must cumulatively add those predicted changes back to the last observed level.

# 13. Multivariate Forecasting: VAR

## 13.1 Why VAR Exists

A univariate AR model predicts one series using only its own past. But many real systems contain multiple related series. VAR models each variable using lagged values of all variables.

For five countries' COVID case series, a VAR model can ask whether past case changes in one country help predict future case changes in another.

## 13.2 VAR(p) Concept

For two variables, a VAR(1) might look like:

$$
y_t = a_1 + b_{11}y_{t-1}+b_{12}x_{t-1}+e_{1t}
$$

$$
x_t = a_2 + b_{21}y_{t-1}+b_{22}x_{t-1}+e_{2t}
$$

Each variable gets its own equation, and each equation can use lagged values of all variables.

## 13.3 VAR vs. Independent AR

| Model | Uses own past? | Uses other series' past? |
|---|---:|---:|
| Independent AR | yes | no |
| VAR | yes | yes |

A VAR model may improve forecasts if the series contain predictive information about each other. But it can also overfit if there are too many variables or lags relative to the amount of data.

## 13.4 Forecasting Differences with VAR

Like ARIMA, VAR often works better on stationary transformed data. The assignment fits VAR on first-order differences, then converts forecasts back to original scale.

The conceptual steps are:

1. Difference the multivariate time series.
2. Fit VAR(p) on the differenced data.
3. Forecast future differences.
4. Cumulatively add the differences to the last observed original values.

**Oral-exam explanation:** VAR extends autoregression to multiple time series. Instead of each series depending only on its own lagged values, each series can depend on lagged values of all variables. This is useful when variables may contain predictive information about each other.

# 14. Granger Causality

## 14.1 Core Idea

Granger causality is based on predictive improvement.

Variable $X$ Granger-causes variable $Y$ if past values of $X$ improve prediction of $Y$ beyond what past values of $Y$ already provide.

The null hypothesis is usually:

$$
X \text{ does not Granger-cause } Y
$$

A small p-value suggests rejecting the null, meaning the lagged values of $X$ add predictive information for $Y$.

## 14.2 Direction Matters

Granger causality is directional. $X$ may Granger-cause $Y$, while $Y$ may not Granger-cause $X$.

That is why the assignment's Granger p-value matrix is not necessarily symmetric.

Rows represent the caused variable. Columns represent the causing variable.

## 14.3 Predictive Causality vs. True Causality

Granger causality does not prove true causal mechanism. It only shows that one variable's past improves prediction of another variable under the model and lag structure.

Possible reasons for Granger causality include:

- true causal influence;
- shared external drivers;
- reporting delays;
- temporal ordering artifacts;
- omitted variables.

**Oral-exam explanation:** Granger causality tests whether lagged values of one series improve prediction of another series after accounting for the target series' own past. It is a predictive notion of causality, not proof of true causal effect.

# 15. Data Streams

## 15.1 What Makes Streams Different?

A data stream is a sequence of items arriving continuously. Unlike a fixed dataset, a stream may be too large to store and may never end.

Stream mining constraints:

- one-pass access;
- limited memory;
- limited processing time per item;
- unknown stream length;
- changing distribution over time.

This forces approximate methods.

## 15.2 Exactness vs. Feasibility

Traditional batch mining often aims for exact answers after scanning all data. Stream mining often aims for approximate answers using compact summaries.

| Batch data | Stream data |
|---|---|
| can often store all data | cannot assume full storage |
| can scan repeatedly | often one pass |
| exact algorithms more feasible | approximate algorithms common |
| static distribution more plausible | concept drift possible |

## 15.3 Window Models

Streams may be summarized using windows.

| Window | Meaning |
|---|---|
| Landmark | from fixed start time to now |
| Sliding | most recent k items or time period |
| Damped | recent items have more weight |

The assignments use stream simulation with tweets and emoji extraction. The key idea is not Twitter specifically, but item processing under limited storage.

**Oral-exam explanation:** Data streams require algorithms that process items quickly, often in one pass, without storing everything. Because exact storage and repeated scans are infeasible, stream mining uses sampling, hashing, and compact summaries.

# 16. Sampling Streams

## 16.1 Random Sampling

Simple random sampling in a stream can include each item independently with some probability $p$. This is easy but the final sample size is random.

In the Twitter emoji assignment, the sample is not just a list of tweets; it also supports approximate counts of emojis in the sampled tweets. That means adding or removing a tweet requires updating the emoji count dictionary.

## 16.2 Reservoir Sampling

Reservoir sampling maintains a fixed-size sample of $k$ items from a stream of unknown length.

The algorithm:

1. Store the first $k$ items.
2. For item $i$ after the reservoir is full, include it with probability $k/i$.
3. If included, replace a randomly chosen item in the reservoir.

After seeing $n$ items, every item has probability:

$$
\frac{k}{n}
$$

of being in the reservoir.

## 16.3 Why Reservoir Sampling Is Fair

Early items enter automatically, but they can be replaced later. Later items have a lower probability of entering because there are more total items, but if they enter, they replace an existing item uniformly.

This balances early and late arrivals so each stream item has equal final inclusion probability.

## 16.4 Implementation Concept: Count Maintenance

If the reservoir stores tweets and the assignment asks for emoji counts in the sample, replacement has two operations:

1. Remove the old tweet's emojis from counts.
2. Add the new tweet's emojis to counts.

If an emoji count reaches zero, delete the key. This matters for autograders because `{"smile": 0}` is not equivalent to omitting `"smile"` if the expected dictionary excludes zero counts.

**Oral-exam explanation:** Reservoir sampling keeps a fixed-size uniform sample from a stream without knowing the stream length. Each item has equal probability of remaining in the reservoir after processing the stream, and replacements require careful updates to any derived counts.

# 17. Synopsis Structures and Bloom Filters

## 17.1 Synopsis Structures

A synopsis structure is a compact summary of a stream. It stores enough information to answer a query approximately without storing the full data.

Examples:

| Synopsis | Query supported |
|---|---|
| Reservoir sample | approximate distribution or downstream analysis |
| Bloom filter | membership query |
| Lossy counter | approximate frequent items |
| Histogram | approximate distribution |
| Sketch | approximate counts or moments |

The major theme is memory reduction.

## 17.2 Bloom Filter Purpose

A Bloom filter answers membership questions:

> Has this item appeared before?

It uses a bit array and multiple hash functions.

Insert item:

1. Hash the item with each hash function.
2. Set the corresponding slots to 1.

Check item:

1. Hash the item with each hash function.
2. If any slot is 0, the item definitely has not appeared.
3. If all slots are 1, the item possibly appeared.

## 17.3 False Positives but No False Negatives

A Bloom filter can say an item possibly appeared even if it did not. This happens when other items set all the same slots by coincidence.

But a standard insertion-only Bloom filter should not say "not appeared" for an inserted item, because every inserted item has all its hash slots set to 1.

| Bloom filter answer | Correct interpretation |
|---|---|
| any checked slot is 0 | definitely not present |
| all checked slots are 1 | possibly present |

## 17.4 Memory and Hash Tradeoff

More slots generally reduce collisions. More hash functions can reduce false positives up to a point, but too many hash functions fill the array too quickly and increase computation.

**Oral-exam explanation:** A Bloom filter is a compact probabilistic membership structure. It can produce false positives but not false negatives in the standard insertion-only setting. It saves memory by storing hashed fingerprints rather than the items themselves.

# 18. Lossy Counting

## 18.1 Why Lossy Counting Exists

In a stream, exact counting of every distinct item may require too much memory. Lossy counting approximates frequent items by maintaining counts for candidate items and dropping items that are unlikely to remain important.

The assignment's version emphasizes bucket-based updating for emojis in tweets.

## 18.2 Bucket Intuition

Instead of updating the summary after every individual item, lossy counting can process items in buckets. At bucket boundaries, counts are updated and stale or low-frequency items are decreased or removed.

The purpose is to gradually forget less frequent or less recent items while retaining high-frequency items.

## 18.3 Error-Memory Tradeoff

Lossy counting is controlled by parameters such as bucket size or error tolerance. More aggressive pruning saves memory but may lose more detail. Less aggressive pruning preserves more candidates but uses more memory.

| Choice | Effect |
|---|---|
| smaller bucket / lower error | more accurate, more memory/work |
| larger bucket / higher error | less accurate, less memory/work |

## 18.4 Implementation Concepts

The assignment highlights several implementation details:

- counts update only after a full bucket arrives;
- increments come from emojis in the new bucket;
- decrements remove influence of stale/less frequent items;
- zero-count items should be removed;
- the generator should yield at every stream iteration.

The conceptual algorithm is:

1. Accumulate stream items into a bucket.
2. When the bucket is full, update counts.
3. Decrement existing counts as specified.
4. Remove entries with zero counts.
5. Continue processing without storing the full stream.

**Oral-exam explanation:** Lossy counting approximates frequent item counts in a stream. It keeps a compact set of candidate counts and periodically removes or reduces items that are not frequent enough. The tradeoff is controlled by how much approximation we allow in exchange for memory savings.

# 19. Common Conceptual Comparisons

## 19.1 N-Gram vs. Markov Chain vs. HMM

| Method | Observed states? | Hidden states? | Main purpose |
|---|---:|---:|---|
| N-gram | yes | no | next-token prediction |
| Markov chain | yes | no | transition modeling among states |
| HMM | observations visible, states hidden | yes | infer hidden state sequence |

A bigram model is a Markov chain over words. An HMM is like a Markov chain over hidden states plus an observation model.

## 19.2 Moving Average vs. ARIMA MA Term

These are easy to confuse.

| Term | Meaning |
|---|---|
| Moving average smoother | averages nearby observed values to smooth noise |
| MA term in ARIMA | models dependence in past forecast errors |

A rolling moving average is descriptive smoothing. The MA part of ARIMA is a stochastic model component.

## 19.3 Euclidean vs. Cosine vs. DTW

| Measure | Focus | Weakness |
|---|---|---|
| Euclidean | pointwise magnitude difference | sensitive to shifts and scale |
| Cosine | direction/shape | ignores some magnitude differences |
| DTW | flexible temporal alignment | more computationally expensive; may over-align |

## 19.4 AR vs. VAR vs. Granger

| Concept | Meaning |
|---|---|
| AR | one series predicted from its own past |
| VAR | multiple series predicted from all series' pasts |
| Granger causality | tests whether one series' past improves prediction of another |

## 19.5 Reservoir Sampling vs. Bloom Filter vs. Lossy Counting

| Method | Question answered | Output |
|---|---|---|
| Reservoir sampling | Can we keep a representative fixed-size sample? | sampled items |
| Bloom filter | Has this item appeared before? | definitely no / possibly yes |
| Lossy counting | Which items are frequent? | approximate counts/candidates |

# 20. High-Yield Oral Exam Answers

## 20.1 Why is a bigram model a first-order Markov chain?

A bigram model predicts the next token using only the immediately previous token. That is exactly the first-order Markov assumption: the next state depends only on the current state, not the entire history. In this view, each token is a state and the bigram probabilities are transition probabilities.

## 20.2 Why do we add `<s>` and `</s>` tokens?

They let the model learn how sentences begin and end. Without start padding, the first real token would not have enough context for an N-gram model. Without end tokens, the model would not explicitly learn when a sequence should terminate.

## 20.3 What is the difference between a Markov chain and an HMM?

In a Markov chain, the states are directly observed. In an HMM, the state sequence is hidden and only emissions are observed. For POS tagging, the words are observed and the POS tags are hidden states inferred by the model.

## 20.4 Why does stationarity matter?

Many forecasting models assume that relationships learned from the past will remain stable in the future. If the mean, variance, or autocorrelation structure changes over time, then the past may not be a reliable guide. Transformations like differencing can make a series more stationary.

## 20.5 Why is Granger causality not true causal proof?

Granger causality is based on predictive improvement. If past values of X improve prediction of Y, X Granger-causes Y. But this could happen because of shared causes, omitted variables, or timing artifacts. It does not establish an intervention-based causal effect.

## 20.6 Why are stream algorithms approximate?

Streams may be too large or fast to store, and they may only be available in one pass. Exact algorithms often require full storage or repeated scans. Approximate methods use compact summaries to trade some accuracy for feasible memory and time.

# 21. Concept Self-Test Questions

1. Why does an N-gram model need padding tokens?
2. In a trigram model, what is the denominator for $P(w_3 \mid w_1,w_2)$?
3. Why is a bigram model a first-order Markov chain?
4. What is hidden in an HMM POS tagger?
5. What is the difference between transition and emission probabilities?
6. What is the difference between trend and seasonality?
7. When would multiplicative decomposition be more appropriate than additive decomposition?
8. How does EMA differ from a simple moving average?
9. Why might Euclidean distance fail for similar but shifted time series?
10. What problem does DTW solve?
11. What does differencing do to a time series?
12. What is the difference between ACF and PACF?
13. What do the p, d, and q parameters mean in ARIMA?
14. Why fit VAR on first differences?
15. What does a small Granger causality p-value suggest?
16. Why does reservoir sampling not need to know stream length?
17. Why can a Bloom filter have false positives?
18. Why should a standard Bloom filter not have false negatives?
19. What is the purpose of lossy counting?
20. What is the memory-accuracy tradeoff in stream mining?

# 22. Final Concept Map

```text
SIADS 632 Data Mining II
|
|-- Sequences
|   |-- tokens, vocabulary, padding
|   |-- N-grams
|   |-- MLE conditional probabilities
|   |-- Markov chains
|   |-- Hidden Markov Models
|
|-- Time Series Patterns
|   |-- trend
|   |-- seasonality
|   |-- cycles and irregularity
|   |-- moving averages
|   |-- exponential moving averages
|   |-- symbolic representations
|
|-- Time Series Similarity
|   |-- Euclidean distance
|   |-- cosine similarity
|   |-- DTW
|   |-- edit distance connection
|
|-- Forecasting
|   |-- stationarity
|   |-- differencing
|   |-- log returns
|   |-- ACF and PACF
|   |-- AR, MA, ARMA, ARIMA
|   |-- VAR
|   |-- Granger causality
|
|-- Data Streams
    |-- one-pass constraints
    |-- sampling
    |-- reservoir sampling
    |-- synopsis structures
    |-- Bloom filters
    |-- lossy counting
```

# 23. What to Prioritize Before an Exam

Highest conceptual priority:

1. N-gram probability denominator logic.
2. Bigram model as a first-order Markov chain.
3. Markov chain vs. HMM.
4. Trend vs. seasonality vs. cycle.
5. Moving average vs. exponential moving average.
6. Euclidean distance vs. cosine similarity vs. DTW.
7. Stationarity, differencing, and log returns.
8. ACF vs. PACF.
9. ARIMA parameter meanings.
10. VAR vs. independent AR.
11. Granger causality as predictive, not definitive causal, evidence.
12. Stream constraints and why approximation is necessary.
13. Reservoir sampling fairness.
14. Bloom filter false positives and no false negatives.
15. Lossy counting as approximate frequent-item tracking.

If you can explain these clearly and connect each one to a Python implementation pattern, you are prepared for both conceptual and code-based questions.
