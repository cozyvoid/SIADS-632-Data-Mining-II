# Detailed Overview: Chapter 14 — Mining Time Series Data

**Source:** Charu C. Aggarwal, *Data Mining: The Textbook*, Chapter 14, “Mining Time Series Data.”  
**Core theme:** Time series data mining requires methods that explicitly account for temporal context. Unlike ordinary multidimensional data, the order and timing of values matter.

---

## 1. Big Picture of the Chapter

Chapter 14 focuses on **continuous temporal data**, called **time series data**. A time series records one or more numeric measurements over time. Examples include stock prices, heart-rate signals, ECG/EEG readings, environmental sensor readings, and machine-monitoring data.

The chapter distinguishes between:

- **Time series data:** continuous numeric measurements over time.
- **Sequence data:** discrete events over time, such as clicks, words, purchases, or symbolic actions.

Although time series and sequence data are related, they usually require different algorithms. However, the chapter also explains that time series can be converted into discrete sequences through discretization methods such as **SAX**, allowing sequence-mining algorithms to be used.

The chapter’s major tasks are:

1. Preparing and transforming time series.
2. Measuring similarity between time series.
3. Forecasting future values.
4. Discovering repeated motifs.
5. Clustering time series.
6. Detecting point and shape outliers.
7. Classifying time series or events within time series.

---

## 2. Time Series as Contextual Data

A central idea in the chapter is that time series data is a form of **contextual data**.

### 2.1 Contextual Attribute

The **contextual attribute** gives the reference point for a measurement. In time series data, the contextual attribute is usually **time**.

Examples:

- timestamp
- time index
- trading day
- sensor tick
- second, minute, hour, day, or year

The time dimension tells us *where* or *when* the behavioral value occurred.

### 2.2 Behavioral Attribute

The **behavioral attribute** is the actual numeric value measured at each time point.

Examples:

- temperature
- stock price
- blood pressure
- heart rate
- pressure reading
- sales volume

The behavioral value cannot be interpreted properly without the contextual attribute. For example, a temperature of 100°F means something different depending on whether it occurs in a human patient, a room, or a furnace.

### 2.3 Univariate vs. Multivariate Time Series

A **univariate time series** has one behavioral attribute at each timestamp.

Example:

```text
Time:        1   2   3   4   5
Temperature 70  71  73  72  74
```

A **multivariate time series** has multiple behavioral attributes at each timestamp.

Example:

```text
Time:          1    2    3    4
Temperature   70   71   73   72
Pressure      30   31   29   32
Humidity      45   46   48   47
```

The dimensionality of a multivariate time series equals the number of behavioral attributes tracked at each timestamp.

---

## 3. Real-Time vs. Retrospective Analysis

The chapter separates time series analysis into two broad settings.

### 3.1 Real-Time Analysis

In **real-time analysis**, data arrives continuously and must be analyzed as it arrives. Usually, only a recent window of data is used.

Common real-time tasks include:

- forecasting the next value
- detecting sudden deviations
- detecting events
- clustering streams that are currently moving together

Example: monitoring ICU patient data to detect dangerous changes as soon as they happen.

### 3.2 Retrospective Analysis

In **retrospective analysis**, the entire time series is already available. The goal is to analyze the stored series after the fact.

Common retrospective tasks include:

- finding repeated motifs
- clustering stored ECG traces
- identifying unusual historical patterns
- classifying full time series by shape

Example: comparing ECG recordings from many patients to group similar heart-rhythm patterns.

---

## 4. Time Series Preparation and Similarity

Before applying data mining algorithms, time series data often needs preprocessing. The chapter highlights four major preparation tasks:

1. Handling missing values.
2. Removing noise.
3. Normalizing scales.
4. Transforming or reducing the data.

---

## 5. Handling Missing Values

Time series data often has missing, irregularly spaced, or unsynchronized values. This is especially common when readings are collected from independent sensors.

### 5.1 Linear Interpolation

The most common method discussed is **linear interpolation**.

If we know two points:

- value `yi` at time `ti`
- value `yj` at time `tj`

then a missing value at time `t` between `ti` and `tj` is estimated by assuming the value changes linearly between the two known points.

Formula:

```text
y = yi + ((t - ti) / (tj - ti)) * (yj - yi)
```

Plain-English version:

> Start at the earlier value, then add the fraction of the total change that corresponds to how far the missing timestamp is between the two known timestamps.

### 5.2 Why Interpolation Matters

Interpolation allows the analyst to create time series that are:

- equally spaced
- synchronized across variables
- free of missing values

This is important because many time series algorithms assume aligned timestamps.

---

## 6. Noise Removal

Time series often contain short-term random fluctuations caused by measurement error or noisy sensors. The chapter discusses three smoothing methods:

1. Binning / Piecewise Aggregate Approximation.
2. Moving-average smoothing.
3. Exponential smoothing.

A key caution is that **noise removal can remove meaningful outliers**. Smoothing is useful for trend analysis, but it may be harmful for outlier detection if the “noise” is actually the signal of interest.

---

## 7. Binning / Piecewise Aggregate Approximation

**Binning** divides the time series into non-overlapping intervals of equal size and replaces each bin with a summary value, usually the mean.

Example:

```text
Original series: 2, 7, 5, 3, 3, 5, 5, 3
Bin size = 2
Bins: (2,7), (5,3), (3,5), (5,3)
Binned means: 4.5, 4, 4, 4
```

### 7.1 Advantages

- Reduces noise.
- Compresses the series.
- Speeds up distance computations.

### 7.2 Disadvantages

- Loses data points.
- Can be very lossy for large bin sizes.
- May hide short-lived but important events.

The chapter also notes that binning is often called **Piecewise Aggregate Approximation**, or **PAA**.

---

## 8. Moving-Average Smoothing

A **moving average** computes averages over overlapping windows.

Example with window size 2:

```text
Original series: 2, 7, 5, 3, 3, 5, 5, 3
Windows: (2,7), (7,5), (5,3), (3,3), (3,5), (5,5), (5,3)
Moving averages: 4.5, 6, 4, 3, 4, 5, 4
```

### 8.1 Key Properties

- It smooths short-term fluctuations.
- It uses overlapping windows, so it loses fewer points than binning.
- It introduces lag because a moving average depends on a window of observations.

### 8.2 Important Warning

Larger moving-average windows create more smoothing but also more lag. This can make the smoothed curve appear to turn later than the original series. In some cases, peaks and troughs can be shifted enough to mislead interpretation.

### 8.3 Figure 14.1: Smoothing IBM Stock Prices

The chart on page 5 compares actual IBM stock prices with 20-day and 50-day moving averages. The 50-day moving average is smoother but lags more than the 20-day moving average. The same page also compares exponential smoothing with different alpha values, showing that exponential smoothing preserves all starting points and usually reduces lag compared with moving averages.

---

## 9. Exponential Smoothing

**Exponential smoothing** combines the current value with the previous smoothed value.

Formula:

```text
y'_i = α * yi + (1 - α) * y'_(i-1)
```

Where:

- `yi` = current observed value
- `y'_i` = current smoothed value
- `y'_(i-1)` = previous smoothed value
- `α` = smoothing parameter between 0 and 1

Plain-English version:

> The new smoothed value is part current observation and part previous smoothed history.

### 9.1 Meaning of Alpha

- If `α = 1`, there is no smoothing.
- If `α = 0`, the smoothed series becomes constant.
- Larger `α` gives more weight to recent observations.
- Smaller `α` gives more weight to older smoothed history.

### 9.2 Why It Is Called “Exponential”

Older observations receive exponentially decreasing weight. Recent observations matter more, and older observations fade away gradually.

### 9.3 Advantages Over Moving Average

- Does not lose early points.
- Gives more importance to recent observations.
- Usually reduces lag for the same amount of smoothing.

---

## 10. Normalization

Normalization is necessary when series are measured on different scales.

Example:

- temperature might range from 50 to 100
- pressure might range from 1,000 to 10,000
- stock prices might range from 10 to 1,000

Without normalization, large-scale variables can dominate distance calculations.

### 10.1 Range-Based Normalization

Formula:

```text
y'_i = (yi - min) / (max - min)
```

This rescales values into the range 0 to 1.

### 10.2 Standardization

Formula:

```text
zi = (yi - μ) / σ
```

Where:

- `μ` = mean of the series
- `σ` = standard deviation of the series

This converts each value into a z-score.

### 10.3 Preferred Method

The chapter states that **standardization is generally preferred**, though it does not guarantee a fixed range.

---

## 11. Data Transformation and Reduction

Time series can be long, noisy, and highly redundant. Transformation methods reduce the series into smaller representations while preserving important structure.

The chapter discusses three major approaches:

1. Discrete Wavelet Transform.
2. Discrete Fourier Transform.
3. Symbolic Aggregate Approximation.

---

## 12. Discrete Wavelet Transform

The **Discrete Wavelet Transform**, or **DWT**, represents a time series using coefficients that capture changes in local time regions.

### 12.1 Main Idea

Wavelets are useful when important variation is **local**, meaning it occurs in specific portions of the series.

Examples:

- sudden stock-market crash
- brief abnormal heart rhythm
- temporary machine malfunction
- local spike in sensor readings

### 12.2 Why DWT Is Useful

DWT preserves temporal locality. Instead of treating every timestamp as an unrelated dimension, wavelet coefficients describe changes over contiguous time intervals.

### 12.3 Dimensionality Reduction

Only the largest wavelet coefficients may be needed to reconstruct the original series approximately. This makes DWT useful for compression, denoising, motif discovery, and classification.

---

## 13. Discrete Fourier Transform

The **Discrete Fourier Transform**, or **DFT**, represents a time series as a combination of sinusoidal waves with different frequencies.

### 13.1 Main Idea

DFT is useful when the series contains **global periodicity**.

Examples:

- daily temperature cycles
- weekly traffic patterns
- seasonal sales
- periodic machine vibrations

### 13.2 Figure 14.2: DWT vs. DFT

The chart on page 7 contrasts two types of time series patterns. One series is dominated by local variations, which is better suited for DWT. The other series is decomposable into periodic variations, which is better suited for DFT.

### 13.3 Fourier Coefficients

The DFT converts a time series into complex-valued coefficients. Coefficients with large energy capture important periodic components.

Energy of a coefficient:

```text
energy = a_k^2 + b_k^2
```

where `a_k` is the real part and `b_k` is the imaginary part.

### 13.4 Why DFT Is Useful

DFT is useful for:

- dimensionality reduction
- periodic pattern detection
- fast distance computations
- compression
- frequency-domain analysis

### 13.5 Parseval’s Theorem

The chapter notes that DFT preserves distance relationships in a scaled way. This allows Euclidean distances between time series to be computed through their Fourier coefficients.

---

## 14. Symbolic Aggregate Approximation

**Symbolic Aggregate Approximation**, or **SAX**, converts a continuous time series into a discrete symbolic sequence.

### 14.1 Steps

1. Divide the series into equal-sized windows.
2. Compute the average value in each window using PAA.
3. Discretize the average values into symbols.

Example:

```text
Numeric averages: 0.2, 1.1, -0.5, 0.7
Symbols:          b,   d,   a,    c
```

### 14.2 Why SAX Matters

SAX allows continuous time series to be treated as discrete sequences. Once converted, sequence-mining tools can be applied.

Useful for:

- motif discovery
- approximate matching
- pruning distance computations
- clustering subsequences
- shape-based outlier detection

---

## 15. Time Series Similarity Measures

Similarity measures are crucial because many time series data mining tasks depend on them.

### 15.1 Euclidean Distance

Euclidean distance treats each timestamp as a dimension.

It works best when:

- series have the same length
- timestamps are aligned
- one-to-one correspondence exists between time points

It works poorly when:

- series are misaligned
- one series is stretched or compressed in time
- events occur at slightly different speeds

### 15.2 Dynamic Time Warping

**Dynamic Time Warping**, or **DTW**, allows one time series to be stretched or compressed along the time dimension to find the best match.

DTW is useful when two series have similar shapes but are not perfectly synchronized.

Example:

- two patients have similar ECG patterns but one heartbeat sequence is slightly faster
- two stocks follow similar trajectories but one reacts later
- two gestures have the same shape but different speed

### 15.3 Other Measures

The chapter also mentions:

- Edit Distance
- Longest Common Subsequence

These are more common for discrete sequences than continuous time series.

---

## 16. Time Series Forecasting

Forecasting predicts future values using past values.

Applications include:

- retail sales prediction
- economic forecasting
- weather forecasting
- stock price modeling
- sensor prediction
- healthcare monitoring

A major issue in forecasting is whether the series is **stationary** or **nonstationary**.

---

## 17. Stationary vs. Nonstationary Series

### 17.1 Stationary Time Series

A stationary time series has statistical properties that do not change over time.

Important properties:

- mean stays stable
- variance stays stable
- covariance/autocorrelation structure stays stable

A stationary series is easier to forecast because past statistical behavior is more likely to remain relevant.

### 17.2 Nonstationary Time Series

A nonstationary series has properties that change over time.

Examples:

- increasing price levels due to inflation
- growing sales trends
- changing variance over time
- seasonal patterns with changing magnitude

Most real-world time series are nonstationary.

### 17.3 Figure 14.3: Differencing and Log Scaling

The chart on page 9 shows how operations affect stationary and nonstationary behavior. The unscaled increasing series remains nonstationary after simple differencing. After logarithmic scaling, differencing produces a more stationary-looking series.

---

## 18. Differencing

**Differencing** replaces each value with the difference between it and the previous value.

Formula:

```text
y'_i = yi - y_(i-1)
```

Plain-English version:

> Instead of modeling the level of the series, model the change from one time point to the next.

### 18.1 Why Differencing Helps

If a series has an upward or downward trend, differencing can remove the trend and make the series more stationary.

### 18.2 Higher-Order Differencing

Second-order differencing models changes in changes.

Plain-English version:

> First-order differencing asks, “How much did the value change?” Second-order differencing asks, “How much did the change itself change?”

The chapter notes that differencing beyond second order is rare.

### 18.3 Seasonal Differencing

Seasonal differencing subtracts the value from one full seasonal period earlier.

Formula:

```text
y'_i = yi - y_(i-m)
```

where `m` is the seasonal period.

Example:

- daily data with weekly seasonality: `m = 7`
- monthly data with yearly seasonality: `m = 12`

---

## 19. Autoregressive Models

An **autoregressive model**, or **AR model**, predicts the current value using previous values of the same series.

### 19.1 Autocorrelation

Autocorrelation measures how strongly values at one time are related to values at a later lag.

Formula:

```text
Autocorrelation(L) = Covariance(yt, y_(t+L)) / Variance(yt)
```

Plain-English version:

> Autocorrelation asks: “If I know the value now, how much does that help me predict the value L steps later?”

### 19.2 AR(p) Model

Formula:

```text
yt = a1*y_(t-1) + a2*y_(t-2) + ... + ap*y_(t-p) + c + εt
```

Where:

- `p` = number of lagged values used
- `a1 ... ap` = learned coefficients
- `c` = constant/intercept
- `εt` = error/noise term

### 19.3 Choosing p

The chapter recommends using the autocorrelation plot to choose `p`. If autocorrelation becomes small after a certain lag, adding more lag terms may not help and may cause overfitting.

### 19.4 Figure 14.4: Autocorrelation Plots

The chart on page 12 compares autocorrelation for IBM stock data and a sine wave. The IBM autocorrelation fades with lag, while the sine wave shows periodic autocorrelation. This illustrates why autocorrelation plots can guide model selection and reveal periodic structure.

### 19.5 Model Quality: R-Squared

The chapter uses an R-squared-style measure to quantify how much variability is explained by the autoregressive model rather than noise.

High `R²` means the model explains much of the time series variation.

---

## 20. Moving Average Models

A **moving average model**, or **MA model**, predicts values using previous forecast errors or shocks.

### 20.1 MA(q) Model

Formula:

```text
yt = b1*ε_(t-1) + b2*ε_(t-2) + ... + bq*ε_(t-q) + c + εt
```

Where:

- `q` = number of previous error terms used
- `b1 ... bq` = learned coefficients
- `ε` = forecast error or shock

### 20.2 Difference Between AR and MA

| Model | Predicts using | Main idea |
|---|---|---|
| AR | previous values | the past level predicts the current level |
| MA | previous errors | previous shocks influence the current value |

AR models use the series values themselves. MA models use past deviations from expected values.

---

## 21. ARMA and ARIMA Models

### 21.1 ARMA Model

An **ARMA model** combines autoregressive terms and moving-average terms.

Formula:

```text
yt = AR terms + MA terms + c + εt
```

More explicitly:

```text
yt = Σ(ai*y_(t-i)) + Σ(bi*ε_(t-i)) + c + εt
```

An ARMA model is written as:

```text
ARMA(p, q)
```

where:

- `p` = number of autoregressive lag terms
- `q` = number of moving-average error terms

### 21.2 Choosing p and q

The goal is to use values that are large enough to fit the data but small enough to avoid overfitting.

### 21.3 ARIMA Model

An **ARIMA model** adds differencing to ARMA so it can handle nonstationary data.

ARIMA is written as:

```text
ARIMA(p, d, q)
```

where:

- `p` = autoregressive order
- `d` = differencing order
- `q` = moving-average order

Plain-English version:

> ARIMA first transforms the series by differencing, then models the transformed series using AR and MA components.

---

## 22. Multivariate Forecasting with Hidden Variables

Many real applications contain many correlated time series.

Examples:

- thousands of stock prices
- many sensors in a machine
- multiple medical readings from a patient
- weather variables across locations

Modeling each series separately ignores cross-series relationships. Modeling all series directly can be computationally difficult. The chapter describes a hidden-variable approach based on PCA.

### 22.1 Main Idea

Use **Principal Component Analysis**, or **PCA**, to transform many correlated series into a smaller number of uncorrelated hidden series.

Steps:

1. Build the covariance matrix across the different series.
2. Compute eigenvectors and eigenvalues.
3. Keep the top components with the largest eigenvalues.
4. Convert the original series into hidden-variable series.
5. Forecast each hidden series using univariate forecasting.
6. Transform the predictions back to the original variables.

### 22.2 Why This Helps

- Reduces dimensionality.
- Separates cross-series correlation from temporal autocorrelation.
- Allows simpler univariate forecasting on hidden components.
- Smooths out noise from individual series.

### 22.3 Figure 14.5: Hidden Variables for Correlated ETFs

The chart on page 15 shows four correlated precious-metal ETF price series and their derived hidden-variable series. The original ETF series move together, while the hidden variables are uncorrelated and capture the dominant shared movements.

---

## 23. Time Series Motifs

A **motif** is a frequently occurring pattern or shape in a time series.

Examples:

- repeated heartbeat shape
- recurring sensor failure signature
- repeating stock price movement
- repeated seasonal sales pattern

### 23.1 Motif Discovery Settings

The chapter describes several variations.

#### Single Series vs. Database of Series

In a single long series, a motif count is based on how often a pattern appears in that series.

In a database of many series, support may be counted by how many series contain the pattern at least once.

#### Contiguous vs. Noncontiguous Motifs

A contiguous motif occurs in one continuous window.

A noncontiguous motif allows gaps between elements.

#### Single-Granularity vs. Multigranularity Motifs

Some motifs occur over short windows. Others occur over longer windows.

Example:

- flash crash = short-duration motif
- recessionary market trend = long-duration motif

### 23.2 Figure 14.6: Repeated Motif

The chart on page 16 shows a repeated shape appearing three times in one time series. This illustrates the basic idea of motif count in a single series.

---

## 24. Distance-Based Motifs

Distance-based motif discovery defines a motif as a subsequence that approximately matches many windows of the time series.

### 24.1 Approximate Distance Match

A motif approximately matches a window if the distance between the motif and the window is at most a threshold.

Important inputs:

- window length `w`
- distance threshold `ε`
- distance function, often Euclidean distance

### 24.2 Motif Count

The count of a motif is the number of windows that approximately match it.

### 24.3 Avoiding Trivial Matches

Overlapping windows can create meaningless matches. For example, a window almost always matches itself and may match nearly identical overlapping windows nearby. These are called **trivial matches** and must be excluded.

### 24.4 Algorithmic Cost

The basic nested-loop approach compares every candidate window to every other window.

Complexity:

```text
O(n²) distance computations
```

This is expensive for long time series.

### 24.5 Speedups

The chapter discusses lower-bound pruning using compressed representations such as PAA or SAX. If the lower bound already exceeds the threshold, the exact distance computation can be skipped.

---

## 25. Transformation to Sequential Pattern Mining

Another way to discover motifs is to transform the time series into a symbolic sequence.

### 25.1 General Process

1. Discretize the time series values or transformed coefficients.
2. Create a sequence of symbols.
3. Apply sequential pattern mining algorithms.

### 25.2 Why This Is Flexible

The motifs discovered depend heavily on how the time series is discretized.

For example:

- discretizing raw values finds level-based motifs
- discretizing differences finds change-based motifs
- discretizing wavelet coefficients finds multiresolution trend motifs

This approach can also discover noncontiguous motifs because many sequential pattern mining algorithms allow gaps.

---

## 26. Periodic Patterns

Periodic patterns repeat at regular intervals.

DFT is used to detect periodic patterns because it decomposes a time series into sinusoidal components.

### 26.1 Main Idea

If one Fourier coefficient has much larger amplitude than the others, the series is strongly influenced by that periodic component.

Amplitude:

```text
sqrt(a_k² + b_k²)
```

Periodicity:

```text
n / k
```

where:

- `n` = length of the time series
- `k` = Fourier component index

### 26.2 Detecting Important Periods

The chapter describes flagging amplitudes that are several standard deviations above the mean amplitude. A large threshold such as 3 can isolate only the most dominant periodic components.

---

## 27. Time Series Clustering

Time series clustering has two major forms:

1. Online clustering of coevolving series.
2. Shape-based clustering of stored series.

---

## 28. Online Clustering of Coevolving Series

In online clustering, multiple series are observed at the same time and grouped based on how they move together in a recent time window.

Example:

- stocks that currently rise and fall together
- sensors that show correlated behavior
- patients with synchronized physiological patterns

### 28.1 Correlation Clustering

Series are clustered based on correlation or predictability, not necessarily direct similarity.

Important point:

> Two series can be considered similar even if they are negatively correlated, as long as one can predict the other.

### 28.2 Figure 14.8: Time Series Correlation Clustering

The chart on page 22 shows three series. Series A and B are highly related because they are negatively correlated and can predict one another. Series C is different because it has low predictability relative to the others.

### 28.3 Representative-Based Clustering

The chapter describes maintaining a set of representative series, similar to k-medoids.

Each nonrepresentative series is assigned to the representative that best predicts it.

The similarity between two streams is based on regression error.

### 28.4 Applications

- sensor selection
- financial-market trend analysis
- real-time monitoring
- reducing operational cost in sensor networks

---

## 29. Shape-Based Clustering

In shape-based clustering, stored time series are grouped by overall shape.

Example:

- ECG patterns from different patients
- stock-price histories with similar trajectories
- sensor traces with similar failure signatures

### 29.1 Main Challenge

Series may differ by:

- scale
- translation
- noise
- length
- time warping

### 29.2 Figure 14.10: Scaling, Translation, Noise, and Warping

The chart on page 24 shows hypothetical stock tickers that have similar shapes but differ in scaling, missing readings, noise, or time stretching. This demonstrates why Euclidean distance may fail and why DTW is often useful.

---

## 30. Clustering Algorithms for Time Series

### 30.1 k-Means

k-means can be adapted to time series when series have the same length and aligned timestamps.

Best used when:

- Euclidean distance is appropriate
- each time point corresponds across series
- centroids can be meaningfully computed

Limitations:

- does not naturally handle DTW
- assumes one-to-one timestamp alignment
- requires equal-length series for centroid computation

### 30.2 k-Medoids

k-medoids is more flexible because cluster centers are actual time series, not averages.

Advantages:

- can use arbitrary distance functions
- can handle DTW more naturally
- does not require mean series to be computed

### 30.3 Hierarchical Methods

Hierarchical clustering can use pairwise time series distances.

Advantage:

- works with many distance functions

Disadvantage:

- expensive because all pairwise distances may be needed
- DTW and other time series distances can be computationally costly

### 30.4 Graph-Based Methods

Graph-based clustering converts time series into a similarity graph.

Process:

1. Each time series becomes a node.
2. Edges connect each series to its nearest neighbors.
3. Edge weights reflect similarity.
4. Graph clustering/community detection is applied.
5. Node communities become time series clusters.

---

## 31. Time Series Outlier Detection

The chapter identifies two major types of time series outliers:

1. Point outliers.
2. Shape outliers.

---

## 32. Point Outliers

A **point outlier** is a sudden unexpected value at a timestamp.

Example:

- sudden spike in blood pressure
- abrupt stock-price drop
- sensor reading far from forecast

Point outlier detection is closely related to forecasting. A point is anomalous if it deviates significantly from its expected value.

### 32.1 General Process

1. Forecast the expected value at each timestamp.
2. Compute the deviation between forecasted and observed values.
3. Normalize deviations using mean and standard deviation.
4. Flag timestamps with large normalized deviations.

Deviation formula:

```text
Δr = Wr - Yr
```

Normalized deviation:

```text
δz = (δ - μ) / σ
```

A common absolute threshold is about 3, following the z-score interpretation.

### 32.2 Unified Alarm Level

For multivariate time series, deviations can be combined into one alarm score. The chapter suggests using the maximum normalized deviation across dimensions, though averages or squared aggregates may also be used.

---

## 33. Shape Outliers

A **shape outlier** is an unusual subsequence or window, not necessarily a single extreme point.

Example:

- irregular heartbeat pattern
- unusual multi-minute pressure pattern
- abnormal movement trajectory

### 33.1 Figure 14.11: Flash Crash vs. Market Decline

The charts on page 26 compare two S&P 500 behaviors. One shows the flash crash, which is unusual both as a point deviation and as a shape. The other shows a larger market decline over 2001, where the overall shape resembles patterns often seen in bear markets. This illustrates that point anomalies and shape anomalies are related but not identical.

### 33.2 Hotsax Approach

The chapter discusses **Hotsax** for shape-based outlier detection.

Main idea:

1. Extract all sliding windows of length `W`.
2. Compare each window to nonoverlapping windows.
3. Use k-nearest neighbor distance as the outlier score.
4. Report windows with the largest nearest-neighbor distances.

### 33.3 Why SAX Helps Hotsax

SAX maps subsequences into symbolic words. Subsequences with the same SAX word are treated as roughly clustered.

This helps pruning:

- candidates in rare SAX clusters are examined first
- likely outliers are found earlier
- early abandonment avoids unnecessary distance computations

---

## 34. Time Series Classification

The chapter describes two forms of time series classification:

1. **Point labels:** labels are assigned to individual timestamps.
2. **Whole-series labels:** labels are assigned to entire time series.

---

## 35. Supervised Event Detection

Supervised event detection classifies individual timestamps, often in real time.

Example:

- normal machine operation vs. malfunction
- normal patient readings vs. medical emergency
- normal sensor behavior vs. pipe rupture

### 35.1 Difference from Unsupervised Point Outlier Detection

Unsupervised point outlier detection flags large deviations. Supervised event detection uses labels to learn which deviations correspond to meaningful events.

This matters because not every deviation is important.

Example:

- real pipe rupture = important event
- pressure sensor failure = spurious abnormality

Both may produce unusual readings, but only one may be the event of interest.

### 35.2 Figure 14.12: Pipe Rupture vs. Sensor Failure

The four charts on page 30 compare temperature and pressure readings in two scenarios. In a pipe rupture, both pressure and temperature behavior change. In a pressure sensor failure, pressure drops to zero while temperature remains normal. This visual example shows why multivariate supervised event detection can distinguish meaningful events from spurious sensor problems.

### 35.3 Composite Alarm Level

The chapter describes learning coefficients that combine deviations across streams into a composite alarm score.

Formula:

```text
Zn = α1*z1n + α2*z2n + ... + αd*zdn
```

Where:

- `zjn` = deviation level for stream `j` at time `n`
- `αj` = learned weight for stream `j`
- `Zn` = composite alarm level

The weights are learned so true events score higher than normal periods.

---

## 36. Whole-Series Classification

In whole-series classification, the label belongs to the entire time series.

Examples:

- classify ECG trace as normal or abnormal
- classify a stock history as recessionary or non-recessionary
- classify a machine sensor trace by failure type

The chapter discusses three methods:

1. Wavelet-based rules.
2. Nearest-neighbor classification.
3. Graph-based classification.

---

## 37. Wavelet-Based Rules

Wavelet-based classification uses wavelet coefficients as features.

### 37.1 Why Wavelets Are Useful

Important class signals may occur at different time granularities.

Examples:

- flash crash: short time scale
- recessionary trend: long time scale

Wavelets capture both short-term and long-term patterns.

### 37.2 Process

1. Transform each series into wavelet coefficients.
2. Discretize the coefficients into categorical values.
3. Learn classification rules.
4. Use fired rules to classify new series.

The rule antecedents represent shape signatures associated with classes.

### 37.3 Fourier Alternative

If class labels depend more on periodicity than local trends, Fourier coefficients may be used instead of wavelet coefficients.

---

## 38. Nearest Neighbor Classifier

A nearest-neighbor classifier can be used for time series if an appropriate distance function is available.

Process:

1. For a test time series, compute distances to training series.
2. Find the `k` nearest neighbors.
3. Predict the dominant label among those neighbors.

The value of `k` can be selected using leave-one-out cross-validation.

This method is simple and often strong, especially when using a domain-appropriate distance such as DTW.

---

## 39. Graph-Based Classification

Graph-based classification uses both labeled and unlabeled time series.

Process:

1. Build a similarity graph from training and test instances.
2. Labeled nodes correspond to training instances.
3. Unlabeled nodes correspond to test instances.
4. Connect nodes to their nearest neighbors.
5. Propagate or infer labels through the graph.

This is a **transductive semisupervised** approach because it uses the test instances while building the classification structure.

---

## 40. Key Formulas and Plain-English Meaning

| Concept | Formula | Plain-English meaning |
|---|---|---|
| Linear interpolation | `y = yi + ((t-ti)/(tj-ti))*(yj-yi)` | Estimate a missing value by moving proportionally between two known values. |
| Exponential smoothing | `y'_i = αyi + (1-α)y'_(i-1)` | Blend the current value with the previous smoothed value. |
| Range normalization | `y'_i = (yi-min)/(max-min)` | Rescale values to the 0–1 range. |
| Standardization | `zi = (yi-μ)/σ` | Express values as standard deviations from the mean. |
| Differencing | `y'_i = yi-y_(i-1)` | Model changes instead of raw levels. |
| Seasonal differencing | `y'_i = yi-y_(i-m)` | Compare a value to the value one seasonal period earlier. |
| Autocorrelation | `Cov(yt,y_(t+L))/Var(yt)` | Measure how much a value predicts another value L steps later. |
| AR(p) | `yt = Σ ai*y_(t-i) + c + εt` | Predict from previous values. |
| MA(q) | `yt = Σ bi*ε_(t-i) + c + εt` | Predict from previous errors or shocks. |
| ARIMA(p,d,q) | ARMA after differencing | Difference the series, then model lag values and lag errors. |
| Point outlier deviation | `Δr = Wr - Yr` | Difference between expected and observed value. |
| Composite alarm | `Zn = Σ αi*zi_n` | Weighted combination of deviations across multiple streams. |

---

## 41. High-Yield Concept Map

```text
Time Series Data
│
├── Preparation
│   ├── missing values → interpolation
│   ├── noise → binning, moving average, exponential smoothing
│   ├── scale differences → normalization / standardization
│   └── reduction → DWT, DFT, SAX
│
├── Similarity
│   ├── Euclidean distance → aligned equal-length series
│   └── DTW → warped or misaligned series
│
├── Forecasting
│   ├── stationarity matters
│   ├── differencing can remove trends
│   ├── AR → previous values
│   ├── MA → previous shocks
│   ├── ARMA → AR + MA
│   └── ARIMA → ARMA + differencing
│
├── Motifs
│   ├── distance-based motifs
│   ├── sequential pattern mining after discretization
│   └── periodic motifs using DFT
│
├── Clustering
│   ├── online coevolving series → correlation / predictability
│   └── stored series → shape similarity
│
├── Outlier Detection
│   ├── point outliers → unexpected timestamp value
│   └── shape outliers → unusual subsequence/window
│
└── Classification
    ├── supervised event detection → timestamp labels
    └── whole-series classification → series labels
```

---

## 42. Common Exam-Style Questions and Answers

### Q1. Why is time series data considered contextual data?

Because the behavioral value cannot be interpreted without the timestamp or time index. The same value can have different meaning depending on when it occurs and what happened immediately before it.

### Q2. When should Euclidean distance be used for time series?

Use Euclidean distance when the two series are equal length, synchronized, and have one-to-one timestamp correspondence.

### Q3. When should DTW be used?

Use DTW when two time series have similar shapes but differ in timing, speed, or local alignment.

### Q4. Why is stationarity important for forecasting?

Forecasting models rely on past statistical behavior. If the mean, variance, or autocorrelation structure changes over time, the past may not predict the future well.

### Q5. What is the difference between AR and MA models?

An AR model predicts using previous observed values. An MA model predicts using previous forecast errors or shocks.

### Q6. What makes ARIMA useful?

ARIMA handles nonstationary data by differencing the series before applying ARMA-style modeling.

### Q7. What is a motif?

A motif is a repeated shape or pattern in a time series.

### Q8. What is the difference between point outliers and shape outliers?

A point outlier is one unexpected timestamp value. A shape outlier is an unusual window or subsequence whose pattern is abnormal, even if no single point is extreme.

### Q9. Why is SAX useful?

SAX converts continuous time series into symbolic sequences, enabling sequence mining, approximate matching, clustering, and pruning.

### Q10. Why are wavelets useful in classification?

Wavelets capture patterns at multiple time scales, so they can detect both short-term and long-term class-specific shape signatures.

---

## 43. Python Implementation Notes

Below are practical Python equivalents for several chapter concepts.

### 43.1 Binning / PAA

```python
import numpy as np

def paa(series, bin_size):
    series = np.asarray(series)
    usable_length = len(series) - (len(series) % bin_size)
    trimmed = series[:usable_length]
    return trimmed.reshape(-1, bin_size).mean(axis=1)
```

### 43.2 Moving Average

```python
import pandas as pd

def moving_average(series, window):
    return pd.Series(series).rolling(window=window).mean()
```

### 43.3 Exponential Smoothing

```python
def exponential_smoothing(series, alpha):
    smoothed = [series[0]]
    for value in series[1:]:
        new_value = alpha * value + (1 - alpha) * smoothed[-1]
        smoothed.append(new_value)
    return smoothed
```

### 43.4 Differencing

```python
import numpy as np

def difference(series, lag=1):
    series = np.asarray(series)
    return series[lag:] - series[:-lag]
```

### 43.5 Standardization

```python
import numpy as np

def standardize(series):
    series = np.asarray(series)
    return (series - series.mean()) / series.std()
```

### 43.6 Simple Autoregressive Feature Matrix

```python
import numpy as np

def make_lag_matrix(series, p):
    series = np.asarray(series)
    X = []
    y = []
    for t in range(p, len(series)):
        X.append(series[t-p:t][::-1])
        y.append(series[t])
    return np.asarray(X), np.asarray(y)
```

### 43.7 Point Outlier Scores from Forecast Errors

```python
import numpy as np

def zscore_deviations(observed, forecast):
    deviations = np.asarray(forecast) - np.asarray(observed)
    return (deviations - deviations.mean()) / deviations.std()
```

---

## 44. Most Important Takeaways

1. Time series mining is different from ordinary multidimensional data mining because time provides context.
2. Preprocessing is critical: missing values, noise, scale differences, and dimensionality all affect the analysis.
3. Similarity measures are central. Euclidean distance works for aligned series; DTW works for warped or misaligned series.
4. Forecasting often requires stationarity. Differencing and log transforms can help convert nonstationary data into more stationary forms.
5. AR models use past values; MA models use past errors; ARMA combines both; ARIMA adds differencing.
6. DWT is best for local patterns; DFT is best for periodic patterns.
7. SAX converts continuous series into discrete sequences, opening the door to sequence-mining methods.
8. Motifs are repeated shapes; outliers are unexpected points or unusual shapes.
9. Online clustering focuses on coevolution and predictability; retrospective clustering focuses on shape similarity.
10. Classification can target individual events or entire time series.

---

## 45. Quick Study Checklist

Use this list to test whether you understand the chapter.

- [ ] I can explain the difference between contextual and behavioral attributes.
- [ ] I can distinguish univariate and multivariate time series.
- [ ] I can explain linear interpolation.
- [ ] I can compare binning, moving averages, and exponential smoothing.
- [ ] I can explain why normalization matters.
- [ ] I can distinguish DWT, DFT, and SAX.
- [ ] I know when to use Euclidean distance vs. DTW.
- [ ] I can define stationarity and nonstationarity.
- [ ] I can explain differencing and seasonal differencing.
- [ ] I can distinguish AR, MA, ARMA, and ARIMA.
- [ ] I can explain hidden-variable multivariate forecasting with PCA.
- [ ] I can define motif discovery.
- [ ] I can distinguish point outliers and shape outliers.
- [ ] I can explain supervised event detection.
- [ ] I can describe whole-series classification methods.

