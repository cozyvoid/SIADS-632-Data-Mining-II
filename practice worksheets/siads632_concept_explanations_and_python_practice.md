# SIADS 632 Formula + Concept Explanations + Python Practice Workbook

This workbook is designed to make the most common Data Mining II topics feel less overwhelming. Each topic is explained four ways:

1. **5-year-old version** - intuition only.
2. **10-year-old version** - simple but closer to the real idea.
3. **Undergraduate version** - course-ready explanation.
4. **Graduate student version** - more formal explanation.

Each topic also includes:

- an **analogy legend** so the analogy does not become vague;
- a **plain-English formula translation**;
- a **by-hand mini problem**;
- a **Python implementation pattern** similar to assignment-style coding.

Use this workbook in two passes. First, read only the 5-year-old and 10-year-old versions. Second, return to the undergraduate and graduate versions after the formula feels less intimidating.

## 1. N-gram Language Models: Counting What Comes Next
**Core idea:** N-gram models estimate the probability of the next token using the previous n-1 tokens as context. A unigram uses no previous word, a bigram uses one previous word, and a trigram uses two previous words.
### Version a 5-year-old can understand
Imagine you always hear your friend say, "I love ___". If they usually say "dogs" after that, you guess "dogs" next. You are using the words before the blank to guess the next word.
**Analogy legend:**
- friend sentence = the training text / corpus
- words before the blank = context
- word in the blank = next token
- how many times it happened = count

### Version a 10-year-old can understand
An n-gram model is like autocomplete. It looks at how often word groups appeared before. If "I love" was followed by "pizza" 8 times and "dogs" 2 times, then "pizza" is the more likely next word.
### Undergraduate version
An n-gram language model approximates P(w_i | w_1,...,w_{i-1}) with a shorter context P(w_i | w_{i-n+1},...,w_{i-1}). This makes estimation possible from finite data because we only count local token patterns.
### Graduate student version
N-gram models impose a Markov assumption: the conditional distribution of the next token depends only on a bounded history. This trades representational capacity for estimability and sparse-count tractability. Model quality depends heavily on tokenization, boundary tokens, smoothing, and domain match.
### Formula in plain English
**Formula-style statement:** `P(next word | previous word) = count(previous word, next word) / count(previous word)`

**Plain-English statement:** Probability of the next word = how many times the exact pair happened divided by how many times the context word happened.
### Practice problem
Corpus: "I like cats", "I like dogs", "I like cats". Use <s> and </s>. Find P(cats | like) and P(dogs | like).
**By-hand solution:**
- Count "like cats" = 2.
- Count "like dogs" = 1.
- Count "like" as context = 3.
- P(cats | like) = 2/3.
- P(dogs | like) = 1/3.

**Python implementation / solution:**
```python
from collections import Counter

sentences = ["I like cats", "I like dogs", "I like cats"]

# tokenize each sentence and add sentence boundary markers
tokenized = [["<s>"] + s.lower().split() + ["</s>"] for s in sentences]

bigram_counts = Counter()
context_counts = Counter()

for sent in tokenized:
    for i in range(len(sent) - 1):
        context = sent[i]
        next_word = sent[i + 1]
        bigram_counts[(context, next_word)] += 1
        context_counts[context] += 1

def bigram_probability(context, next_word):
    return bigram_counts[(context, next_word)] / context_counts[context]

print(bigram_probability("like", "cats"))  # 0.666...
print(bigram_probability("like", "dogs"))  # 0.333...
```

## 2. Add-k Smoothing: Giving Unseen Events a Small Chance
**Core idea:** Smoothing prevents zero probabilities for n-grams not observed in the training data. Add-k smoothing adds a small amount k to every possible next-token count.
### Version a 5-year-old can understand
Imagine you count favorite snacks, but nobody picked apples today. You still do not want to say apples are impossible forever. So you give every snack one tiny pretend vote.
**Analogy legend:**
- snack = possible word
- pretend vote = k
- snack votes = word count
- all snack choices = vocabulary

### Version a 10-year-old can understand
If a word pair never appeared, normal counting gives it probability 0. Add-k smoothing adds a small number to every count so unseen pairs still get a small probability.
### Undergraduate version
Add-k smoothing changes the MLE numerator from count(context, word) to count(context, word) + k and the denominator from count(context) to count(context) + kV, where V is the vocabulary size.
### Graduate student version
Add-k smoothing is equivalent to a symmetric Dirichlet prior over multinomial next-token distributions. It is simple but often poorly calibrated for language because it assigns too much mass to unseen events when V is large.
### Formula in plain English
**Formula-style statement:** `P_k(w | c) = [count(c,w) + k] / [count(c) + k * V]`

**Plain-English statement:** Smoothed probability = adjusted pair count divided by adjusted total context count. The denominator grows because every vocabulary word gets k extra count.
### Practice problem
For context "like", count(like,cats)=2, count(like,dogs)=1, count(like,birds)=0, total count(like)=3, V=3, k=1. Compute P(birds | like).
**By-hand solution:**
- Numerator = 0 + 1 = 1.
- Denominator = 3 + 1*3 = 6.
- P(birds | like) = 1/6.

**Python implementation / solution:**
```python
def add_k_probability(context, word, bigram_counts, context_counts, vocabulary, k=1):
    numerator = bigram_counts[(context, word)] + k
    denominator = context_counts[context] + k * len(vocabulary)
    return numerator / denominator

vocabulary = {"cats", "dogs", "birds"}
print(add_k_probability("like", "birds", bigram_counts, context_counts, vocabulary, k=1))
```

## 3. Interpolation: Mixing Multiple Models
**Core idea:** Interpolation combines probabilities from different n-gram models, usually unigram, bigram, and trigram models, using weights that sum to 1.
### Version a 5-year-old can understand
You ask three friends what word comes next. One friend looks at no words, one looks at one word, and one looks at two words. You listen to all of them, but maybe trust one friend more.
**Analogy legend:**
- three friends = unigram, bigram, trigram models
- trust = lambda weight
- final answer = interpolated probability

### Version a 10-year-old can understand
Sometimes the longest context is too rare. Interpolation mixes a specific guess with more general guesses. If the trigram is unreliable, the bigram and unigram can still help.
### Undergraduate version
A linear interpolation model computes P(w|context) as lambda_1*P_unigram(w) + lambda_2*P_bigram(w|w_{i-1}) + lambda_3*P_trigram(w|w_{i-2},w_{i-1}), with lambda values summing to 1.
### Graduate student version
Interpolation is a smoothing strategy that reduces variance from sparse high-order contexts by backing off probabilistic mass to lower-order estimates. Lambdas can be fixed, tuned on held-out data, or context dependent.
### Formula in plain English
**Formula-style statement:** `P_mix = lambda1 * P1 + lambda2 * P2 + lambda3 * P3, where lambda1 + lambda2 + lambda3 = 1`

**Plain-English statement:** Final probability = weighted average of several probability guesses. Bigger weight means that model matters more.
### Practice problem
Given P_unigram=0.10, P_bigram=0.40, P_trigram=0.70 and lambdas 0.2, 0.3, 0.5, compute P_mix.
**By-hand solution:**
- P_mix = 0.2*0.10 + 0.3*0.40 + 0.5*0.70.
- P_mix = 0.02 + 0.12 + 0.35 = 0.49.

**Python implementation / solution:**
```python
def interpolated_probability(p_uni, p_bi, p_tri, lambdas=(0.2, 0.3, 0.5)):
    l1, l2, l3 = lambdas
    return l1 * p_uni + l2 * p_bi + l3 * p_tri

print(interpolated_probability(0.10, 0.40, 0.70))  # 0.49
```

## 4. Perplexity: How Surprised the Model Is
**Core idea:** Perplexity measures how surprised a language model is by a sequence. Lower perplexity means the model assigned higher probability to the observed text.
### Version a 5-year-old can understand
A guessing robot reads a story. If it keeps guessing the next word correctly, it is not surprised. If it keeps guessing badly, it is very surprised.
**Analogy legend:**
- guessing robot = language model
- story = test sequence
- surprise = perplexity
- good guesses = high probabilities

### Version a 10-year-old can understand
Perplexity is like the average number of choices the model feels it has at each step. Lower is better. A perplexity of 2 feels like choosing between 2 likely words; 100 feels much more confused.
### Undergraduate version
For tokens w_1,...,w_N, perplexity is often computed as PP = (1 / product_i P(w_i | context_i))^(1/N), or equivalently exp(-mean log probability).
### Graduate student version
Perplexity is the exponentiated cross-entropy of the empirical test distribution under the model. It is sensitive to tokenization and test-set comparability, so it should only be compared under identical preprocessing and evaluation conditions.
### Formula in plain English
**Formula-style statement:** `PP = exp(- average log probability)`

**Plain-English statement:** Perplexity = turn the model’s average negative log-probability back into an easier-to-read scale. Smaller means less surprise.
### Practice problem
A model assigns probabilities 0.5, 0.25, and 0.25 to a 3-token test sequence. Compute perplexity.
**By-hand solution:**
- Use product form: PP = (1 / (0.5*0.25*0.25))^(1/3).
- Product = 0.03125.
- 1/product = 32.
- Cube root of 32 is about 3.17.

**Python implementation / solution:**
```python
import math

def perplexity(probabilities):
    log_probs = [math.log(p) for p in probabilities]
    avg_log_prob = sum(log_probs) / len(log_probs)
    return math.exp(-avg_log_prob)

print(perplexity([0.5, 0.25, 0.25]))  # about 3.17
```

## 5. HMM Forward Algorithm: Probability of Observations
**Core idea:** A Hidden Markov Model has hidden states that produce visible observations. The forward algorithm computes the probability of an observed sequence by summing over all possible hidden-state paths.
### Version a 5-year-old can understand
You see muddy footprints, but you cannot see the weather. The weather could be rainy or sunny. You guess how likely the footprints are by thinking about all possible weather stories.
**Analogy legend:**
- weather = hidden state
- muddy footprints = observation
- weather story = state path
- chance of footprints = observation sequence probability

### Version a 10-year-old can understand
The forward algorithm keeps a running table. At each time step, it asks: what is the total probability of being in each hidden state after seeing observations so far?
### Undergraduate version
The recurrence is alpha_t(j) = emission_j(o_t) * sum_i alpha_{t-1}(i) * transition_{i,j}. Initialization uses initial probabilities times the first emission probability.
### Graduate student version
Forward inference marginalizes latent state trajectories using dynamic programming. It reduces exponential enumeration over hidden paths to O(TS^2), where T is sequence length and S is number of states.
### Formula in plain English
**Formula-style statement:** `alpha_t(state j) = emit(j, observation_t) * sum over previous states i [alpha_{t-1}(i) * transition(i -> j)]`

**Plain-English statement:** New score for a state = how well that state explains today’s observation times the total probability of arriving there from yesterday’s possible states.
### Practice problem
States Rainy/Sunny. Initial: R=.6, S=.4. Transition: R->R=.7, R->S=.3, S->R=.4, S->S=.6. Emission for obs walk: R=.1, S=.6. Compute first forward values for observation walk.
**By-hand solution:**
- alpha_1(R) = initial(R)*emit_R(walk) = .6*.1 = .06.
- alpha_1(S) = initial(S)*emit_S(walk) = .4*.6 = .24.
- Total probability after first observation = .06 + .24 = .30.

**Python implementation / solution:**
```python
states = ["Rainy", "Sunny"]
start = {"Rainy": 0.6, "Sunny": 0.4}
emit = {
    "Rainy": {"walk": 0.1},
    "Sunny": {"walk": 0.6}
}

alpha = {}
first_obs = "walk"
for state in states:
    alpha[state] = start[state] * emit[state][first_obs]

print(alpha)              # {'Rainy': 0.06, 'Sunny': 0.24}
print(sum(alpha.values())) # 0.30
```

## 6. Viterbi Algorithm: Most Likely Hidden Path
**Core idea:** Viterbi finds the single most likely sequence of hidden states that could have produced the observations.
### Version a 5-year-old can understand
You are a detective. You do not just ask, "How likely are these clues?" You ask, "What is the most likely story that made these clues happen?"
**Analogy legend:**
- detective story = hidden-state path
- clues = observations
- best story = Viterbi path
- story score = path probability

### Version a 10-year-old can understand
Viterbi is like the forward algorithm, but instead of adding all possible paths, it keeps only the best path to each state.
### Undergraduate version
The recurrence is delta_t(j) = emission_j(o_t) * max_i [delta_{t-1}(i) * transition_{i,j}], while storing backpointers to recover the best path.
### Graduate student version
Viterbi computes the MAP state sequence under an HMM by replacing the sum-product semiring used in forward inference with the max-product semiring. Backpointers preserve argmax decisions.
### Formula in plain English
**Formula-style statement:** `delta_t(j) = emit(j, observation_t) * max over previous states i [delta_{t-1}(i) * transition(i -> j)]`

**Plain-English statement:** Best score for a state today = how well the state explains today’s observation times the best way to get there from yesterday.
### Practice problem
Using the same HMM, suppose alpha-like previous best scores are R=.06, S=.24 and next observation is walk. Compute best score for Sunny at t=2.
**By-hand solution:**
- Path from R to S: .06*.3 = .018.
- Path from S to S: .24*.6 = .144.
- Best arrival to S is .144.
- Multiply by Sunny emission for walk: .144*.6 = .0864.

**Python implementation / solution:**
```python
prev_scores = {"Rainy": 0.06, "Sunny": 0.24}
transition = {
    "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
    "Sunny": {"Rainy": 0.4, "Sunny": 0.6}
}

current_state = "Sunny"
obs = "walk"

best_prev_state, best_arrival = max(
    ((prev_state, prev_scores[prev_state] * transition[prev_state][current_state])
     for prev_state in states),
    key=lambda x: x[1]
)

score = best_arrival * emit[current_state][obs]
print(best_prev_state, score)  # Sunny, 0.0864
```

## 7. Similarity Metrics: Comparing Objects
**Core idea:** Similarity and distance metrics compare items such as documents, vectors, or sets. Common metrics include Jaccard similarity, cosine similarity, and Euclidean distance.
### Version a 5-year-old can understand
You compare two toy boxes. You can ask: how many toys do they share, do they point in the same direction, or how far apart are they on the floor?
**Analogy legend:**
- toy boxes = sets or vectors
- shared toys = intersection
- all toys = union
- direction = vector orientation
- floor distance = Euclidean distance

### Version a 10-year-old can understand
Jaccard is for overlap between sets. Cosine is for direction of numeric vectors. Euclidean distance is straight-line distance between points.
### Undergraduate version
Jaccard(A,B)=|A∩B|/|A∪B|. Cosine(x,y)=dot(x,y)/(||x|| ||y||). Euclidean distance is sqrt(sum_i (x_i-y_i)^2). Choice depends on representation and task.
### Graduate student version
Metric selection encodes assumptions about invariances: Jaccard ignores counts and focuses on set overlap, cosine normalizes magnitude and focuses on angle, while Euclidean preserves magnitude and coordinate-scale effects.
### Formula in plain English
**Formula-style statement:** `Jaccard = overlap / total unique items; Cosine = dot product / product of vector lengths; Euclidean = square root of summed squared differences`

**Plain-English statement:** Jaccard asks "what fraction overlaps?" Cosine asks "do these point the same way?" Euclidean asks "how far apart are these numbers?"
### Practice problem
A={cat,dog,bird}, B={dog,bird,fish}. Compute Jaccard similarity.
**By-hand solution:**
- Intersection = {dog,bird}, size 2.
- Union = {cat,dog,bird,fish}, size 4.
- Jaccard = 2/4 = 0.5.

**Python implementation / solution:**
```python
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)

A = {"cat", "dog", "bird"}
B = {"dog", "bird", "fish"}
print(jaccard(A, B))  # 0.5
```

## 8. Edit Distance: Minimum Changes Between Strings
**Core idea:** Edit distance measures how many insertions, deletions, or substitutions are needed to transform one string into another.
### Version a 5-year-old can understand
You have the word "cat" and want "cut". You only need to change the "a" to "u", so the distance is 1.
**Analogy legend:**
- word = string or sequence
- change a letter = substitution
- add a letter = insertion
- remove a letter = deletion
- fewest changes = edit distance

### Version a 10-year-old can understand
Edit distance is the smallest number of edits needed. It is useful for spell check, DNA comparison, and matching similar text.
### Undergraduate version
Levenshtein distance is computed with dynamic programming over a matrix where each cell stores the minimum cost to align prefixes of the two strings.
### Graduate student version
Edit distance solves a shortest-path problem on an alignment lattice. The recurrence minimizes over deletion, insertion, and substitution costs, and can be generalized with weighted operations.
### Formula in plain English
**Formula-style statement:** `D[i,j] = min(D[i-1,j]+delete, D[i,j-1]+insert, D[i-1,j-1]+substitute_cost)`

**Plain-English statement:** Cost at a cell = cheapest of three choices: delete from first word, insert into first word, or match/change the current letters.
### Practice problem
Find the edit distance between cat and cut with substitution cost 1.
**By-hand solution:**
- c matches c: cost 0.
- a changes to u: cost 1.
- t matches t: cost 0.
- Total minimum edits = 1.

**Python implementation / solution:**
```python
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            substitution_cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost
            )
    return dp[m][n]

print(edit_distance("cat", "cut"))  # 1
```

## 9. Dynamic Time Warping: Matching Time Series That Move at Different Speeds
**Core idea:** Dynamic Time Warping, or DTW, measures similarity between two sequences that may be stretched or shifted in time.
### Version a 5-year-old can understand
Two people sing the same song, but one sings some parts slower. DTW helps match the same parts of the song even if the timing is different.
**Analogy legend:**
- two singers = two time series
- same notes = similar values
- singing slower = time stretching
- matching notes = warping path

### Version a 10-year-old can understand
DTW lines up two sequences by allowing one point in a sequence to match one or more points in another sequence. This helps when patterns are similar but not perfectly aligned.
### Undergraduate version
DTW builds a cost matrix where local cost is usually absolute or squared difference. The accumulated cost at each cell is local cost plus the minimum of left, up, or diagonal predecessor.
### Graduate student version
DTW computes an optimal monotonic alignment path under boundary and continuity constraints. It is useful for elastic sequence matching but can over-warp without window constraints or normalization.
### Formula in plain English
**Formula-style statement:** `DTW[i,j] = cost(x_i, y_j) + min(DTW[i-1,j], DTW[i,j-1], DTW[i-1,j-1])`

**Plain-English statement:** Best cost to align up to two points = current mismatch cost plus the cheapest previous alignment path.
### Practice problem
For x=[1,2] and y=[1,2], using absolute difference, what is the DTW distance?
**By-hand solution:**
- Match 1 with 1: cost 0.
- Match 2 with 2: cost 0.
- Total alignment cost = 0.

**Python implementation / solution:**
```python
def dtw_distance(x, y):
    m, n = len(x), len(y)
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

print(dtw_distance([1, 2], [1, 2]))  # 0
```

## 10. Time-Series Components and Moving Averages
**Core idea:** A time series can contain trend, seasonality, cycles, and noise. A moving average smooths short-term noise to make the underlying pattern easier to see.
### Version a 5-year-old can understand
Imagine daily weather. One weird hot day does not mean the whole season changed. A moving average looks at a few days together so the line is less jumpy.
**Analogy legend:**
- daily weather = time series values
- one weird day = noise/outlier
- season getting warmer = trend
- repeating summer/winter pattern = seasonality
- few days together = window

### Version a 10-year-old can understand
A moving average replaces each value with the average of nearby values. This smooths bumps and helps show the bigger direction.
### Undergraduate version
For a window size k, a simple moving average at time t is the mean of the previous k values, often including the current value. It is a simple low-pass filter.
### Graduate student version
Moving averages reduce high-frequency variance at the cost of lag and boundary effects. The choice of window size encodes a smoothing scale and can obscure abrupt regime changes.
### Formula in plain English
**Formula-style statement:** `MA_t = average of the last k observations`

**Plain-English statement:** Moving average today = add the most recent k values and divide by k.
### Practice problem
For values [2,4,6,8] and window k=2, compute moving averages after enough values exist.
**By-hand solution:**
- Average of 2 and 4 = 3.
- Average of 4 and 6 = 5.
- Average of 6 and 8 = 7.
- Result = [3,5,7].

**Python implementation / solution:**
```python
def moving_average(values, k):
    return [sum(values[i-k:i]) / k for i in range(k, len(values) + 1)]

print(moving_average([2, 4, 6, 8], 2))  # [3.0, 5.0, 7.0]
```

## 11. AR(1) Forecasting: Predicting from the Previous Value
**Core idea:** An autoregressive model predicts the current value using previous values. AR(1) uses only the immediately previous value.
### Version a 5-year-old can understand
If today was very busy, tomorrow might also be busy. You use today to guess tomorrow.
**Analogy legend:**
- today = previous time value y_{t-1}
- tomorrow = predicted value y_t
- usual starting amount = intercept
- how much today matters = phi coefficient

### Version a 10-year-old can understand
AR(1) says the next value is a baseline plus some fraction of the last value. If the fraction is high, the series remembers the past strongly.
### Undergraduate version
An AR(1) model is y_t = c + phi*y_{t-1} + error_t. Forecasting often uses yhat_t = c + phi*y_{t-1}.
### Graduate student version
AR(1) is a linear stochastic process with first-order dependence. Stationarity typically requires |phi| < 1, and phi controls persistence and mean reversion.
### Formula in plain English
**Formula-style statement:** `forecast next value = intercept + phi * previous value`

**Plain-English statement:** Prediction = baseline amount plus the previous value multiplied by how strongly the past carries forward.
### Practice problem
If intercept c=2, phi=0.5, and previous value is 10, forecast the next value.
**By-hand solution:**
- Forecast = 2 + 0.5*10.
- Forecast = 2 + 5 = 7.

**Python implementation / solution:**
```python
def ar1_forecast(previous_value, intercept, phi):
    return intercept + phi * previous_value

print(ar1_forecast(previous_value=10, intercept=2, phi=0.5))  # 7.0
```

## 12. Reservoir Sampling: Keeping a Fair Sample from a Stream
**Core idea:** Reservoir sampling keeps a random sample of fixed size from a stream when the total number of items is unknown or too large to store.
### Version a 5-year-old can understand
Toys keep coming down a slide forever, but your basket only holds 3 toys. Reservoir sampling gives every toy a fair chance to end up in the basket.
**Analogy legend:**
- toys = stream items
- basket = reservoir sample
- basket size = k
- toy number = stream index n

### Version a 10-year-old can understand
When you cannot store everything, reservoir sampling fills the sample first. After that, each new item has a chance to replace an old item. The chance gets smaller as more items arrive.
### Undergraduate version
For reservoir size k, keep the first k items. For the nth item where n > k, include it with probability k/n. If included, replace one existing reservoir item uniformly at random.
### Graduate student version
Reservoir sampling provides an unbiased fixed-size sample over a streaming population using O(k) memory and one pass, assuming random replacement and correct inclusion probabilities.
### Formula in plain English
**Formula-style statement:** `For item n: include with probability k/n; if included, replace a random reservoir position`

**Plain-English statement:** The later an item arrives, the harder it is to enter the sample, but the replacement rule keeps all items equally likely overall.
### Practice problem
Reservoir size k=2. The 5th item arrives. What is the probability it enters the reservoir?
**By-hand solution:**
- Use k/n.
- k=2 and n=5.
- Probability = 2/5 = 0.4.

**Python implementation / solution:**
```python
import random

def reservoir_sample(stream, k, seed=42):
    random.seed(seed)
    reservoir = []

    for n, item in enumerate(stream, start=1):
        if n <= k:
            reservoir.append(item)
        else:
            # choose a random integer from 1 to n
            j = random.randint(1, n)
            if j <= k:
                reservoir[j - 1] = item
    return reservoir

print(reservoir_sample(range(1, 11), k=2))
```

## 13. Bloom Filters: Memory-Efficient Membership Checks
**Core idea:** A Bloom filter tests whether an item might be in a set using hashes and a bit array. It can have false positives but never false negatives, assuming no deletions and correct implementation.
### Version a 5-year-old can understand
You have a sticker board for books you have seen. If a book’s sticker spots are empty, you definitely have not seen it. If the spots are filled, maybe you have seen it.
**Analogy legend:**
- book = item
- sticker board = bit array
- sticker spots = hash positions
- maybe seen = possible member
- definitely not seen = not a member

### Version a 10-year-old can understand
A Bloom filter saves space. To add an item, several hash functions turn on several bits. To check an item, look at those bits. If any are off, the item is definitely not present. If all are on, it might be present.
### Undergraduate version
Insertion hashes an item to k positions in an m-length bit vector and sets them to 1. Querying checks whether all k positions are 1. Collisions create false positives.
### Graduate student version
Bloom filters implement approximate set membership with tunable false-positive probability. Memory m, number of hashes k, and inserted items n determine the false-positive tradeoff.
### Formula in plain English
**Formula-style statement:** `Query rule: if any hashed bit is 0 -> definitely not present; if all hashed bits are 1 -> maybe present`

**Plain-English statement:** One empty required spot proves the item was never added. All full spots are only suggestive because other items may have filled them.
### Practice problem
A query item hashes to positions [2,5,7]. The bit array has bit 5 = 0. Is the item present?
**By-hand solution:**
- Bloom filter membership requires all hashed positions to be 1.
- Since bit 5 is 0, the item is definitely not present.

**Python implementation / solution:**
```python
class SimpleBloomFilter:
    def __init__(self, size=10):
        self.size = size
        self.bits = [0] * size

    def _hashes(self, item):
        # simple teaching hashes, not production-quality hashes
        return [hash(str(item)) % self.size, hash(str(item) + "salt") % self.size]

    def add(self, item):
        for pos in self._hashes(item):
            self.bits[pos] = 1

    def query(self, item):
        return all(self.bits[pos] == 1 for pos in self._hashes(item))

bf = SimpleBloomFilter(size=20)
bf.add("cat")
print(bf.query("cat"))   # probably True
print(bf.query("zebra")) # False or possibly True due to false positive
```

## 14. Lossy Counting: Finding Frequent Items in a Stream
**Core idea:** Lossy counting estimates frequent items in a data stream using limited memory. It tracks approximate counts and periodically removes items that cannot still be frequent.
### Version a 5-year-old can understand
You count favorite candies, but your paper is small. Every few minutes, you erase candies that have too few marks to matter.
**Analogy legend:**
- candies = stream items
- marks = counts
- small paper = limited memory
- erase low-count candies = pruning
- popular candies = frequent items

### Version a 10-year-old can understand
Lossy counting divides the stream into buckets. It keeps approximate counts and removes items whose count plus possible error is too small. This helps find frequent items without storing everything.
### Undergraduate version
With error parameter epsilon, bucket width is ceil(1/epsilon). Each tracked item stores frequency f and error delta. At bucket boundaries, remove items where f + delta <= current bucket id.
### Graduate student version
Lossy counting is a deterministic streaming algorithm for approximate frequent-item mining with bounded error. It trades exactness for sublinear memory and supports threshold-based heavy hitter detection.
### Formula in plain English
**Formula-style statement:** `Bucket width = ceiling(1/epsilon); prune item if frequency + error <= current bucket number`

**Plain-English statement:** Smaller epsilon means wider buckets, more memory, and more accurate counts. Pruning removes items that cannot plausibly become frequent enough.
### Practice problem
If epsilon=0.2, what is the bucket width?
**By-hand solution:**
- Bucket width = ceiling(1/epsilon).
- 1/0.2 = 5.
- Bucket width = 5.

**Python implementation / solution:**
```python
import math

class LossyCounter:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.width = math.ceil(1 / epsilon)
        self.n = 0
        self.table = {}

    def add(self, item):
        self.n += 1
        bucket = math.ceil(self.n / self.width)

        if item in self.table:
            count, error = self.table[item]
            self.table[item] = (count + 1, error)
        else:
            self.table[item] = (1, bucket - 1)

        if self.n % self.width == 0:
            self._prune(bucket)

    def _prune(self, bucket):
        self.table = {
            item: (count, error)
            for item, (count, error) in self.table.items()
            if count + error > bucket
        }

counter = LossyCounter(epsilon=0.2)
for item in ["a", "b", "a", "c", "a", "b", "a"]:
    counter.add(item)
print(counter.table)
```

## Quick Formula Reframe Cheat Sheet

| Topic | Instead of seeing the formula as... | Say this to yourself |
|---|---|---|
| N-gram probability | count(pair) / count(context) | "Of the times I saw this context, how often did this next word happen?" |
| Add-k smoothing | add k and add kV | "Give every possible word a tiny pretend count, then recalculate." |
| Interpolation | weighted probability sum | "Ask multiple models and average them based on trust." |
| Perplexity | exp(-mean log p) | "How confused is the model on average? Lower is better." |
| Forward algorithm | emission times sum of previous paths | "Add all possible hidden stories that reach this state." |
| Viterbi | emission times max previous path | "Keep only the best hidden story that reaches this state." |
| Jaccard | intersection / union | "Shared items divided by total unique items." |
| Cosine | dot / lengths | "Do the vectors point in the same direction?" |
| Edit distance | dynamic min edits | "What is the cheapest set of insert/delete/change operations?" |
| DTW | current cost + cheapest previous path | "What is the cheapest flexible alignment so far?" |
| Moving average | average last k values | "Smooth the line by averaging nearby values." |
| AR(1) | c + phi previous value | "Baseline plus carryover from the last value." |
| Reservoir sampling | k/n inclusion | "Keep a fair sample when the stream is too large." |
| Bloom filter | all hashed bits must be 1 | "Any empty required spot means definitely not present." |
| Lossy counting | prune low possible counts | "Keep likely frequent items and drop items too small to matter." |
