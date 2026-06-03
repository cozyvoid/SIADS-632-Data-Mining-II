# Chapter 3 Study Guide: N-gram Language Models  
**Source:** *Speech and Language Processing*, Chapter 3, Daniel Jurafsky & James H. Martin, draft of January 6, 2026.

---

## 1. Big Picture

This chapter introduces **n-gram language models**, one of the simplest forms of language modeling. A **language model (LM)** assigns probabilities to word or token sequences. It can answer questions like:

- What word is likely to come next?
- How probable is this full sentence?
- Which of two candidate sentences is more plausible?
- How surprised should a model be by a held-out test set?

The chapter uses n-grams to introduce core ideas that also matter for modern large language models:

- prediction of upcoming tokens
- training, development, and test sets
- maximum likelihood estimation
- perplexity
- sampling
- overfitting and generalization
- smoothing, interpolation, and backoff

Although modern language models are neural and transformer-based, n-grams are valuable because the mechanics are transparent and mathematically tractable.

---

## 2. What Is a Language Model?

A **language model** gives a probability distribution over possible next words or tokens.

Example:

> The water of Walden Pond is so beautifully ___

A language model might assign high probability to words like:

- blue
- clear
- green

and very low probability to words like:

- refrigerator
- this

A language model can also assign a probability to a whole sentence. For example, it should assign a higher probability to:

> all of a sudden I notice three guys standing on the sidewalk

than to the same words in a random order:

> on guys all I of notice sidewalk three a sudden standing the

### Why language models matter

Language models are useful in:

- spelling correction
- grammar correction
- speech recognition
- machine translation
- autocomplete
- augmentative and alternative communication systems
- modern large language model training

The key idea is that predicting the next word forces a model to learn patterns about syntax, semantics, discourse, genre, and world knowledge.

---

## 3. N-grams

An **n-gram** is a sequence of `n` words or tokens.

| Name | Size | Example |
|---|---:|---|
| unigram | 1 | `water` |
| bigram | 2 | `the water` |
| trigram | 3 | `the water of` |
| 4-gram | 4 | `the water of Walden` |

An **n-gram language model** estimates the probability of a word using only the previous `n - 1` words.

For example:

- a **bigram** model uses the previous 1 word
- a **trigram** model uses the previous 2 words
- a **4-gram** model uses the previous 3 words

---

## 4. The Chain Rule of Probability

For a sequence of words:

```text
w1, w2, ..., wn
```

the joint probability is:

```text
P(w1, w2, ..., wn)
```

Using the chain rule:

```text
P(w1, w2, ..., wn)
= P(w1)P(w2|w1)P(w3|w1,w2)...P(wn|w1,...,w(n-1))
```

Compactly:

```text
P(w1:n) = product from k=1 to n of P(wk | w1:k-1)
```

### Explanation

This says that the probability of a full sentence can be decomposed into a sequence of next-word predictions.

For example:

```text
P(I want Chinese food)
= P(I)P(want|I)P(Chinese|I,want)P(food|I,want,Chinese)
```

The problem is that exact histories are often too long and too rare to estimate from counts. Most full histories will never appear in a finite corpus.

---

## 5. The Markov Assumption

The **Markov assumption** says that we approximate the probability of the next word using only a limited recent context.

For a bigram model:

```text
P(wn | w1, ..., w(n-1)) ≈ P(wn | w(n-1))
```

Example:

```text
P(blue | The water of Walden Pond is so beautifully)
```

is approximated as:

```text
P(blue | beautifully)
```

For a general n-gram model of order `N`:

```text
P(wn | w1, ..., w(n-1)) ≈ P(wn | w(n-N+1), ..., w(n-1))
```

### Plain-language explanation

Instead of asking:

> Given everything that came before, what comes next?

an n-gram model asks:

> Given only the last few words, what comes next?

This is computationally much easier, but it loses long-range information.

---

## 6. Estimating N-gram Probabilities with Maximum Likelihood Estimation

The simplest way to estimate n-gram probabilities is **maximum likelihood estimation (MLE)**.

MLE estimates probabilities by counting events in a corpus and normalizing the counts.

### Bigram MLE formula

```text
P(wn | w(n-1)) = C(w(n-1), wn) / C(w(n-1))
```

where:

- `C(w(n-1), wn)` is the count of the bigram
- `C(w(n-1))` is the count of the previous word as a context

### General n-gram MLE formula

```text
P(wn | w(n-N+1:n-1)) = C(w(n-N+1:n-1), wn) / C(w(n-N+1:n-1))
```

### Example corpus

Consider:

```text
<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
```

The symbols `<s>` and `</s>` mark sentence beginning and sentence ending.

Some bigram probabilities:

```text
P(I | <s>) = 2/3
P(Sam | <s>) = 1/3
P(am | I) = 2/3
```

The model estimates these by counting what follows each context.

---

## 7. Computing Sentence Probability with a Bigram Model

Given a bigram model, the probability of a sentence is approximated as:

```text
P(w1:n) ≈ product from k=1 to n of P(wk | w(k-1))
```

Example sentence:

```text
<s> i want english food </s>
```

Bigram decomposition:

```text
P(i|<s>) P(want|i) P(english|want) P(food|english) P(</s>|food)
```

Using the chapter's example probabilities:

```text
0.25 × 0.33 × 0.0011 × 0.5 × 0.68 = 0.000031
```

### Interpretation

The sentence probability is small because multiplying many probabilities together produces a small number. This is normal. For long sequences, probabilities become extremely tiny, which is why language models usually use log probabilities.

---

## 8. Log Probabilities

Language model probabilities are usually stored and computed in **log space**.

Instead of multiplying probabilities:

```text
p1 × p2 × p3
```

we add log probabilities:

```text
log(p1) + log(p2) + log(p3)
```

To convert back:

```text
p1p2p3 = exp(log(p1) + log(p2) + log(p3))
```

### Why log probabilities are used

Raw probabilities get extremely small as sentences get longer. This can cause **numerical underflow**, where the computer rounds a tiny number down to zero. Log probabilities avoid this problem.

---

## 9. Larger N-gram Models

The chapter focuses heavily on bigrams for clarity, but real n-gram systems often use:

- trigrams
- 4-grams
- 5-grams

Larger contexts can improve predictions when enough training data is available.

However, the number of possible n-grams grows explosively.

If the vocabulary size is `V`:

- possible bigrams: `V^2`
- possible trigrams: `V^3`
- possible 4-grams: `V^4`

This creates serious sparsity and storage problems.

### Efficiency techniques

Large n-gram systems may use:

- quantized probabilities
- hashes for word strings
- reverse tries
- pruning
- entropy-based pruning
- specialized toolkits such as KenLM

---

## 10. Training, Development, and Test Sets

To evaluate a language model, we split data into separate sets.

| Dataset | Purpose |
|---|---|
| training set | learn model parameters/counts |
| development set | tune choices and hyperparameters |
| test set | final unbiased evaluation |

### Training set

For n-grams, the training set provides the counts used to estimate probabilities.

### Development set

The development set is used to make modeling decisions, such as:

- choosing the n-gram order
- choosing smoothing parameters
- choosing interpolation weights

### Test set

The test set is held out until the end. It estimates how well the model generalizes to unseen data.

### Data contamination

**Data contamination** occurs when test data leaks into the training process.

This makes the model look better than it really is because the test sentences have already influenced the model.

---

## 11. Extrinsic vs. Intrinsic Evaluation

### Extrinsic evaluation

An **extrinsic evaluation** tests the language model inside a real downstream application.

Examples:

- Does speech recognition accuracy improve?
- Does machine translation quality improve?
- Does autocomplete help users type faster?

This is the most meaningful evaluation, but it can be expensive.

### Intrinsic evaluation

An **intrinsic evaluation** measures the model directly, independent of a full application.

The standard intrinsic metric for language models is **perplexity**.

---

## 12. Perplexity

Perplexity measures how surprised a model is by a test set.

For a test sequence:

```text
W = w1, w2, ..., wN
```

perplexity is:

```text
PP(W) = P(w1, w2, ..., wN)^(-1/N)
```

Equivalently:

```text
PP(W) = Nth_root(1 / P(w1, w2, ..., wN))
```

Using the chain rule:

```text
PP(W) = Nth_root(product from i=1 to N of 1 / P(wi | w1, ..., w(i-1)))
```

### Key interpretation

- Higher probability test set → lower perplexity
- Lower probability test set → higher perplexity
- Lower perplexity is better

### Bigram perplexity

For a bigram model:

```text
PP(W) = Nth_root(product from i=1 to N of 1 / P(wi | w(i-1)))
```

### Why not just use raw probability?

Raw sequence probability depends heavily on length. Longer texts automatically have smaller probabilities because they involve more multiplied terms. Perplexity normalizes by the number of tokens, making comparisons more meaningful.

---

## 13. Perplexity as Weighted Average Branching Factor

Perplexity can be interpreted as a **weighted average branching factor**.

The branching factor is the number of possible next words.

Suppose a language only has:

```text
red, blue, green
```

If each word is equally likely:

```text
P(red)=P(blue)=P(green)=1/3
```

then perplexity is 3.

If `red` is much more likely:

```text
P(red)=0.8, P(blue)=0.1, P(green)=0.1
```

then a test set mostly containing `red` will have lower perplexity because the model is less surprised.

---

## 14. Sampling from a Language Model

**Sampling** means generating text by randomly choosing words according to the model's probabilities.

### Unigram sampling

A unigram model chooses each word independently based on its overall frequency.

This tends to produce incoherent text because word order is ignored.

### Bigram sampling

A bigram model:

1. starts with `<s>`
2. samples a likely next word
3. uses that sampled word as the next context
4. repeats until it samples `</s>`

### Higher-order sampling

Trigram and 4-gram models use longer histories, so their generated text often looks more coherent.

However, very high-order n-grams can start copying the training corpus instead of generalizing.

---

## 15. Generalization vs. Overfitting

As n increases, the model uses more context.

This can improve local coherence, but it also increases the risk of **overfitting**.

### Unigram model

- weak coherence
- no word order
- high generalization
- low specificity

### Bigram model

- some local coherence
- captures common word pairs
- still limited

### Trigram model

- better local fluency
- captures short phrases
- more sparse

### 4-gram or 5-gram model

- may sound very fluent
- can copy exact training phrases
- high sparsity
- overfitting risk

### Genre dependence

A model trained on Shakespeare will produce Shakespeare-like text. A model trained on the Wall Street Journal will produce business-news-like text.

If the training and test domains differ too much, performance suffers.

---

## 16. Unseen Words and Tokens

Traditional word-based n-grams can struggle when the test set contains words unseen in training.

Modern NLP systems often avoid this by using **subword tokens**, such as Byte Pair Encoding (BPE). A new word can be broken into smaller known pieces.

Example:

```text
Jurafsky
```

could be represented as smaller subword units if the full word was unseen.

---

## 17. The Zero-count Problem

A major problem with MLE n-gram models is that many valid word sequences are missing from the training corpus.

If an n-gram never appears in training:

```text
C(w(n-1), wn) = 0
```

then:

```text
P(wn | w(n-1)) = 0
```

This creates two problems:

1. the model underestimates valid unseen sequences
2. if any test-set n-gram has probability 0, the full test-set probability becomes 0

This makes perplexity impossible to compute because it involves inverse probability.

---

## 18. Smoothing and Discounting

**Smoothing** or **discounting** solves the zero-count problem by moving some probability mass from seen events to unseen events.

The basic intuition:

> Take a little probability from events we have already seen and give it to events we have not seen.

---

## 19. Laplace/Add-one Smoothing

The simplest smoothing method is **Laplace smoothing**, also called **add-one smoothing**.

### Unigram add-one smoothing

Unsmoothed unigram probability:

```text
P(wi) = ci / N
```

Add-one smoothed probability:

```text
P_Laplace(wi) = (ci + 1) / (N + V)
```

where:

- `ci` is the count of word `wi`
- `N` is the total number of tokens
- `V` is the vocabulary size

### Bigram add-one smoothing

```text
P_Laplace(wn | w(n-1)) = (C(w(n-1), wn) + 1) / (C(w(n-1)) + V)
```

### Why the denominator adds `V`

If we add 1 to every possible next word, and there are `V` possible next words, then we have added `V` total counts.

### Problem with add-one smoothing

Add-one smoothing often moves too much probability mass to unseen events. It is useful pedagogically, but it performs poorly for serious n-gram language modeling.

---

## 20. Add-k Smoothing

Add-k smoothing generalizes add-one smoothing by adding a fractional value `k`.

```text
P_Add-k(wn | w(n-1)) = (C(w(n-1), wn) + k) / (C(w(n-1)) + kV)
```

where `0 < k < 1` is often used.

Example values might be:

- `k=0.5`
- `k=0.1`
- `k=0.01`

The value of `k` can be tuned on a development set.

Add-k smoothing is less extreme than add-one smoothing, but it is still not considered strong for modern n-gram language modeling.

---

## 21. Interpolation

**Interpolation** combines multiple n-gram models.

For a trigram model, instead of relying only on:

```text
P(wn | w(n-2), w(n-1))
```

we combine:

- unigram probability
- bigram probability
- trigram probability

### Linear interpolation

```text
P_hat(wn | w(n-2), w(n-1))
= lambda1 * P(wn)
+ lambda2 * P(wn | w(n-1))
+ lambda3 * P(wn | w(n-2), w(n-1))
```

The weights must sum to 1:

```text
lambda1 + lambda2 + lambda3 = 1
```

### Interpretation

Interpolation says:

> Use the most specific context when it is reliable, but also mix in more general evidence from shorter contexts.

This helps when higher-order n-gram counts are sparse.

---

## 22. Context-conditioned Interpolation

A more advanced interpolation method allows the lambda weights to depend on the context.

For example, if a particular trigram context has lots of data, the trigram weight can be higher.

If the context is rare, the model can rely more on bigram or unigram probabilities.

---

## 23. Backoff and Stupid Backoff

**Backoff** uses a shorter-context model only when the longer-context n-gram is unavailable.

Example:

1. Try trigram probability.
2. If unavailable, back off to bigram.
3. If unavailable, back off to unigram.

### Stupid backoff

Stupid backoff is a simple, scalable method.

It does not produce a true probability distribution. Instead, it produces scores.

If the higher-order n-gram exists, use its relative frequency. Otherwise, back off and multiply by a fixed weight, often:

```text
lambda = 0.4
```

General form:

```text
S(wi | w(i-N+1:i-1)) =
    count(w(i-N+1:i)) / count(w(i-N+1:i-1)), if the higher-order n-gram count is positive
    lambda * S(wi | w(i-N+2:i-1)), otherwise
```

### Interpretation

Stupid backoff is not theoretically elegant, but it is simple and works well at very large scale.

---

## 24. Advanced: Entropy, Cross-entropy, and Perplexity

### Entropy

Entropy measures uncertainty or information content.

```text
H(X) = - sum over x of p(x) log2 p(x)
```

If all outcomes are equally likely, entropy is higher. If one outcome is very likely, entropy is lower.

### Entropy rate

For language, we care about sequences. Entropy rate is the average uncertainty per word.

```text
H(L) = limit as n approaches infinity of (1/n) H(w1:n)
```

### Cross-entropy

Cross-entropy evaluates how well a model `m` approximates the true distribution `p`.

```text
H(p,m) = limit as n approaches infinity of -1/n log m(w1, ..., wn)
```

A better model has lower cross-entropy.

### Perplexity relation

Perplexity is:

```text
PP(W) = 2^H(W)
```

where:

```text
H(W) = -1/N log P(w1, ..., wN)
```

So perplexity is the exponentiated average negative log probability.

---

# Worked Examples

## Example 1: Bigram Probability from Counts

Suppose the corpus contains:

```text
I want Chinese food
I want pizza
I like Chinese food
```

Count:

- `I want`: 2
- `I like`: 1
- `I`: 3

Then:

```text
P(want | I) = C(I, want) / C(I) = 2/3
P(like | I) = 1/3
```

---

## Example 2: Sentence Probability with Bigrams

Suppose:

```text
P(I | <s>) = 0.5
P(want | I) = 0.7
P(food | want) = 0.2
P(</s> | food) = 0.4
```

Then:

```text
P(<s> I want food </s>) = 0.5 × 0.7 × 0.2 × 0.4 = 0.028
```

---

## Example 3: Add-one Smoothing

Suppose:

- vocabulary size `V=5`
- `C(I,want)=2`
- `C(I)=3`

Unsmoothed:

```text
P(want | I) = 2/3
```

Add-one smoothed:

```text
P_Laplace(want | I) = (2 + 1) / (3 + 5) = 3/8 = 0.375
```

The probability decreases because smoothing reserves probability mass for unseen words.

---

## Example 4: Perplexity

Suppose a model assigns the following probabilities to a 3-token test sequence:

```text
0.5, 0.25, 0.125
```

The sequence probability is:

```text
0.5 × 0.25 × 0.125 = 0.015625
```

Perplexity:

```text
PP = (0.015625)^(-1/3) = 4
```

Interpretation:

> On average, the model is as uncertain as choosing among about 4 equally likely next words.

---

# Exercise Solutions and Solution Guides

## Exercise 3.1

**Prompt:** Write out the equation for trigram probability estimation. Then write out all non-zero trigram probabilities for the “I am Sam” corpus.

### Trigram MLE formula

```text
P(w3 | w1, w2) = C(w1, w2, w3) / C(w1, w2)
```

### Corpus

```text
<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
```

### Non-zero trigram probabilities

For context `(<s>, I)`:

```text
P(am | <s>, I) = 1/2
P(do | <s>, I) = 1/2
```

For context `(<s>, Sam)`:

```text
P(I | <s>, Sam) = 1
```

For context `(I, am)`:

```text
P(Sam | I, am) = 1/2
P(</s> | I, am) = 1/2
```

For context `(am, Sam)`:

```text
P(</s> | am, Sam) = 1
```

For context `(Sam, I)`:

```text
P(am | Sam, I) = 1
```

For the remaining contexts:

```text
P(not | I, do) = 1
P(like | do, not) = 1
P(green | not, like) = 1
P(eggs | like, green) = 1
P(and | green, eggs) = 1
P(ham | eggs, and) = 1
P(</s> | and, ham) = 1
```

---

## Exercise 3.2

**Prompt:** Calculate the probability of `i want chinese food`, once using the unsmoothed probabilities and once using add-one smoothed probabilities.

Sentence:

```text
<s> i want chinese food </s>
```

### Unsmoothed probability

Use:

```text
P(i | <s>) = 0.25
P(want | i) = 0.33
P(chinese | want) = 0.0065
P(food | chinese) = 0.52
P(</s> | food) = 0.68
```

Then:

```text
P = 0.25 × 0.33 × 0.0065 × 0.52 × 0.68
P = 0.000189618
```

### Add-one smoothed probability

Use:

```text
P(i | <s>) = 0.19
P(want | i) = 0.21
P(chinese | want) = 0.0029
P(food | chinese) = 0.052
P(</s> | food) = 0.40
```

Then:

```text
P = 0.19 × 0.21 × 0.0029 × 0.052 × 0.40
P = 0.000002406768
```

or:

```text
2.406768 × 10^-6
```

---

## Exercise 3.3

**Prompt:** Which probability from Exercise 3.2 is higher, unsmoothed or smoothed? Explain why.

The unsmoothed probability is higher:

```text
0.000189618
```

The smoothed probability is lower:

```text
0.000002406768
```

### Explanation

Add-one smoothing gives some probability mass to unseen bigrams. To do that, it must take probability mass away from seen bigrams. Since `i want chinese food` is made of bigrams that were already observed in the training data, smoothing reduces their probabilities.

In this case, add-one smoothing especially reduces:

```text
P(food | chinese)
```

from:

```text
0.52
```

to:

```text
0.052
```

This large reduction makes the full sentence probability much smaller.

---

## Exercise 3.4

**Prompt:** Given the modified corpus, use a bigram language model with add-one smoothing to compute `P(Sam | am)`.

### Corpus

```text
<s> I am Sam </s>
<s> Sam I am </s>
<s> I am Sam </s>
<s> I do not like green eggs and Sam </s>
```

### Step 1: Count `C(am, Sam)`

The bigram `am Sam` appears twice:

```text
I am Sam
I am Sam
```

So:

```text
C(am, Sam) = 2
```

### Step 2: Count `C(am)`

The word `am` appears as a previous-word context three times:

```text
I am Sam
I am </s>
I am Sam
```

So:

```text
C(am) = 3
```

### Step 3: Find vocabulary size `V`

Unique tokens:

```text
<s>, I, am, Sam, </s>, do, not, like, green, eggs, and
```

So:

```text
V = 11
```

### Step 4: Apply add-one smoothing

```text
P_Laplace(Sam | am) = (C(am, Sam) + 1) / (C(am) + V)
P_Laplace(Sam | am) = (2 + 1) / (3 + 11)
P_Laplace(Sam | am) = 3/14
P_Laplace(Sam | am) = 0.2143
```

### Final answer

```text
P(Sam | am) = 3/14 ≈ 0.2143
```

---

## Exercise 3.5

**Prompt:** Show that without an end-symbol `</s>`, an unsmoothed bigram grammar does not assign a single probability distribution across all sentence lengths.

### Training corpus without end-symbol

```text
<s> a b
<s> b b
<s> b a
<s> a a
```

### Step 1: Estimate start probabilities

From `<s>`:

- `<s> a`: 2 times
- `<s> b`: 2 times

```text
P(a | <s>) = 2/4 = 0.5
P(b | <s>) = 2/4 = 0.5
```

### Step 2: Estimate transition probabilities

From `a`:

- `a b`: 1
- `a a`: 1

```text
P(a | a) = 0.5
P(b | a) = 0.5
```

From `b`:

- `b b`: 1
- `b a`: 1

```text
P(a | b) = 0.5
P(b | b) = 0.5
```

### Step 3: Sum probabilities of all 2-word sentences

Possible 2-word sentences over `{a,b}`:

```text
a a
a b
b a
b b
```

Each has probability:

```text
0.5 × 0.5 = 0.25
```

Sum:

```text
4 × 0.25 = 1.0
```

### Step 4: Sum probabilities of all 3-word sentences

There are `2^3 = 8` possible 3-word sentences.

Each has probability:

```text
0.5 × 0.5 × 0.5 = 0.125
```

Sum:

```text
8 × 0.125 = 1.0
```

### Explanation

The model assigns total probability 1.0 to all 2-word sentences and also total probability 1.0 to all 3-word sentences. Therefore, it is not defining one probability distribution over all possible sentence lengths. Instead, each length behaves like a separate distribution.

This is why an end-symbol `</s>` is needed: it lets the model assign probability to stopping.

---

## Exercise 3.6

**Prompt:** Suppose we train a trigram language model with add-one smoothing. The corpus contains `V` word types. Express a formula for estimating `P(w3 | w1, w2)`.

For trigram add-one smoothing:

```text
P(w3 | w1, w2) = (c(w1, w2, w3) + 1) / (c(w1, w2) + V)
```

### Explanation

The numerator adds 1 to the observed trigram count.

The denominator adds `V` because each possible next word gets one added count.

---

## Exercise 3.7

**Prompt:** Using linear interpolation smoothing between an MLE bigram model and an MLE unigram model with `lambda1 = 1/2` and `lambda2 = 1/2`, compute `P(Sam | am)`.

### Corpus

```text
<s> I am Sam </s>
<s> Sam I am </s>
<s> I am Sam </s>
<s> I do not like green eggs and Sam </s>
```

### Step 1: Bigram probability

From Exercise 3.4:

```text
C(am, Sam) = 2
C(am) = 3
```

So:

```text
P_bigram(Sam | am) = 2/3
```

### Step 2: Unigram probability

Total tokens:

- sentence 1: 5
- sentence 2: 5
- sentence 3: 5
- sentence 4: 10

```text
N = 25
```

`Sam` appears 4 times.

```text
P_unigram(Sam) = 4/25 = 0.16
```

### Step 3: Interpolate

```text
P(Sam | am) = lambda1 * P_unigram(Sam) + lambda2 * P_bigram(Sam | am)
P(Sam | am) = (1/2)(4/25) + (1/2)(2/3)
P(Sam | am) = 0.08 + 0.3333
P(Sam | am) = 0.4133
```

### Final answer

```text
P(Sam | am) ≈ 0.4133
```

---

## Exercise 3.8

**Prompt:** Write a program to compute unsmoothed unigrams and bigrams.

```python
from collections import Counter
from typing import List

def tokenize_sentences(sentences: List[str]) -> List[List[str]]:
    """add sentence boundary markers and split each sentence into tokens."""
    return [["<s>"] + sent.split() + ["</s>"] for sent in sentences]

def compute_unigrams(tokenized_sentences: List[List[str]]) -> Counter:
    """count individual tokens across the corpus."""
    unigram_counts = Counter()
    for sent in tokenized_sentences:
        unigram_counts.update(sent)
    return unigram_counts

def compute_bigrams(tokenized_sentences: List[List[str]]) -> Counter:
    """count adjacent token pairs across the corpus."""
    bigram_counts = Counter()
    for sent in tokenized_sentences:
        for i in range(len(sent) - 1):
            bigram_counts[(sent[i], sent[i + 1])] += 1
    return bigram_counts

def bigram_probability(prev_word: str, word: str, unigram_counts: Counter, bigram_counts: Counter) -> float:
    """compute unsmoothed maximum likelihood bigram probability."""
    if unigram_counts[prev_word] == 0:
        return 0.0
    return bigram_counts[(prev_word, word)] / unigram_counts[prev_word]

sentences = [
    "I am Sam",
    "Sam I am",
    "I do not like green eggs and ham"
]

tokenized = tokenize_sentences(sentences)
unigrams = compute_unigrams(tokenized)
bigrams = compute_bigrams(tokenized)

print(unigrams)
print(bigrams)
print(bigram_probability("I", "am", unigrams, bigrams))
```

---

## Exercise 3.9

**Prompt:** Run your n-gram program on two small corpora of your choice and compare unigram and bigram statistics.

Because this depends on your chosen corpora, here is a reusable template.

```python
from collections import Counter

corpus_a = [
    "the patient needs a follow up appointment",
    "the nurse called the patient",
    "the doctor reviewed the chart"
]

corpus_b = [
    "the team scored a goal",
    "the coach called the play",
    "the player passed the ball"
]

def get_counts(corpus):
    tokenized = tokenize_sentences(corpus)
    return compute_unigrams(tokenized), compute_bigrams(tokenized)

uni_a, bi_a = get_counts(corpus_a)
uni_b, bi_b = get_counts(corpus_b)

print("Corpus A common unigrams:")
print(uni_a.most_common(10))

print("\nCorpus B common unigrams:")
print(uni_b.most_common(10))

print("\nCorpus A common bigrams:")
print(bi_a.most_common(10))

print("\nCorpus B common bigrams:")
print(bi_b.most_common(10))
```

### Example explanation

Corpus A is healthcare-oriented, so words like `patient`, `nurse`, `doctor`, and `chart` are more common. Corpus B is sports-oriented, so words like `team`, `goal`, `coach`, `player`, and `ball` are more common.

Interesting bigrams also reflect domain differences:

- healthcare: `the patient`, `the doctor`, `follow up`
- sports: `the team`, `scored a`, `the ball`

This shows that n-gram statistics encode genre and domain information.

---

## Exercise 3.10

**Prompt:** Add an option to your program to generate random sentences.

```python
import random
from collections import defaultdict

def build_bigram_next_word_distribution(bigram_counts: Counter):
    """map each previous word to possible next words and their counts."""
    next_words = defaultdict(list)
    for (prev_word, word), count in bigram_counts.items():
        next_words[prev_word].append((word, count))
    return next_words

def sample_next_word(options):
    """sample one word according to count-based probabilities."""
    words = [word for word, count in options]
    weights = [count for word, count in options]
    return random.choices(words, weights=weights, k=1)[0]

def generate_sentence(bigram_counts: Counter, max_len: int = 30) -> str:
    """generate a sentence from a bigram model."""
    next_words = build_bigram_next_word_distribution(bigram_counts)
    current = "<s>"
    output = []

    for _ in range(max_len):
        if current not in next_words:
            break

        next_word = sample_next_word(next_words[current])

        if next_word == "</s>":
            break

        output.append(next_word)
        current = next_word

    return " ".join(output)

print(generate_sentence(bigrams))
```

### Explanation

The generator starts at `<s>`, samples a likely next word, then repeatedly samples from the distribution associated with the current word.

---

## Exercise 3.11

**Prompt:** Add an option to compute the perplexity of a test set.

```python
import math

def corpus_bigram_log_probability(test_sentences, unigram_counts, bigram_counts):
    """compute the log probability of a test corpus under an unsmoothed bigram model."""
    tokenized = tokenize_sentences(test_sentences)
    log_prob = 0.0
    token_count = 0

    for sent in tokenized:
        for i in range(1, len(sent)):
            prev_word = sent[i - 1]
            word = sent[i]

            prob = bigram_probability(prev_word, word, unigram_counts, bigram_counts)

            if prob == 0:
                return float("-inf"), token_count

            log_prob += math.log(prob)
            token_count += 1

    return log_prob, token_count

def bigram_perplexity(test_sentences, unigram_counts, bigram_counts):
    """compute perplexity from average negative log probability."""
    log_prob, token_count = corpus_bigram_log_probability(
        test_sentences,
        unigram_counts,
        bigram_counts
    )

    if log_prob == float("-inf"):
        return float("inf")

    return math.exp(-log_prob / token_count)

test_sentences = [
    "I am Sam"
]

print(bigram_perplexity(test_sentences, unigrams, bigrams))
```

### Explanation

Perplexity can be computed using log probabilities:

```text
PP(W) = exp(-(1/N) * sum_i log P(wi | context))
```

If any bigram probability is zero in an unsmoothed model, perplexity becomes infinite.

---

## Exercise 3.12

**Prompt:** A training set of 100 numbers has 91 zeros and one each of digits 1–9. The test set is:

```text
0 0 0 0 0 3 0 0 0 0
```

What is the unigram perplexity?

### Step 1: Estimate unigram probabilities

Training counts:

```text
P(0) = 91/100 = 0.91
```

For each digit 1 through 9:

```text
P(d) = 1/100 = 0.01
```

So:

```text
P(3) = 0.01
```

### Step 2: Count test-set items

The test set contains:

- nine `0`s
- one `3`

### Step 3: Compute test-set probability

```text
P(W) = 0.91^9 × 0.01
P(W) ≈ 0.00428
```

### Step 4: Compute perplexity

The test set has `N=10` tokens.

```text
PP(W) = P(W)^(-1/10)
PP(W) = (0.91^9 × 0.01)^(-1/10)
PP(W) ≈ 1.725
```

### Final answer

```text
PP(W) ≈ 1.73
```

### Interpretation

The perplexity is low because the test set mostly contains `0`, and the model assigns very high probability to `0`.

---

# Key Terms

| Term | Meaning |
|---|---|
| language model | model that assigns probabilities to word/token sequences |
| n-gram | sequence of `n` tokens |
| unigram | one-token sequence |
| bigram | two-token sequence |
| trigram | three-token sequence |
| Markov assumption | assumption that the future depends only on limited recent history |
| MLE | estimate probabilities from relative frequency counts |
| training set | data used to learn model parameters |
| development set | data used to tune model choices |
| test set | held-out data used for final evaluation |
| data contamination | leakage of test information into training |
| perplexity | inverse normalized probability of a test set |
| sampling | generating text according to model probabilities |
| smoothing | assigning nonzero probability to unseen events |
| discounting | taking probability mass from seen events |
| Laplace smoothing | add-one smoothing |
| add-k smoothing | smoothing by adding fractional count `k` |
| interpolation | combining multiple n-gram orders |
| backoff | using lower-order n-grams when higher-order n-grams are unavailable |
| entropy | measure of uncertainty/information |
| cross-entropy | measure of how well a model approximates a true distribution |

---

# High-yield Oral Exam Notes

## What is the main limitation of n-gram models?

N-gram models only look at a fixed window of previous words. They cannot naturally model long-distance dependencies. They also suffer from sparsity because the number of possible n-grams grows exponentially as `n` increases.

## Why do we need smoothing?

Without smoothing, unseen n-grams get probability zero. If any test-set n-gram has probability zero, the full test-set probability becomes zero and perplexity becomes undefined or infinite.

## Why is lower perplexity better?

Perplexity is inversely related to probability. A model with lower perplexity assigns higher probability to the test data, meaning it is less surprised by the observed sequence.

## Why should train and test sets be separate?

A separate test set gives an unbiased estimate of generalization. If the model has seen test data during training, evaluation is contaminated and overly optimistic.

## Why might a higher-order n-gram overfit?

A higher-order n-gram uses a longer context. With limited data, many long contexts occur only once, so the model may memorize exact training phrases rather than learn generalizable patterns.

## How does interpolation help?

Interpolation combines specific but sparse higher-order evidence with broader but more reliable lower-order evidence. This allows the model to use context when available while still assigning probability to unseen or rare contexts.

---

# Compact Formula Sheet

## Chain rule

```text
P(w1:n) = product from k=1 to n of P(wk | w1:k-1)
```

## Bigram approximation

```text
P(wn | w1, ..., w(n-1)) ≈ P(wn | w(n-1))
```

## General n-gram approximation

```text
P(wn | w1, ..., w(n-1)) ≈ P(wn | w(n-N+1:n-1))
```

## Bigram MLE

```text
P(wn | w(n-1)) = C(w(n-1), wn) / C(w(n-1))
```

## General n-gram MLE

```text
P(wn | w(n-N+1:n-1)) = C(w(n-N+1:n-1), wn) / C(w(n-N+1:n-1))
```

## Add-one unigram

```text
P(wi) = (ci + 1) / (N + V)
```

## Add-one bigram

```text
P(wn | w(n-1)) = (C(w(n-1), wn) + 1) / (C(w(n-1)) + V)
```

## Add-k bigram

```text
P(wn | w(n-1)) = (C(w(n-1), wn) + k) / (C(w(n-1)) + kV)
```

## Linear interpolation

```text
P_hat(wn | w(n-2), w(n-1))
= lambda1 * P(wn)
+ lambda2 * P(wn | w(n-1))
+ lambda3 * P(wn | w(n-2), w(n-1))
```

## Perplexity

```text
PP(W) = P(W)^(-1/N)
```

## Perplexity from log probabilities

```text
PP(W) = exp(-(1/N) * sum_i log P(wi | context))
```
