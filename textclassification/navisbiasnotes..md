


> The model uses the TF-IDF matrix `X` (numbers) and learns
> how strongly each word (feature) is associated with each class.

Then, for a new document, it **computes a score per class** using:

$$
\text{score(Class)} = \sum_i \text{TFIDF}(w_i) \times \log P(w_i \mid \text{Class}) + \log P(\text{Class})
$$

Whichever class has the *higher* score → that’s the predicted class.


##  Mini example

Let’s pretend we’re classifying two types of messages:

| Message | Text               | Class |
| ------- | ------------------ | ----- |
| D₁      | "buy pills"        | spam  |
| D₂      | "cheap pills"      | spam  |
| D₃      | "project meeting"  | ham   |
| D₄      | "meeting schedule" | ham   |

### Vocabulary (from all docs)

`[buy, cheap, pills, project, meeting, schedule]`

So there are 6 words → 6 columns.

---

### Step 1: TF-IDF matrix (simplified numbers)

To keep it readable, let’s assign rough TF-IDF values:

| Doc       | buy | cheap | pills | project | meeting | schedule |
| --------- | --- | ----- | ----- | ------- | ------- | -------- |
| D₁ (spam) | 0.7 | 0     | 0.9   | 0       | 0       | 0        |
| D₂ (spam) | 0   | 0.7   | 0.9   | 0       | 0       | 0        |
| D₃ (ham)  | 0   | 0     | 0     | 0.7     | 0.9     | 0        |
| D₄ (ham)  | 0   | 0     | 0     | 0       | 0.7     | 0.9      |

---

### Step 2: Naïve Bayes learns ( P(w_i \mid \text{Class}) )

The model sums TF-IDF weights within each class and normalizes.

| Word | Sum in Spam | Sum in Ham | (P(w|Spam)) | (P(w|Ham)) |
|------|--------------|-------------|----------------|---------------|
| buy | 0.7 | 0 | 0.7 / (0.7+0.7+0.9+0.9) ≈ 0.19 | 0 |
| cheap | 0.7 | 0 | 0.19 | 0 |
| pills | 1.8 | 0 | 0.49 | 0 |
| project | 0 | 0.7 | 0 | 0.27 |
| meeting | 0 | 1.6 | 0 | 0.62 |
| schedule | 0 | 0.9 | 0 | 0.35 |

(These are simplified normalized values — just to illustrate that spam words get probability mass in spam, ham words in ham.)


### Step 3: Predict a new message

> `"cheap pills meeting"`

TF-IDF (simplified) for this new doc:

| Word    | TF-IDF |
| ------- | ------ |
| cheap   | 0.6    |
| pills   | 0.8    |
| meeting | 0.7    |

Now compute class scores.

#### Spam score

$$
\begin{aligned}
\text{score(Spam)} &= 0.6 \log P(cheap|Spam) + 0.8 \log P(pills|Spam) + 0.7 \log P(meeting|Spam) + \log P(Spam)
\end{aligned}
$$

Since (P(meeting|Spam)) ≈ 0 (word unseen in spam), this term heavily reduces the score — but smoothing keeps it small, not infinite.

#### Ham score

$$
\text{score(Ham)} = 0.6 \log P(cheap|Ham) + 0.8 \log P(pills|Ham) + 0.7 \log P(meeting|Ham) + \log P(Ham)
$$

Here, (P(cheap|Ham)) and (P(pills|Ham)) are near 0 (never seen in ham), but (P(meeting|Ham)) is large.

---

### Step 4: Compare

* Spam: big positive contributions from “cheap”, “pills”
* Ham: big positive contribution from “meeting”, but zero for others

If the total spam score is higher → classified as **spam**.

