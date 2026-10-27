# Formative 3 - Probability Distributions, Bayesian Probability and Gradient Descent 

## Team members (Group 5)
- Liliane Uwase 
- Abigail Salem Tendo 
- Bonane Niyigena 

## How to run 

1. Open the notebook in Colab (use the "open in colab" badge at the top on each notebook).
2. Upload the dataset with the folder icon and wait for the upload to finish:
   - Part 1: `GaltonFamilies.csv`
   - Part 2: `IMDB Dataset.csv` (about 66 MB)
3. Run **Runtime → Run all**.

only python build it `csv` module, Numpy, Scipy and Matplotlib are used so no machine learning libraries were used for this assignment. 


## Part 1 - EM algorithm 

We treat 1,139 heights (205 mothers + 934 children) as a mixture of two Gaussians, with no labels.

- **Initialization:** means at the overall mean ± one standard deviation (62.7 and 69.8), equal variances, 50/50 shares.
- **E-step:** for each height, the probability of belonging to each group (the responsibilities).
- **M-step:** new means, variances and shares, using the responsibilities as weights.
- **Stopping rule:** stop when the log-likelihood changes by less than 0.000001.

**Why not split at the global mean?** The two groups overlap, so a straight line misclassifies people in the middle and biases both averages. EM uses soft assignments instead.

**Test height:** `predict_height(h)` prints P(Child) and P(Parent). For example, 67 inches gives P(Parent) = 0.66 and P(Child) = 0.34.

**Finding:** EM groups by height only. The shorter group holds mostly mothers and daughters, and the taller group mostly sons, because the children in this dataset are adults.

## Part 2 – Bayesian probability

Question: if a review contains a keyword, how likely is it to be positive?
`P(Positive | word) = P(word | Positive) × P(Positive) / P(word)`, using 50,000 IMDb reviews (prior = 0.5).

| Keyword | P(Positive \| word) |
|---|---|
| wonderful | 0.8122 |
| excellent | 0.8074 |
| loved | 0.7536 |
| masterpiece | 0.7274 |
| boring | 0.1940 |
| awful | 0.0985 |
| worst | 0.0927 |
| waste | 0.0691 |

Keywords were chosen because they clearly express an opinion and each appears in more than 1,000 reviews.
"boring" and "masterpiece" are weaker because of negation ("not boring") and sarcasm ("no masterpiece").

## Parts 3 & 4 – Gradient descent

Model: ŷ = X·m + b, with m = [−1, 2], b = [1, 1], learning rate α = 0.01.
Gradients from the chain rule: ∂MSE/∂m = (2/n)·Xᵀe and ∂MSE/∂b = (2/n)·e.

| Update | m | b | MSE after |
|---|---|---|---|
| Start | [−1, 2] | [1, 1] | 61.00 |
| 1 | [−1.45, 0.87] | [0.99, 0.89] | 6.50 |
| 2 | [−1.3331, 1.1765] | [1.0185, 0.9121] | 2.50 |
| 3 | [−1.3690, 1.0956] | [1.0364, 0.8987] | 2.16 |
