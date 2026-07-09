# Part 1: The Test

**Topic: k-Nearest Neighbors (KNN) Theory & Properties**

**Question 1 (2 Points)**
According to the lecture slides, which of the following statements regarding the fundamental properties of the k-Nearest Neighbor (KNN) algorithm are correct?
- [ ] It is a parametric method that learns a fixed set of weights during the training phase.
- [ ] It requires an extensive computationally expensive training phase before predictions can be made.
- [ ] It is capable of handling complex decision boundaries due to its non-parametric nature.
- [ ] It relies on the assumption that similar data points should have similar labels.

**Question 2 (3 Points)**
Based on the visual examples and explanations provided for choosing the right value of $k$ in KNN classification, which of the following outcomes are expected?
- [ ] Setting $k=1$ typically leads to an overly smooth decision boundary, indicating underfitting.
- [ ] A very large $k$ value can lead to underfitting the training data.
- [ ] Setting $k=1$ usually creates highly complex and fragmented decision boundaries, which is a symptom of overfitting.
- [ ] Cross-validation is recommended to find a balance between overfitting and underfitting when choosing $k$.

**Question 3 (2 Points)**
Which of the following are explicitly identified in the slides as limitations of the KNN algorithm for classification?
- [ ] It is computationally expensive for large datasets.
- [ ] It cannot inherently handle multiclass classification problems.
- [ ] It is highly sensitive to noisy data.
- [ ] It is highly sensitive to irrelevant features.

**Topic: KNN Mechanics & Regression**

**Question 4 (3 Points)**
When using KNN for regression problems, the prediction can be calculated as a weighted average. Given a new data point $x$ and a set of the $k$ closest training points $x^{(i)} \in S$, which of the following mathematically reflects the weighting scheme presented in the slides?
- [ ] The weight $w_i$ assigned to a neighbor is directly proportional to the Euclidean distance between $x$ and $x^{(i)}$.
- [ ] The weight $w_i$ is calculated as $\frac{1}{||x - x^{(i)}||}$.
- [ ] As a training point $x^{(i)}$ gets infinitely close to the query point $x$, its weight $w_i$ approaches zero.
- [ ] The final weighted prediction $y$ requires dividing the sum of the weighted neighbor labels by the sum of all weights $w_i$.

**Question 5 (2 Points)**
According to the conclusion slide on KNN, the algorithm is noted to be mathematically or conceptually very similar to using which of the following?
- [ ] Linear models with a Gaussian kernel
- [ ] Probabilistic models with conditionally independent features
- [ ] Decision trees with Laplacian smoothing
- [ ] Linear regression with highly parameterized polynomials

**Question 6 (2 Points)**
According to the lecture slides, what are the characteristics or techniques associated with addressing irrelevant features and noisy data in k-Nearest Neighbors (KNN)?
- [ ] KNN is inherently sensitive to irrelevant features and noisy data.
- [ ] The choice of the hyperparameter $k$ is an important consideration when handling noisy data.
- [ ] Performance on noisy data can be improved using dimensionality reduction methods.
- [ ] Large-margin nearest neighbor methods can be utilized to improve performance when irrelevant features are present.

**Topic: Naive Bayes Fundamentals**

**Question 7 (3 Points)**
Which of the following statements regarding the Naive Bayes algorithm's assumptions and theoretical calculations are explicitly supported by the lecture slides?
- [ ] The algorithm assumes that the features (words in an email) are marginally independent of one another regardless of the class label.
- [ ] Because it is a non-parametric model, Naive Bayes requires no parameter estimation.
- [ ] To compute the final probability of an email being spam, you must strictly calculate the exact denominator value $p(w_1, w_2, ..., w_d)$.
- [ ] The algorithm always performs poorly in practice because its core assumption of conditional independence is almost never true in reality.

**Question 8 (3 Points)**
The Naive Bayes classification rule relies on comparing probabilities. To classify an email consisting of words $w_1, ..., w_d$ as spam or ham, the algorithm evaluates which class yields the higher probability. Which of the following expressions correctly represents the proportional quantity calculated for the `spam` class according to the slides?
- [ ] $p(spam) \cdot p(w_1|spam) \cdot p(w_2|spam) \cdot ... \cdot p(w_d|spam)$
- [ ] $p(spam|w_1) \cdot p(spam|w_2) \cdot ... \cdot p(spam|w_d)$
- [ ] $\frac{p(spam) \cdot p(w_1, w_2, ..., w_d | spam)}{p(spam)}$
- [ ] $\frac{p(w_1|spam) \cdot p(w_2|spam) \cdot ... \cdot p(w_d|spam)}{p(w_1, w_2, ..., w_d)}$

**Topic: Naive Bayes Applied (Spam vs. Ham Example)**

**Question 9 (4 Points)**
In the "Spam vs. No Spam" example (Slide 19), a test email consists of the words: `meeting, meeting, meeting, win`. The training data results in the following probabilities: 
$p(meeting|ham) = \frac{4}{8}$, $p(win|ham) = \frac{0}{8}$, $p(ham) = \frac{3}{4}$. 
If we do **not** use Laplacian smoothing, what occurs when calculating the proportional probability that this email is `ham` ($p(ham|w) \sim \dots$)?
- [ ] The model successfully predicts the email is `ham` because $\frac{4}{8}$ is higher than the equivalent spam probability.
- [ ] The calculated proportional probability for `ham` becomes exactly $0$.
- [ ] The calculation demonstrates the "zero-probability problem" because the word `win` never appeared in a ham email in the training data.
- [ ] The probability calculation for `spam` also automatically evaluates to $0$ to balance the equation.

**Question 10 (3 Points)**
Prior to evaluating any features (words) in the test email, the Naive Bayes algorithm calculates the prior probabilities of the classes. Based on the provided Spam vs. No Spam training data containing 4 emails (1 spam, 3 ham), which of the following is true?
- [ ] $p(spam) = \frac{1}{4}$
- [ ] $p(ham) = \frac{3}{4}$
- [ ] $p(spam) = \frac{1}{3}$
- [ ] The prior probabilities are ignored in the final Naive Bayes proportional comparison.

**Topic: Laplacian Smoothing & Model Regularization**

**Question 11 (4 Points)**
Laplacian Smoothing is introduced to solve probabilities of zero. According to the formula provided in the slides, $p(w_i|spam) = \frac{\text{frequency of } w_i + 1}{\text{number of words in spam } + \text{ total number of unique words}}$. 
Given a training corpus where the total number of words in spam emails is 3, the total number of unique words across ALL emails is 6, and the word `meeting` appears 1 time in spam emails, what is the smoothed probability $p(meeting|spam)$?
- [ ] $\frac{1}{3}$
- [ ] $\frac{2}{9}$
- [ ] $\frac{1 + 1}{3 + 6}$
- [ ] $\frac{1}{6}$

**Question 12 (2 Points)**
According to the Naive Bayes conclusion slides, Laplacian Smoothing serves as a form of:
- [ ] Dimensionality reduction
- [ ] Cross-validation
- [ ] Regularization
- [ ] Non-parametric optimization

**Topic: Algorithm Comparisons & Edge Cases**

**Question 13 (3 Points)**
Based on the concluding remarks for Naive Bayes (Slide 22), why is Naive Bayes considered a very good method when dealing with many features but few data points?
- [ ] Because it uses a complex kernel trick to project features into a higher dimension.
- [ ] Because it requires no training phase whatsoever, making small data sizes irrelevant.
- [ ] Because only very few parameters need to be estimated (specifically, the independent conditional probabilities and class priors).
- [ ] Because it relies heavily on measuring the Euclidean distance between all features, which is highly robust in small datasets.

**Question 14 (2 Points)**
The lecture mentions different approaches to estimating individual conditional probabilities lead to different variants of Naive Bayes. Which of the following variants are explicitly listed in the slides?
- [ ] Multinomial NB
- [ ] Gaussian NB
- [ ] Binomial NB
- [ ] Kernelized NB

**Question 15 (3 Points)**
Comparing KNN and Naive Bayes based strictly on the provided slides, which of the following classifications of these algorithms is correct?
- [ ] Both KNN and Naive Bayes are defined as non-parametric methods.
- [ ] KNN makes predictions based on proximity (distance), while Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem.
- [ ] Naive Bayes assumes individual features are conditionally independent, whereas KNN makes no such independence assumption, relying instead on distance metrics like Euclidean or Manhattan.
- [ ] Both algorithms are noted for requiring absolutely no parameter estimations during the training phase.

---

# Part 2: Answer Key & Explanations

**Question 1**
- [ ] It is a parametric method that learns a fixed set of weights during the training phase.
  *Incorrect.* Slide 1 explicitly states KNN is a "non-parametric method".
- [ ] It requires an extensive computationally expensive training phase before predictions can be made.
  *Incorrect.* Slide 5 lists "no training phase required" as one of the primary advantages of KNN.
- [x] It is capable of handling complex decision boundaries due to its non-parametric nature.
  *Correct.* Slide 5 lists "non-parametric nature allows handling complex decision boundaries" as an advantage.
- [x] It relies on the assumption that similar data points should have similar labels.
  *Correct.* Slide 1 states the core idea of KNN is: "similar data points should have similar labels".

**Question 2**
- [ ] Setting $k=1$ typically leads to an overly smooth decision boundary, indicating underfitting.
  *Incorrect.* Slide 3 visually shows $k=1$ creates highly jagged, complex boundaries, and Slide 4 states "a small k value can lead to overfitting", not underfitting.
- [x] A very large $k$ value can lead to underfitting the training data.
  *Correct.* Slide 4 explicitly states "a large k value can lead to underfitting".
- [x] Setting $k=1$ usually creates highly complex and fragmented decision boundaries, which is a symptom of overfitting.
  *Correct.* This matches the visual evidence on Slide 3 (fragmented boundaries for $k=1$) and the text on Slide 4 (small $k$ leads to overfitting).
- [x] Cross-validation is recommended to find a balance between overfitting and underfitting when choosing $k$.
  *Correct.* Slide 4 states "a good value for k can be found using cross-validation".

**Question 3**
- [x] It is computationally expensive for large datasets.
  *Correct.* Slide 5 explicitly lists this under "Limitations".
- [ ] It cannot inherently handle multiclass classification problems.
  *Incorrect.* Slide 5 lists "can handle multiclass classification problems" under Advantages.
- [x] It is highly sensitive to noisy data.
  *Correct.* Slide 5 lists "sensitive to irrelevant features and noisy data" under Limitations.
- [x] It is highly sensitive to irrelevant features.
  *Correct.* Slide 5 explicitly states it is sensitive to irrelevant features.

**Question 4**
- [ ] The weight $w_i$ assigned to a neighbor is directly proportional to the Euclidean distance between $x$ and $x^{(i)}$.
  *Incorrect.* The weight is *inversely* proportional to the distance, as represented by the fraction $\frac{1}{||x - x^{(i)}||}$ (Slide 6).
- [x] The weight $w_i$ is calculated as $\frac{1}{||x - x^{(i)}||}$.
  *Correct.* This is the exact formula provided for $w_i$ on Slide 6.
- [ ] As a training point $x^{(i)}$ gets infinitely close to the query point $x$, its weight $w_i$ approaches zero.
  *Incorrect.* As the distance approaches zero, the denominator becomes tiny, meaning the weight $w_i$ approaches infinity, not zero.
- [x] The final weighted prediction $y$ requires dividing the sum of the weighted neighbor labels by the sum of all weights $w_i$.
  *Correct.* Slide 6 provides the formula $y = \frac{1}{\sum_{i \in S} w_i} \cdot \sum_{i \in S} w_i \cdot y^{(i)}$, which precisely describes dividing the weighted sum by the sum of the weights.

**Question 5**
- [x] Linear models with a Gaussian kernel
  *Correct.* Slide 8 (KNN - Conclusion) states KNN is "very similar to using linear models with Gaussian kernel".
- [ ] Probabilistic models with conditionally independent features
  *Incorrect.* This describes Naive Bayes, not KNN.
- [ ] Decision trees with Laplacian smoothing
  *Incorrect.* Neither decision trees nor Laplacian smoothing are compared to KNN in the conclusion.
- [ ] Linear regression with highly parameterized polynomials
  *Incorrect.* This is not mentioned in the slides.

**Question 6**
- [x] KNN is inherently sensitive to irrelevant features and noisy data.
  *Correct.* Slide 5 lists this explicitly under "Limitations".
- [x] The choice of the hyperparameter $k$ is an important consideration when handling noisy data.
  *Correct.* Slide 8 notes that "the choice of k and the handling of noisy data are important considerations".
- [x] Performance on noisy data can be improved using dimensionality reduction methods.
  *Correct.* Slide 8 explicitly states performance with noisy data/irrelevant features can be improved using dimensionality reduction methods.
- [x] Large-margin nearest neighbor methods can be utilized to improve performance when irrelevant features are present.
  *Correct.* Slide 8 explicitly names "large-margin nearest neighbor methods" as an example of a dimensionality reduction method to improve performance.

**Question 7**
- [ ] The algorithm assumes that the features (words in an email) are marginally independent of one another regardless of the class label.
  *Incorrect.* Slide 15 says we assume words are *conditionally* independent (meaning independent *given* the class), not unconditionally/marginally independent.
- [ ] Because it is a non-parametric model, Naive Bayes requires no parameter estimation.
  *Incorrect.* Slide 22 states Naive Bayes is a model where "only a very few parameters need to be estimated". Furthermore, KNN is the non-parametric one (Slide 1).
- [ ] To compute the final probability of an email being spam, you must strictly calculate the exact denominator value $p(w_1, w_2, ..., w_d)$.
  *Incorrect.* Slide 16 explicitly demonstrates that we drop the denominator $p(w)$ and only need to *compare* the proportional numerators ($\sim$) to decide the class.
- [ ] The algorithm always performs poorly in practice because its core assumption of conditional independence is almost never true in reality.
  *Incorrect.* Slide 22 points out that "despite its simplicity (and often incorrect assumptions), it often shows surprisingly good results in practice."

**Question 8**
- [x] $p(spam) \cdot p(w_1|spam) \cdot p(w_2|spam) \cdot ... \cdot p(w_d|spam)$
  *Correct.* Slide 16 shows the proportional formula: $p(spam|w_1, w_2, ...,w_d) \sim p(spam) \cdot p(w_1|spam) \cdot p(w_2|spam) \cdot ... \cdot p(w_d|spam)$.
- [ ] $p(spam|w_1) \cdot p(spam|w_2) \cdot ... \cdot p(spam|w_d)$
  *Incorrect.* The probabilities being multiplied are the likelihoods of the features given the class $p(w_i|spam)$, not the probability of the class given a single feature.
- [ ] $\frac{p(spam) \cdot p(w_1, w_2, ..., w_d | spam)}{p(spam)}$
  *Incorrect.* Dividing by $p(spam)$ is mathematically incorrect according to Bayes theorem (Slide 14), where the denominator should be the probability of the evidence $p(w)$.
- [ ] $\frac{p(w_1|spam) \cdot p(w_2|spam) \cdot ... \cdot p(w_d|spam)}{p(w_1, w_2, ..., w_d)}$
  *Incorrect.* This omits the prior probability $p(spam)$ in the numerator, which is required by Bayes theorem (Slide 14).

**Question 9**
- [ ] The model successfully predicts the email is `ham` because $\frac{4}{8}$ is higher than the equivalent spam probability.
  *Incorrect.* Because the term $p(win|ham)$ is 0, the entire multiplication evaluates to 0, destroying the other calculated probabilities.
- [x] The calculated proportional probability for `ham` becomes exactly $0$.
  *Correct.* As shown on Slide 19, $p(ham) \rightarrow 3/4 \cdot 4/8 \cdot 4/8 \cdot 4/8 \cdot 0/8 = 0$.
- [x] The calculation demonstrates the "zero-probability problem" because the word `win` never appeared in a ham email in the training data.
  *Correct.* Slide 20 explains "Probabilities of zero can be very problematic. This often happens when words are not present in the training data."
- [ ] The probability calculation for `spam` also automatically evaluates to $0$ to balance the equation.
  *Incorrect.* Slide 19 shows the spam probability still successfully evaluates to $0.003$ because all words had non-zero probabilities in the spam training set.

**Question 10**
- [x] $p(spam) = \frac{1}{4}$
  *Correct.* Slide 11 calculates $p(spam) = 1/4$ based on 1 out of 4 total training emails being spam.
- [x] $p(ham) = \frac{3}{4}$
  *Correct.* Slide 11 calculates $p(ham) = 3/4$ based on 3 out of 4 total training emails being ham.
- [ ] $p(spam) = \frac{1}{3}$
  *Incorrect.* The denominator is the total number of documents (4), not the number of ham documents (3).
- [ ] The prior probabilities are ignored in the final Naive Bayes proportional comparison.
  *Incorrect.* Slide 16 shows the priors ($p(spam)$ and $p(ham)$) are explicitly multiplied in the final comparison formula.

**Question 11**
- [ ] $\frac{1}{3}$
  *Incorrect.* This ignores both parts of the Laplacian smoothing addition (+1 to numerator, + unique words to denominator).
- [x] $\frac{2}{9}$
  *Correct.* Evaluating the formula yields $\frac{1+1}{3+6} = \frac{2}{9}$. 
- [x] $\frac{1 + 1}{3 + 6}$
  *Correct.* This exactly maps the values to the Laplacian smoothing formula on Slide 20: (frequency + 1) / (words in spam + total unique words). This is mathematically identical to Slide 21's example.
- [ ] $\frac{1}{6}$
  *Incorrect.* This represents an incorrect substitution into the smoothing formula.

**Question 12**
- [ ] Dimensionality reduction
  *Incorrect.* LMNN (for KNN) is dimensionality reduction (Slide 8); Laplacian smoothing is not.
- [ ] Cross-validation
  *Incorrect.* Cross-validation is used to find $k$ in KNN (Slide 4).
- [x] Regularization
  *Correct.* Slide 22 explicitly states: "Laplacian Smoothing prevents probabilities of zero (= a form of regularization)".
- [ ] Non-parametric optimization
  *Incorrect.* Naive Bayes is not non-parametric (it estimates parameters), and smoothing is specifically regularizing those parameters.

**Question 13**
- [ ] Because it uses a complex kernel trick to project features into a higher dimension.
  *Incorrect.* This describes kernelized models (like SVMs), not Naive Bayes.
- [ ] Because it requires no training phase whatsoever, making small data sizes irrelevant.
  *Incorrect.* Naive Bayes *does* have a training phase (estimating parameters like $p(w_i|class)$). KNN is the algorithm with "no training phase" (Slide 5).
- [x] Because only very few parameters need to be estimated (specifically, the independent conditional probabilities and class priors).
  *Correct.* Slide 22 literally states: "only a very few parameters need to be estimated; therefore, it is a very good method when there are many features and few data points".
- [ ] Because it relies heavily on measuring the Euclidean distance between all features, which is highly robust in small datasets.
  *Incorrect.* Naive Bayes relies on probabilities, not Euclidean distance (which is KNN).

**Question 14**
- [x] Multinomial NB
  *Correct.* Explicitly listed on Slide 22 as a variant.
- [x] Gaussian NB
  *Correct.* Explicitly listed on Slide 22 as a variant.
- [x] Binomial NB
  *Correct.* Explicitly listed on Slide 22 as a variant.
- [ ] Kernelized NB
  *Incorrect.* This is not mentioned anywhere in the provided slides.

**Question 15**
- [ ] Both KNN and Naive Bayes are defined as non-parametric methods.
  *Incorrect.* Only KNN is defined as non-parametric (Slide 1). Naive Bayes estimates parameters (Slide 22).
- [x] KNN makes predictions based on proximity (distance), while Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem.
  *Correct.* This perfectly summarizes Slide 1/8 for KNN (predictions based on distance/proximity) and Slide 9 for NB (probabilistic, based on Bayes' theorem).
- [x] Naive Bayes assumes individual features are conditionally independent, whereas KNN makes no such independence assumption, relying instead on distance metrics like Euclidean or Manhattan.
  *Correct.* Slide 15/22 state NB relies on conditional independence. Slide 2 states KNN relies on distance metrics like Euclidean or Manhattan.
- [ ] Both algorithms are noted for requiring absolutely no parameter estimations during the training phase.
  *Incorrect.* While KNN requires no training phase (Slide 5), Naive Bayes explicitly requires parameter estimation (Slide 22).