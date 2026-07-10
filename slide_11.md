# Master’s Level Exam: Recommender Systems and Matrix Factorization
**Course:** Machine Learning & Data Mining
**Examiner:** Prof. [Expert AI Identity]
**Duration:** 120 Minutes
**Format:** 0, 1, or Many (Checkboxes). Any number of options may be correct.

---

## Part 1: The Test

### Topic 1: Mathematical Foundations and Rating Prediction (Slide 436-442)

**Question 1 [2 Points]:** Given a set of $m$ movies and $n$ users with a latent feature dimension $k$, which of the following statements regarding the matrix factorization $M = UV^\top$ are mathematically correct?
- [ ] $U$ is an $n \times k$ matrix representing user affinities.
- [ ] The predicted rating $m_{i,j}$ is calculated as the Euclidean distance between vectors $u^{(i)}$ and $v^{(j)}$.
- [ ] $V^\top$ is a $k \times n$ matrix where each column $j$ represents the latent profile of user $j$.
- [ ] The rank of the resulting matrix $M$ is at most $\min(m, n, k)$.

**Question 2 [3 Points]:** In the optimization objective $\min_{U,V} \sum_{(i,j) \in \Omega} (r_{i,j} - (u^{(i)})^\top v^{(j)})^2$, the set $\Omega$ represents:
- [ ] The set of all possible pairs $(i,j)$ in the $m \times n$ matrix.
- [ ] Only the indices $(i,j)$ for which the ground truth rating $r_{i,j}$ is known/observed.
- [ ] The set of latent features that contribute most to the variance (Principal Components).
- [ ] A regularization term used to prevent the weights from growing too large.

**Question 3 [2 Points]:** Consider the Frobenius norm notation $\|R - UV^\top\|^2_{\Omega}$. Which of the following is a correct property of this formulation?
- [ ] It is equivalent to $\|R - UV^\top\|^2_{Fro}$ if all entries in $R$ are observed.
- [ ] It ignores all entries $r_{i,j}$ where $(i,j) \notin \Omega$.
- [ ] The objective function is jointly convex in $U$ and $V$.
- [ ] The objective function is smooth, allowing for the use of gradient-based methods.

### Topic 2: Optimization and Algorithms (Slide 443-444, 452)

**Question 4 [4 Points]:** Regarding the complexity and solvability of the Matrix Factorization problem for Recommender Systems:
- [ ] Finding the global minimum of $\|R - UV^\top\|^2_{\Omega}$ is generally NP-hard when $\Omega$ is a proper subset of all indices.
- [ ] Standard Gradient Descent is guaranteed to find the global optimum because the problem is smooth.
- [ ] If $\Omega$ contains all indices (no missing values), the global optimum can be found efficiently via Singular Value Decomposition (SVD).
- [ ] The problem is non-convex, meaning the algorithm may get stuck in local minima or saddle points.

**Question 5 [3 Points]:** In the Alternating Least Squares (ALS) approach:
- [ ] Both $U$ and $V$ are updated simultaneously in a single gradient step.
- [ ] When $U$ is fixed, the problem of solving for $V$ decomposes into $n$ independent linear least squares problems.
- [ ] ALS is a specific instance of the block-coordinate descent algorithm.
- [ ] Solving for a fixed $U$ requires a non-linear solver because the objective is non-convex in $V$.

**Question 6 [2 Points]:** Which of the following are valid optimization strategies for Non-negative Matrix Factorization (NMF)?
- [ ] Projected Gradient Descent, where negative values are clipped to zero after each step.
- [ ] Standard SVD followed by taking the absolute value of all entries in the factor matrices.
- [ ] Block-coordinate descent leading to non-negative least squares sub-problems.
- [ ] Lagrangian Relaxation to handle the $U, V \geq 0$ constraints.

### Topic 3: Variants and Regularization (Slide 447, 455)

**Question 7 [3 Points]:** Why is the regularized objective $\min \|R - UV^\top\|^2_{\Omega} + \lambda(\|U\|^2_{Fro} + \|V\|^2_{Fro})$ often preferred over the standard version?
- [ ] It ensures the problem becomes globally convex.
- [ ] It prevents the latent factors from overfitting to the sparse observed ratings in $\Omega$.
- [ ] It encourages the latent factors to have small magnitudes, improving generalization.
- [ ] It reduces the computational complexity from NP-hard to Polynomial time.

**Question 8 [3 Points]:** Comparing the Frobenius norm objective to the Kullback-Leibler (KL) divergence objective in NMF:
- [ ] KL divergence is strictly used for binary matrices, while Frobenius is for continuous ratings.
- [ ] KL divergence $KL(W \| UV^\top)$ is particularly useful when treating the rows of the matrices as probability distributions.
- [ ] The Frobenius norm assumes Gaussian noise, whereas KL divergence relates better to Poisson distributed counts.
- [ ] Both objectives are convex when optimized over $U$ and $V$ simultaneously.

### Topic 4: NLP and Topic Modeling (Slide 450-454)

**Question 9 [3 Points]:** In a Document-Term matrix $W$, where $w_{i,j}$ represents the count of term $j$ in document $i$:
- [ ] NMF is used to ensure that topic contributions and word-topic memberships are physically interpretable (non-negative).
- [ ] Matrix $U$ can be interpreted as the document-topic distribution.
- [ ] Matrix $V$ (or $V^\top$) can be interpreted as the word-topic distribution.
- [ ] A high value in $U_{ik}$ indicates document $i$ has a strong affinity for topic $k$.

**Question 10 [4 Points]:** Consider the TF-IDF score calculation $TF \cdot IDF$. Which of the following statements are correct?
- [ ] The Term Frequency (TF) normalizes for document length.
- [ ] The Inverse Document Frequency (IDF) is high for common "stop words" like "the" or "and".
- [ ] The smoothed IDF formula $\log(\frac{1+m}{1 + \text{df}_j})$ prevents division by zero if a word appears in no documents.
- [ ] TF-IDF essentially weights words that are rare in the corpus but frequent in a specific document higher.

**Question 11 [2 Points]:** During the preprocessing phase for topic modeling, why are very frequent and very rare words often discarded?
- [ ] Frequent words (stop words) lack discriminative power across different topics.
- [ ] Rare words significantly increase the dimensionality $n$ without providing enough statistical evidence for topic clusters.
- [ ] Discarding these words makes the matrix $W$ dense, which is required for NMF.
- [ ] It helps to focus the factorization on terms that characterize the underlying themes.

### Topic 5: Clustering and Latent Features (Slide 446, 459-462)

**Question 12 [3 Points]:** Matrix Factorization is mathematically linked to K-means clustering. Which of the following are true?
- [ ] K-means can be viewed as $X^\top \approx UZ$, where $U$ contains cluster centers and $Z$ is a cluster indicator matrix.
- [ ] In "hard" K-means MF, the constraint on $Z$ is $z_{i,j} \in \{0, 1\}$ and $\sum_i z_{i,j} = 1$.
- [ ] Lloyd’s algorithm is essentially a block-coordinate descent approach on the K-means MF formulation.
- [ ] Soft clustering allows $z_{i,j} \in [0, 1]$, representing the fraction of assignment to a cluster.

**Question 13 [3 Points]:** In word embeddings (Topic Modeling Results), the relationship "Queen + Man - Woman $\approx$ King" implies:
- [ ] Latent space arithmetic can capture semantic analogies.
- [ ] The latent vectors for "Man" and "Woman" must be orthogonal.
- [ ] The difference vector (Man - Woman) represents a "gender" direction in the latent space.
- [ ] Similar words are grouped together in the latent space because they share similar context (co-occurrence).

### Topic 6: General Properties and Complexity (Slide 445, 448, 463)

**Question 14 [2 Points]:** Regarding the training time and scalability mentioned in the slides:
- [ ] Matrix factorization on the Netflix dataset (100M ratings) can take weeks to train on a standard machine.
- [ ] Matrix factorization is considered an "easy to use and fast to compute" method for many applications.
- [ ] Adding temporal features to the Netflix model decreased performance by 2% due to overfitting.
- [ ] Matrix factorization is a universal method that can be used for voice separation and image embedding.

**Question 15 [2 Points]:** If we have $M = UV^\top Q$ instead of $M = UV^\top$ (where $Q$ is another factor matrix):
- [ ] The model becomes significantly more informative because it has more parameters.
- [ ] It effectively collapses back to a two-matrix product $U(V^\top Q)$, meaning the rank and representational power remain the same as a standard MF with rank $k$.
- [ ] This is known as a Three-Way Factorization and is always superior for data imputation.
- [ ] The number of latent features $k$ is still the primary bottleneck for information capacity.

---

## Part 2: Answer Key and Explanations

### Topic 1: Mathematical Foundations and Rating Prediction
**Question 1:**
- [ ] **Incorrect:** $U$ is an $m \times k$ matrix (movies/objects $\times$ features) according to slide 441.
- [ ] **Incorrect:** The predicted rating is the dot product $u^\top v$, not Euclidean distance (Slide 439).
- [ ] **Correct:** $V \in \mathbb{R}^{n \times k}$, so $V^\top$ is $k \times n$, where columns correspond to the $n$ users (Slide 441).
- [ ] **Correct:** The rank of a product $AB$ is at most the minimum of the ranks of $A$ and $B$; here $U$ and $V^\top$ have a shared dimension $k$.

**Question 2:**
- [ ] **Incorrect:** This would describe a dense matrix reconstruction, not a recommender system with missing data.
- [ ] **Correct:** Slide 442 explicitly defines $\Omega$ as the set of indices for which we have input ratings.
- [ ] **Incorrect:** Latent features are the output of the process, not the definition of the index set $\Omega$.
- [ ] **Incorrect:** This is a description of regularization, not the index set $\Omega$.

**Question 3:**
- [ ] **Correct:** If all entries are observed, the mask $\Omega$ covers the entire matrix, matching the standard Frobenius norm.
- [ ] **Correct:** The definition $\|A\|^2_{\Omega} = \sum_{(i,j) \in \Omega} a_{i,j}^2$ mathematically ignores entries not in the set (Slide 442).
- [ ] **Incorrect:** While convex in $U$ or $V$ individually, the product $UV^\top$ makes it non-convex jointly.
- [ ] **Correct:** It is a sum of squared differences, which is a smooth polynomial function (Slide 443).

### Topic 2: Optimization and Algorithms
**Question 4:**
- [ ] **Correct:** Slide 443 states the problem is in general NP-hard.
- [ ] **Incorrect:** Gradient descent only guarantees a local minimum for non-convex problems (Slide 443).
- [ ] **Correct:** Slide 448 notes that if the full Frobenius norm is used (all entries known), SVD provides the optimal low-rank approximation.
- [ ] **Correct:** Non-convexity is a fundamental property of the $UV^\top$ formulation (Slide 443).

**Question 5:**
- [ ] **Incorrect:** ALS alternates between updating $U$ and $V$, it does not update them simultaneously.
- [ ] **Correct:** Fixing one factor turns the objective into a standard linear least squares problem for the other, which can be solved independently for each row/column (Slide 444).
- [ ] **Correct:** Slide 444 explicitly identifies ALS as a block-coordinate descent approach.
- [ ] **Incorrect:** Once $U$ is fixed, the objective is a quadratic (convex) function of $V$.

**Question 6:**
- [ ] **Correct:** Projected gradient descent is a standard method for bound constraints like $U, V \geq 0$ (Slide 452).
- [ ] **Incorrect:** SVD factors can be negative; simply taking the absolute value does not minimize the reconstruction error under non-negative constraints.
- [ ] **Correct:** This leads to "non-negative least squares" sub-problems, as mentioned on Slide 452.
- [ ] **Correct:** General optimization knowledge; Lagrangian multipliers are a standard way to handle inequality constraints.

### Topic 3: Variants and Regularization
**Question 7:**
- [ ] **Incorrect:** Regularization does not change a non-convex function into a convex one.
- [ ] **Correct:** Recommender systems are highly sparse; regularization is necessary to prevent the model from capturing noise in the few observed ratings.
- [ ] **Correct:** Frobenius norm regularization penalizes large weights, which is a standard technique for better generalization (Slide 447).
- [ ] **Incorrect:** Regularization does not change the fundamental NP-hardness of the matrix completion problem.

**Question 8:**
- [ ] **Incorrect:** Both can be used for continuous or count data, though they have different statistical assumptions.
- [ ] **Correct:** KL divergence is a measure between probability distributions, making it suitable when rows/columns are normalized (Slide 455).
- [ ] **Correct:** Frobenius norm minimizes Gaussian error, while KL divergence is the loss function derived from the Likelihood of Poisson-distributed data.
- [ ] **Incorrect:** Both are non-convex due to the $UV^\top$ term.

### Topic 4: NLP and Topic Modeling
**Question 9:**
- [ ] **Correct:** Non-negativity ensures that topics and word counts are additive and interpretable (Slide 451).
- [ ] **Correct:** $U \in \mathbb{R}^{m \times k}$ maps $m$ documents to $k$ topics (Slide 451).
- [ ] **Correct:** $V \in \mathbb{R}^{n \times k}$ maps $n$ words to $k$ topics (Slide 451).
- [ ] **Correct:** $U_{ik}$ represents the "amount" of topic $k$ in document $i$.

**Question 10:**
- [ ] **Correct:** TF = (count / length), which accounts for document size variations (Slide 454).
- [ ] **Incorrect:** Stop words appear in almost all documents ($df_j \approx m$), making $m/df_j \approx 1$ and $\log(1) = 0$, so IDF is low.
- [ ] **Correct:** Smoothing (adding 1) ensures the argument of the log is never undefined or zero (Slide 454).
- [ ] **Correct:** This is the core intuition of TF-IDF—identifying words that are characteristic of a specific document relative to the whole corpus.

**Question 11:**
- [ ] **Correct:** Stop words appear everywhere and do not help distinguish one topic from another (Slide 453).
- [ ] **Correct:** Rare words appearing in only one document do not provide enough co-occurrence data to form a "topic" (Slide 453).
- [ ] **Incorrect:** Word removal typically makes the matrix even sparser, not denser.
- [ ] **Correct:** The goal is to focus on words with high "mutual information" with the underlying latent themes.

### Topic 5: Clustering and Latent Features
**Question 12:**
- [ ] **Correct:** Slide 462 shows that $X^\top \approx UZ$ is the matrix form of the K-means objective.
- [ ] **Correct:** The constraints $z \in \{0, 1\}$ and the sum-to-one constraint enforce "hard" assignment to exactly one cluster (Slide 462).
- [ ] **Correct:** Lloyd's algorithm alternates between assigning points (fixing $U$) and updating centers (fixing $Z$), which is block-coordinate descent (Slide 462).
- [ ] **Correct:** This is defined as soft clustering on Slide 462, where $Z$ is allowed to be continuous.

**Question 13:**
- [ ] **Correct:** Slide 459 illustrates that latent space captures semantic analogies through vector addition/subtraction.
- [ ] **Incorrect:** Orthogonality is not required; in fact, "Man" and "Woman" are likely very similar (non-orthogonal) but separated by a specific gender vector.
- [ ] **Correct:** The vector arithmetic works because the "gender" displacement is consistent across different pairs (King/Queen, Man/Woman).
- [ ] **Correct:** This is the fundamental assumption of distributional semantics mentioned on Slide 456.

### Topic 6: General Properties and Complexity
**Question 14:**
- [ ] **Incorrect:** Slide 445 explicitly states the Netflix model needs roughly 10 minutes training time on a standard machine.
- [ ] **Correct:** Slide 460 (Summary) highlights that it is easy to use and fast to compute.
- [ ] **Incorrect:** Adding temporal features improved the Netflix internal rating from 6% to over 8% (Slide 445).
- [ ] **Correct:** Slide 449 and 461 list these as universal applications of MF.

**Question 15:**
- [ ] **Incorrect:** As the note on Slide 463 says, you don't actually get anything "more informative" because the product still results in a matrix of the same rank.
- [ ] **Correct:** $UV^\top Q$ can be rewritten as $U(V^\top Q) = U \tilde{V}^\top$, collapsing back to the standard two-matrix form.
- [ ] **Incorrect:** It is not "always" superior and can often lead to overfitting or redundancy.
- [ ] **Correct:** The bottleneck is the inner dimension $k$, which limits the rank and thus the information capacity (Slide 463).