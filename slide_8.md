# Machine Learning Practice Exam

### Part 1: Practice Multiple-Choice Questions

**Topic 1: Discriminative vs. Generative Models**

**Question 1 (3 Points)**
Which of the following statements correctly describe generative and discriminative models?
- [ ] Discriminative models attempt to model the joint probability distribution $p(x, y)$.
- [ ] Generative models can be used to synthesize new, artificial data points.
- [ ] For a given classification task, a generative model typically requires fewer parameters to be estimated than a discriminative model.
- [ ] Discriminative models focus on modeling the decision boundary between classes, i.e., $p(y|x)$.

**Topic 2: Quadratic Discriminant Analysis (QDA)**

**Question 2 (4 Points)**
Suppose you are applying Quadratic Discriminant Analysis (QDA) to a dataset with classes $c \in \{-1, +1\}$ where the data points $x \in \mathbb{R}^d$. Which of the following statements are true?
- [ ] QDA assumes that the data points for each class follow a uniform distribution.
- [ ] The covariance matrix $\Sigma$ must be calculated shared across all classes to form the quadratic decision boundary.
- [ ] The Maximum Likelihood Estimate (MLE) for the class mean $\mu$ is simply the arithmetic mean of the data points belonging to that class.
- [ ] The total number of parameters to be estimated in a two-class QDA model scales quadratically with the number of dimensions $d$.

**Question 3 (3 Points)**
When deriving the decision boundary for QDA by setting $p(y = -1 | x) = p(y = +1 | x)$, which of the following mathematical outcomes occur?
- [ ] The normalization constant $\frac{1}{\sqrt{(2\pi)^d|\Sigma|}}$ cancels out entirely because it is independent of the class.
- [ ] The log-likelihood involves the determinant of the covariance matrices $|\Sigma_{-1}|$ and $|\Sigma_{+1}|$.
- [ ] The resulting equation simplifies to the form $x^T\Sigma x + v^Tx + t = 0$.
- [ ] The algorithm maps the data into a higher-dimensional space to find a linear boundary.

**Topic 3: Linear Discriminant Analysis (LDA)**

**Question 4 (3 Points)**
Which of the following are true regarding Linear Discriminant Analysis (LDA) compared to QDA?
- [ ] LDA assumes that $\Sigma = \Sigma_{-1} = \Sigma_{+1}$.
- [ ] LDA is a discriminative model, whereas QDA is a generative model.
- [ ] Because the covariance matrices are assumed to be equal in LDA, the quadratic terms $x^T\Sigma^{-1}x$ cancel out, resulting in a linear decision boundary.
- [ ] The linear decision boundary found by LDA always coincides exactly with the boundary found by hard-margin Support Vector Machines.

**Topic 4: Unsupervised Learning & k-means Objective**

**Question 5 (3 Points)**
Which of the following are true about the $k$-means clustering algorithm and its objective function?
- [ ] The goal is to find a discrete partition of the dataset that maximizes the within-cluster distances.
- [ ] The standard $k$-means objective function is a non-convex optimization problem.
- [ ] Finding the global optimal solution for the $k$-means objective is NP-hard.
- [ ] The algorithm always constructs clusters that have non-linear, highly complex boundaries.

**Topic 5: Lloyd's Algorithm and k-means variants**

**Question 6 (4 Points)**
Regarding Lloyd's algorithm (the standard $k$-means algorithm heuristic), which of the following statements are correct?
- [ ] The algorithm is guaranteed to terminate after a finite number of iterations.
- [ ] In every iteration of the while loop, the objective function either decreases or stays the same.
- [ ] The algorithm is guaranteed to find the global optimum if allowed to run for enough iterations.
- [ ] The algorithm's final output heavily depends on the initial placement of the cluster centers.

**Question 7 (3 Points)**
Consider the properties of $k$-means clustering and its variants. Which of the following are true?
- [ ] The partition induced by the $k$-means objective corresponds to a Voronoi partition of the space, meaning all cluster boundaries are linear.
- [ ] $k$-means is highly robust to outliers because it uses the squared Euclidean distance $||x^{(i)} - \mu_j||^2$.
- [ ] $k$-median replaces the squared distance with standard Euclidean distance $||.||_2$, making it more robust against outliers.
- [ ] The `kmeans++` initialization method selects all $k$ initial centers completely uniformly at random from the dataset to ensure unbiasedness.

**Topic 6: Learning Paradigms & Generative vs. Discriminative Characteristics**

**Question 8 (3 Points)**
Based on the introduction to supervised and unsupervised learning, which of the following statements are correct?
- [ ] Semi-supervised learning refers to a scenario where we have an unlabeled dataset but the goal is to predict $p(y|x)$.
- [ ] Dimensionality reduction and density estimation are both forms of unsupervised learning.
- [ ] Attaching labels to data points is generally considered an inexpensive and trivial task in real-world scenarios.
- [ ] Generative models usually require much more data to estimate accurately compared to discriminative models because they have a higher number of parameters.

**Topic 7: Deep Dive into Quadratic Discriminant Analysis (QDA)**

**Question 9 (4 Points)**
When estimating the parameters $\mu$ and $\Sigma$ for QDA using Maximum Likelihood Estimation (MLE), which of the following mathematical statements are true according to the lecture?
- [ ] The algorithm finds the parameters by minimizing the log-likelihood function.
- [ ] The log-likelihood function is concave with respect to $\mu$ and $\Sigma$.
- [ ] Taking the derivative of the log-likelihood with respect to $\mu$ and setting it to 0 yields the equation $\sum_i \Sigma^{-1}(x^{(i)} - \mu) = 0$.
- [ ] The estimated joint distribution for QDA is modeled as $p(x, y) = p(y|x) \cdot p(x)$.

**Question 10 (3 Points)**
Consider the parameter complexity of QDA versus LDA for a $d$-dimensional dataset with $C=2$ classes (classes $\{-1, +1\}$).
- [ ] In QDA, the number of parameters to be estimated is exactly $\frac{d(d+1)}{2} + 2d$.
- [ ] LDA requires estimating fewer parameters than QDA because LDA assumes $\Sigma = \Sigma_{-1} = \Sigma_{+1}$.
- [ ] If you have very limited training data, QDA is generally preferred over LDA because its quadratic boundary prevents overfitting.
- [ ] ChatGPT and DALL-E are provided in the slides as examples of discriminative models capable of generating synthetic data.

**Topic 8: K-means Edge Cases and Computational Complexity**

**Question 11 (4 Points)**
Which of the following statements about the computational complexity and optimality of $k$-means are true?
- [ ] If we fix the dimension $d$ to a constant (e.g., $d=2$), finding the globally optimal solution to the $k$-means problem becomes solvable in polynomial time.
- [ ] Optimizing the $k$-means objective is known to have a polynomial *smoothed complexity*, which helps explain why it usually works well in practice.
- [ ] By arranging 4 data points on a real 1D line with specific spacings ($a, b, c$), it is possible to construct a scenario where Lloyd's algorithm converges to a local optimum that is arbitrarily worse than the global optimum.
- [ ] The `kmeans++` initialization algorithm guarantees that the final objective value will be exactly the global minimum.

**Topic 9: K-means Equivalence and Objective Functions**

**Question 12 (3 Points)**
Which of the following optimization problems are mathematically equivalent to the standard $k$-means objective?
- [ ] Finding cluster centers $\mu_j$ such that the sum of the squared Euclidean distances from data points to their assigned centers is minimized.
- [ ] Finding a discrete partition of the dataset such that the sum of squared distances *between all pairs of points within the same cluster* is minimized.
- [ ] Maximizing the distance between the cluster centers $\mu_1, ..., \mu_k$.
- [ ] Minimizing the L1 norm ($||.||_1$) of the distances from the data points to the cluster centers.

**Topic 10: Practical Heuristics and K-means Variants**

**Question 13 (3 Points)**
In practice, Lloyd's algorithm can get stuck or produce suboptimal results. Which of the following heuristics/practices were recommended in the slides to improve clustering?
- [ ] Because it is a highly non-convex problem, it is standard practice to run the algorithm many times with different initializations and keep the result with the lowest objective value.
- [ ] If a cluster center "loses" all its data points during an iteration, the algorithm must terminate immediately and return an error.
- [ ] A valid local search heuristic is to split a cluster that has a very bad objective function into two pieces and randomly remove another cluster center.
- [ ] A valid local search heuristic is to merge two existing clusters and introduce a completely new cluster center elsewhere.

**Question 14 (3 Points)**
Regarding the initialization step of $k$-means, which of the following are true?
- [ ] The most common (but naive) initialization is to uniformly at random choose arbitrary coordinates in the feature space as starting centers.
- [ ] `kmeans++` selects the first center uniformly at random from the data points.
- [ ] `kmeans++` selects subsequent centers $x$ with a probability inversely proportional to their squared distance from the closest existing center.
- [ ] With careful initialization (like `kmeans++`), one can achieve an expected objective value that is at most a factor of $O(\log k)$ worse than the optimal solution.

**Question 15 (2 Points)**
Which of the following correctly describes a variant of the $k$-means algorithm?
- [ ] $k$-median replaces the squared Euclidean distance $||.||_2^2$ with the standard Euclidean distance $||.||_2$, making it more robust against outliers.
- [ ] Soft $k$-means removes hard assignments, assigning probabilities of belonging to clusters, and is mathematically related to Mixture of Gaussians.
- [ ] Weighted $k$-means introduces weights for individual features (dimensions) to perform implicit dimensionality reduction.

---

### Part 2: Answer Key and Explanations

**Question 1: 2nd and 4th options are correct.**
*   *Option 1 (False):* Discriminative models model $p(y|x)$. Generative models model $p(x, y)$ or $p(x|y)p(y)$.
*   *Option 2 (True):* Generative models model the distribution of the data $p(x, y)$, so you can sample from it to generate new data (Slide 328).
*   *Option 3 (False):* Generative models usually have a *much higher* number of parameters to estimate than discriminative models (Slide 341).
*   *Option 4 (True):* This is the defining characteristic of discriminative models (Slide 327).

**Question 2: 3rd and 4th options are correct.**
*   *Option 1 (False):* QDA assumes a Gaussian (Normal) distribution, not uniform (Slide 329).
*   *Option 2 (False):* QDA assumes *distinct* covariance matrices $\Sigma_c$ for each class. Shared covariance is LDA.
*   *Option 3 (True):* The MLE derivation (Slide 334) shows $\mu = \frac{1}{n}\sum x^{(i)}$.
*   *Option 4 (True):* The number of parameters is $2\frac{d(d+1)}{2} + 2d$, which is dominated by the $d^2$ term (Slide 341).

**Question 3: 2nd and 3rd options are correct.**
*   *Option 1 (False):* Because the covariance matrices $\Sigma_c$ are different per class, the $|\Sigma|$ term does *not* cancel out (Slide 339).
*   *Option 2 (True):* As shown on Slide 339, $\log(|\Sigma_{-1}|)$ and $\log(|\Sigma_{+1}|)$ remain in the equation.
*   *Option 3 (True):* The equation simplifies to this quadratic form, hence the name QDA (Slide 340).
*   *Option 4 (False):* QDA stays in the original feature space and finds a quadratic boundary. Support Vector Machines use the kernel trick to map to higher dimensions to find linear boundaries.

**Question 4: 1st and 3rd options are correct.**
*   *Option 1 (True):* This is the fundamental assumption that separates LDA from QDA (Slide 342).
*   *Option 2 (False):* Both QDA and LDA are generative models (they model distributions). However, LDA's solution *coincides* with a simple discriminative model (linear regression) (Slide 343).
*   *Option 3 (True):* Because $\Sigma$ is shared, the $x^T\Sigma^{-1}x$ terms on both sides of the equation cancel out, leaving a linear boundary (Slide 342).
*   *Option 4 (False):* LDA is related to linear regression ($\min ||Xw-y||^2$) (Slide 343), not Hard-Margin SVMs (which maximize margin based on support vectors).

**Question 5: 2nd and 3rd options are correct.**
*   *Option 1 (False):* The goal is to *minimize* within-cluster distances, not maximize them (Slide 347/359).
*   *Option 2 (True):* It is a highly non-convex optimization problem (Slide 347/366).
*   *Option 3 (True):* Finding the global optimum is NP-hard (Slide 363).
*   *Option 4 (False):* $k$-means always constructs *convex* clusters with *linear* boundaries (Voronoi partitions) (Slide 361).

**Question 6: 1st, 2nd, and 4th options are correct.**
*   *Option 1 (True):* The algorithm is guaranteed to terminate because there are finitely many partitions (Slide 352).
*   *Option 2 (True):* In each iteration, the objective function decreases (or stays exactly the same, causing termination) (Slide 352).
*   *Option 3 (False):* It is highly susceptible to local minima and can be an arbitrary factor away from the global solution (Slide 353).
*   *Option 4 (True):* Because it gets stuck in local optima, initialization is crucial, and practice dictates running it multiple times with different initializations (Slide 357).

**Question 7: 1st and 3rd options are correct.**
*   *Option 1 (True):* $k$-means induces a Voronoi partition, meaning boundaries are linear (Slide 360/361).
*   *Option 2 (False):* Because it uses *squared* distance, it is highly sensitive (not robust) to outliers.
*   *Option 3 (True):* Replacing the squared distance with $L2$ norm ($k$-median) makes it more robust to outliers (Slide 365).
*   *Option 4 (False):* `kmeans++` picks the *first* point at random, but subsequent points are chosen with probability proportional to their squared distance from existing centers, not uniformly (Slide 355).

**Question 8: 2nd and 4th options are correct.**
*   *Option 1 (False):* Semi-supervised learning uses a dataset where *some* data points have labels, not a completely unlabeled dataset (Slide 344). 
*   *Option 2 (True):* Both are explicitly listed as examples of Unsupervised Learning (Slide 345).
*   *Option 3 (False):* The slides explicitly note that attaching labels to data points can be an "expensive task" (Slide 344).
*   *Option 4 (True):* Generative models have much higher parameter counts, hence they usually need much more data (Slide 341 & 343).

**Question 9: 2nd and 3rd options are correct.**
*   *Option 1 (False):* MLE aims to *maximize* the log-likelihood, not minimize it (Slide 331-332: argmax).
*   *Option 2 (True):* The log-likelihood function is concave in $\mu$ and $\Sigma$, allowing us to find the maximum by setting the derivative to 0 (Slide 333).
*   *Option 3 (True):* This is the exact derivative derived in the slides $\frac{\partial}{\partial \mu} = \sum \Sigma^{-1}(x^{(i)} - \mu) \overset{!}{=} 0$ (Slide 333).
*   *Option 4 (False):* While mathematically equivalent via Bayes theorem, generative models explicitly model the joint distribution as $p(x, y) = p(x|y) \cdot p(y)$, not $p(y|x) \cdot p(x)$ (Slide 336).

**Question 10: 2nd option is the only correct answer.**
*   *Option 1 (False):* This is the parameter count for LDA. QDA estimates a different covariance matrix for *each* class, so the count is $2 \frac{d(d+1)}{2} + 2d$ for two classes (Slide 341).
*   *Option 2 (True):* LDA assumes shared covariance, meaning it only needs to estimate one $\Sigma$ matrix instead of two, significantly reducing parameters (Slide 342).
*   *Option 3 (False):* Because QDA has more parameters (a full covariance matrix per class), it is *more* prone to overfitting on limited data than LDA.
*   *Option 4 (False):* ChatGPT and DALL-E are generative models (they generate data), not discriminative (Slide 343).

**Question 11: 2nd and 3rd options are correct.**
*   *Option 1 (False):* The problem is NP-hard *both* if $k$ is fixed or variable, and *both* if the dimension is fixed or variable (Slide 363).
*   *Option 2 (True):* Optimizing it has a polynomial smoothed complexity, which acts as an indicator for why it works in practice (Slide 364).
*   *Option 3 (True):* The slides show a specific 1D example with 4 points where, based on $a, b, c$, the local optimum can be an arbitrary factor $(a/c)^2$ worse than the global optimum (Slide 353-354).
*   *Option 4 (False):* `kmeans++` guarantees a constant-factor approximation (expected value at most $O(\log k)$ worse), but it does *not* guarantee finding the exact global minimum (Slide 364).

**Question 12: 1st and 2nd options are correct.**
*   *Option 1 (True):* This is the standard formulation of the $k$-means optimization problem (Slide 347/359).
*   *Option 2 (True):* The slides explicitly state this is a mathematically equivalent optimization problem: minimizing within-cluster distances between all pairs of points (Slide 359).
*   *Option 3 (False):* The objective function does not include a term for maximizing the distance between the centers themselves.
*   *Option 4 (False):* $k$-means minimizes the *squared* L2 norm ($||.||_2^2$), not the L1 norm. Replacing it with the L2 norm creates the $k$-median variant (Slide 365).

**Question 13: 1st, 3rd, and 4th options are correct.**
*   *Option 1 (True):* Restarting the algorithm many times and taking the best result is the standard procedure for this highly non-convex problem (Slide 357).
*   *Option 2 (False):* If a center loses all points, practice dictates either restarting the algorithm or randomly replacing the empty center with an existing data point (Slide 357).
*   *Option 3 (True):* This is explicitly listed as a local search heuristic to improve the result after termination (Slide 358).
*   *Option 4 (True):* Merging clusters and introducing a new center is another valid local search heuristic (Slide 358).

**Question 14: 2nd and 4th options are correct.**
*   *Option 1 (False):* The most common initialization is to randomly choose *some data points* as starting centers, not arbitrary coordinates (Slide 355).
*   *Option 2 (True):* Step 2 of the `kmeans++` heuristic is to pick $x$ uniformly at random from the data points to be the first center (Slide 355).
*   *Option 3 (False):* Subsequent points are chosen with probability *proportional* to $d(x)$ (which is $||x-c||^2$), not *inversely* proportional. You want to pick points far away! (Slide 355).
*   *Option 4 (True):* This is the mathematical guarantee provided by `kmeans++` (Slide 364).

**Question 15: 1st and 2nd options are correct.**
*   *Option 1 (True):* $k$-median uses the standard $L2$ norm instead of squared, making it more robust against outliers (Slide 365).
*   *Option 2 (True):* Soft $k$-means uses "probabilities" instead of hard assignments and is related to Mixtures of Gaussians (Slide 365).
*   *Option 3 (False):* Weighted $k$-means introduces weights for the *individual data points*, not for features/dimensions (Slide 365).