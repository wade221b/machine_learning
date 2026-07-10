# Part 1: The Test

### Topic 1: Foundations of Dimensionality Reduction and MDS
**Question 1 (2 Points)**
According to the slides on low-dimensional embedding, which of the following statements regarding the distance matrix $D$ are true?
- [ ] If $D$ is a general distance matrix, we can always find a low-dimensional embedding without distorting the data as long as we choose $d \ge n$.
- [ ] For a matrix $D$ to be a "Euclidean distance matrix", it must originate from a point configuration in $\mathbb{R}^d$ such that the embedding perfectly matches $D$.
- [ ] Multi-dimensional scaling (MDS) always explicitly aims to recover the underlying high-dimensional coordinates of the data before projecting to a lower dimension.
- [ ] The primary goal of embedding into Euclidean space is always visualization; applying machine learning algorithms to the embedded data is generally considered an anti-pattern.

**Question 2 (3 Points)**
In the Ekman (1954) color perception example provided in the slides, how was the dissimilarity matrix constructed and structured?
- [ ] 31 individuals provided similarity ratings on a five-point scale (0 to 4) for pairs of 14 colors.
- [ ] The similarity ratings were averaged, scaled, and subtracted from 1 to yield dissimilarities.
- [ ] The resulting $14 \times 14$ dissimilarity matrix was asymmetric due to human cognitive bias in comparing color A to B versus B to A.
- [ ] The colors used in the experiment differed in multiple physical dimensions, specifically hue, saturation, and brightness.

### Topic 2: Classic MDS

**Question 3 (4 Points)**
Classic MDS seeks to minimize the error with respect to pairwise scalar products. Given a distance matrix $D$, the Gram matrix $S$ has entries $(x^{(i)})^\top x^{(j)}$. Based on the derivation in the slides, which of the following mathematical identities correctly represent steps or conclusions in this formulation?
- [ ] $d_{i,j}^2 = (x^{(i)})^\top x^{(i)} + (x^{(j)})^\top x^{(j)} - 2(x^{(i)})^\top x^{(j)}$
- [ ] $(x^{(i)})^\top x^{(j)} = \frac{1}{2}\left(d_{0,x^{(i)}}^2 + d_{0,x^{(j)}}^2 - d_{i,j}^2\right)$
- [ ] Picking the origin freely at the first data point $x^{(1)}$ allows us to express the scalar products solely using entries from the distance matrix $D$.
- [ ] The strain function to be minimized is equivalent to $\min_{X} \sum_{i,j} \left(s_{i,j} - (x^{(i)})^\top x^{(j)}\right)^2$.

**Question 4 (3 Points)**
Consider the optimization problem for Classic MDS: $\min_{X \in \mathbb{R}^{n \times d}} \|S - XX^\top\|_{\text{Fro}}^2$. Which of the following statements accurately describe the properties and solution of this optimization?
- [ ] It is a strictly convex optimization problem, ensuring gradient descent will not get stuck in local minima.
- [ ] Despite the problem's objective landscape, the optimal solution can be computed analytically using eigenvalue decomposition or Singular Value Decomposition (SVD).
- [ ] The coordinate matrix $X$ is recovered by computing $X = V_d \Lambda_d$, where $V_d$ contains the top $d$ eigenvectors and $\Lambda_d$ is the diagonal matrix of the top $d$ eigenvalues.
- [ ] If the data strictly comes from $\mathbb{R}^d$, the Gram matrix $S$ will have exactly $d$ eigenvalues equal to 0.

### Topic 3: Metric and Non-metric MDS

**Question 5 (2 Points)**
Which of the following characteristics strictly apply to Metric MDS, as opposed to Classic MDS?
- [ ] It minimizes a "stress" function defined as $\sum_{i,j} \left(\|x^{(i)} - x^{(j)}\| - d_{i,j}\right)^2$.
- [ ] It guarantees finding the global optimum efficiently via eigenvalue decomposition.
- [ ] The problem can be shown to be NP-hard in general.
- [ ] It minimizes error with respect to the distance function directly, rather than scalar products.

**Question 6 (2 Points)**
According to the slides, Metric MDS can incur errors from which of the following sources?
- [ ] The input distance matrix is not perfectly Euclidean.
- [ ] The use of the $k$-nearest neighbor graph to approximate distances introduces shortest-path distortions.
- [ ] The algorithm returning a local minimum due to the highly non-convex nature of the optimization problem.
- [ ] The conversion of the distance matrix into a Gram matrix using an arbitrary origin point.

**Question 7 (3 Points)**
Which of the following statements are true regarding Non-metric MDS?
- [ ] It is specifically designed for scenarios where explicit distance values are unavailable, but an ordering of distances (e.g., $d_{i,j} < d_{i,k}$) is known.
- [ ] The objective function relies on a monotonic function $f$ to preserve the given orderings of distances.
- [ ] The algorithm typically alternates between updating the point embeddings $x$ and computing the monotonic function $f$.
- [ ] Because it relaxes the strict distance constraints of Metric MDS, the optimization problem for Non-metric MDS becomes convex.

### Topic 4: Isomap and Manifold Learning

**Question 8 (2 Points)**
The Utah teapot dataset is used in the slides to illustrate a key concept about dimensionality. Which of the following statements correctly capture the details of this example?
- [ ] Each image is represented as a high-dimensional vector in $\mathbb{R}^{3 \times 76 \times 101}$.
- [ ] The dataset has an intrinsic dimensionality of 2, representing the rotation of the teapot.
- [ ] Because the ambient dimension is so large, Isomap is theoretically unable to embed this dataset into a two-dimensional circle.
- [ ] The example demonstrates that while data may live in a high-dimensional ambient space, it often resides on a low-dimensional manifold.

**Question 9 (3 Points)**
Isomap replaces global Euclidean distances with geodesic distances. How does the Isomap algorithm conceptually and practically compute these geodesic distances?
- [ ] It assumes that in a small local region, intrinsic manifold distances heavily diverge from Euclidean distances.
- [ ] It keeps local Euclidean distances unchanged and builds a $k$-nearest neighbor graph.
- [ ] It approximates long-range geodesic distances by computing the shortest-path distances walking along the edges of the $k$-nearest neighbor graph.
- [ ] It computes the Gram matrix of the original data and applies a Gaussian kernel before extracting the principal components.

**Question 10 (3 Points)**
What is the exact final step of the Isomap algorithm once the matrix of shortest path distances $D$ has been computed?
- [ ] Run Principal Component Analysis (PCA) on $D$.
- [ ] Run Classic MDS on $D$ to preserve scalar products.
- [ ] Run Metric MDS on $D$ to embed the data into a low-dimensional space while preserving geodesic distances.
- [ ] Run Non-metric MDS on $D$ to preserve the ordering of the geodesic paths.

### Topic 5: Comprehensive Synthesis

**Question 11 (3 Points)**
In the context of the embedding problems summarized in the slides, consider a scenario where you are given a matrix $D$ representing the perceived similarity of brand logos rated by humans, which is highly subjective and only reliable in terms of relative ranking. Which algorithm and corresponding property is the MOST appropriate choice?
- [ ] Classic MDS, because it provides an exact analytical guarantee for subjective human ratings.
- [ ] Non-metric MDS, because humans can often compare two items relatively easily, but it is hard to assign an exact continuous score.
- [ ] Isomap, because the logos reside on a highly non-linear low-dimensional image manifold.
- [ ] Metric MDS, because minimizing the exact distance stress function resolves subjective human error.

**Question 12 (3 Points)**
If we define a data matrix $X$ whose rows are the embedded points $x^{(i)}$, which of the following relationships highlighted in the slides are perfectly true? (Select all that apply, or none if all are false).
- [ ] In Classic MDS, setting $X = V_d \sqrt{\Lambda_d}$ utilizes the $d$ largest eigenvalues, assuming they are sorted in decreasing order.
- [ ] The classic MDS algorithm shares a deep mathematical connection with PCA, particularly in how it handles the spectrum of the similarity matrix to pick a reasonable $d$.
- [ ] The Gram matrix $S$ is strictly defined as $S = XX^\top$.
- [ ] In Classic MDS, if the original data did not come from $\mathbb{R}^d$, the matrix $S$ will have full rank $n$, and we must pick a reasonable $d$ to approximate the data.

**Question 13 (2 Points)**
Which of the algorithms discussed in the slides involve highly non-convex optimization problems? 
- [ ] Classic MDS
- [ ] Metric MDS
- [ ] Non-metric MDS
- [ ] PCA (as referenced in the connection to Classic MDS)

**Question 14 (3 Points)**
Assume you are applying Isomap to the MNIST digits dataset. According to the slides, what justifies the use of manifold learning here?
- [ ] Every vector $p \in \mathbb{R}^{784}$ corresponds to a "proper" digit image, making the ambient space completely dense with valid data.
- [ ] While every valid image $x^{(i)}$ exists in $\mathbb{R}^{784}$, random points $p \in \mathbb{R}^{784}$ generally do not represent valid digits, indicating the real data lies on a lower-dimensional manifold.
- [ ] The distances between different distinct digits (e.g., '1' and '8') in raw Euclidean space accurately reflect their true topological differences.
- [ ] The dataset possesses a strict Euclidean distance matrix, ensuring Isomap incurs zero distortion error.

**Question 15 (2 Points)**
The objective functions (Strain and Stress) across the different MDS variants measure the quality of the embedding. Which of the following match the correct MDS variant to its objective?
- [ ] Classic MDS minimizes error wrt pairwise scalar products.
- [ ] Metric MDS minimizes error wrt pairwise scalar products.
- [ ] Non-metric MDS minimizes error wrt an ordering of distances.
- [ ] Classic MDS minimizes error wrt explicit Euclidean distance values directly.

***

# Part 2: Answer Key & Explanations

**Question 1**
- [ ] Incorrect. The slides explicitly state on slide 7: "for general distance matrices $D$, we cannot achieve such an embedding without distorting the data", regardless of choosing $d \ge n$.
- [ ] Correct. Slide 7 states: "if $D$ comes from points from $\mathbb{R}^d$ then we can find a correct point configuration... in this case, $D$ is called a Euclidean distance matrix".
- [ ] Incorrect. MDS (specifically the variants shown) does not necessarily seek to recover original high-dimensional coordinates, but rather abstract given distance objects into Euclidean space (slide 1: "given abstract objects...").
- [ ] Incorrect. Slide 6 mentions "visualization" as a use case, but also explicitly lists: "many algorithms are just defined for Euclidean data... if want to apply them, we need to find a Euclidean representation of our data."

**Question 2**
- [ ] Correct. Slide 2 explicitly states: "31 people rate on a five-point scale from 0 (no similarity at all) to 4 (identical)".
- [ ] Correct. Slide 2 mentions the average ratings are "scaled and subtracted from 1 to represent dissimilarities".
- [ ] Incorrect. Slide 2 explicitly notes the "resulting $14 \times 14$ dissimilarity matrix is symmetric".
- [ ] Incorrect. Slide 2 emphasizes "14 colors differing only in their hue (i.e., wavelengths...)". They varied in 1 physical dimension, not multiple.

**Question 3**
- [ ] Correct. Slide 9 expands the squared norm exactly as: $d_{i,j}^2 = \|x^{(i)} - x^{(j)}\|^2 = (x^{(i)})^\top x^{(i)} + (x^{(j)})^\top x^{(j)} - 2(x^{(i)})^\top x^{(j)}$.
- [ ] Correct. Slide 9 shows rearranging the terms exactly gives this identity: $(x^{(i)})^\top x^{(j)} = \frac{1}{2}((x^{(i)})^\top x^{(i)} + (x^{(j)})^\top x^{(j)} - d_{i,j}^2)$, where $(x^{(i)})^\top x^{(i)}$ is $d(0, x^{(i)})^2$.
- [ ] Correct. Slide 10 explains: "can pick origin 0 freely -> choose first data point $x^{(1)}$ to be origin", which allows mapping distances from the origin to scalar products.
- [ ] Correct. Slide 10 defines the strain embedding objective which is equivalent to $\min_{X} \sum_{i,j} (s_{i,j} - (x^{(i)})^\top x^{(j)})^2$.

**Question 4**
- [ ] Incorrect. Slide 11 explicitly states this is a "non-convex optimization problem".
- [ ] Correct. Slide 11 states: "however, optimal solution can be computed using eigenvalue decomposition or singular value decomposition (SVD)".
- [ ] Incorrect. Slide 12 gives the correct coordinate recovery formula as $X = V_d \sqrt{\Lambda_d}$, not $V_d \Lambda_d$.
- [ ] Incorrect. Slide 13 states that if data comes from $\mathbb{R}^d$, $S$ has rank $d$, meaning it has $d$ eigenvalues $> 0$ and $n - d$ eigenvalues equal to 0 (not $d$ eigenvalues equal to 0).

**Question 5**
- [ ] Correct. Slide 14 shows the Metric MDS stress function minimizes exactly $\sum_{i,j} \left(\|x^{(i)} - x^{(j)}\| - d_{i,j}\right)^2$.
- [ ] Incorrect. Slide 14 notes that Metric MDS is "NP-hard in general" and suffers from local minima, unlike Classic MDS which can be solved optimally via eigendecomposition.
- [ ] Correct. Slide 14 states for Metric MDS: "can be shown that the problem is NP-hard in general".
- [ ] Correct. Slide 14 contrasts Classic MDS (minimizes wrt scalar products) directly with Metric MDS (minimizes wrt distance function).

**Question 6**
- [ ] Correct. Slide 15 lists "distance matrix is not Euclidean, so we will not be able to recover a perfect embedding" as a source of error.
- [ ] Incorrect. While true for Isomap (Slide 24), this is not listed as a source of error for Metric MDS on Slide 15.
- [ ] Correct. Slide 15 explicitly lists that "the optimization problem are highly non-convex and the algorithm might return a local minima".
- [ ] Incorrect. Converting to a Gram matrix is a step in Classic MDS (Slide 10), not Metric MDS, which directly optimizes the stress on distances.

**Question 7**
- [ ] Correct. Slide 16 explicitly defines Non-metric MDS for cases where "instead of distance values, we are given only an ordering on the distances".
- [ ] Correct. Slide 18 shows the stress function incorporates a monotonic function $f$ applied to the distances to preserve the orderings.
- [ ] Correct. Slide 18 describes the algorithm as "alternate between computing an embedding $X$ and computing a monotonic function $f$".
- [ ] Incorrect. Slide 18 explicitly states the optimization problem for Non-metric MDS remains a "highly non-convex optimization problem".

**Question 8**
- [ ] Correct. Slide 22 states each image $x^{(i)} \in \mathbb{R}^{3 \cdot 76 \cdot 101}$.
- [ ] Correct. Slide 22 points out the "intrinsic dimension is two" due to the rotation of the teapot.
- [ ] Incorrect. Slide 22 concludes: "hence, can embed data set into two dimensions (circle)". The ambient size does not prevent embedding.
- [ ] Correct. The teapot is presented as the prime example on slide 22 of high-dimensional ambient data living on a low-dimensional manifold.

**Question 9**
- [ ] Incorrect. Slide 23 states the opposite: "in a small local region, Euclidean distances between data points coincide with the intrinsic distances of the manifold".
- [ ] Correct. Slide 23 and 24 indicate we "keep the local distances unchanged" and "construct $k$-nearest neighbor graph".
- [ ] Correct. Slide 24 states "shortest-path distances in $k$-nearest neighbor graph approximate geodesic distances".
- [ ] Incorrect. This refers to Kernel PCA (mentioned on slide 26 as an exercise comparison), not the core Isomap mechanism.

**Question 10**
- [ ] Incorrect. PCA is not the final step of Isomap on slide 25.
- [ ] Incorrect. The slides do not use Classic MDS for this step.
- [ ] Correct. Slide 25 explicitly states the final step is: "run metric MDS on $D$ (embed data into low-dimensional space while preserving geodesic distances)".
- [ ] Incorrect. Non-metric MDS preserves rankings, which is not Isomap's objective here.

**Question 11**
- [ ] Incorrect. Classic MDS requires exact values to form a Gram matrix, not just subjective rankings.
- [ ] Correct. Slide 16 explains Non-metric MDS is used because "humans can often compare two items relatively easy but it is usually hard to design a score to it".
- [ ] Incorrect. Isomap is for extracting low-dimensional continuous manifolds from high-dimensional ambient spaces (like image pixels), not arbitrary subjective pairwise rankings.
- [ ] Incorrect. Metric MDS relies on minimizing exact Euclidean distance errors, which are unavailable here since we only have rankings.

**Question 12**
- [ ] Correct. Slide 12 dictates setting $X = V_d \sqrt{\Lambda_d}$ using the top $d$ eigenvectors/values.
- [ ] Correct. Slide 12 asks "do you see the connection to PCA?" and slide 13 notes using the spectrum of $S$ to "pick a reasonable $d$ (as in PCA)".
- [ ] Correct. Slide 11 defines the minimization as $\|S - XX^\top\|_{\text{Fro}}^2$, establishing the intended relationship $S \approx XX^\top$.
- [ ] Correct. Slide 13 states that if the data comes from $\mathbb{R}^d$, $S$ has rank $d$; if it does not, we use the spectrum to "pick a reasonable $d$", implying $S$ has a broader spectrum (full rank).

**Question 13**
- [ ] Correct. Slide 11 states Classic MDS is a "non-convex optimization problem" (even though solvable by SVD).
- [ ] Correct. Slide 14 states Metric MDS is "again, a non-convex optimization problem".
- [ ] Correct. Slide 18 states Non-metric MDS is a "highly non-convex optimization problem".
- [ ] Incorrect. PCA (Eigenvalue decomposition/SVD) is analytically solvable and not characterized in the slides as a non-convex optimization challenge that suffers from local minima.

**Question 14**
- [ ] Incorrect. Slide 21 specifically asserts "not every $p \in \mathbb{R}^{784}$ gives rise to a 'proper' image".
- [ ] Correct. Slide 21 uses this exact logic: because arbitrary vectors don't make digits, the real digit data "lives on a low-dimensional manifold".
- [ ] Incorrect. The slides use manifolds precisely because raw ambient Euclidean distances over large spans *fail* to capture the true intrinsic topological structure.
- [ ] Incorrect. The image manifold is highly non-linear, so its raw pairwise Euclidean distance matrix is not a perfect Euclidean matrix without distortion (which Isomap corrects via geodesics).

**Question 15**
- [ ] Correct. Slide 19 summary states: "classical MDS minimizes the error wrt to the scalar products".
- [ ] Incorrect. Metric MDS minimizes error wrt the distance function, not scalar products (Slide 19).
- [ ] Correct. Slide 19 summary states: "non-metric MDS tries to preserve ordering on the distances".
- [ ] Incorrect. Classic MDS optimizes wrt scalar products (the Gram matrix S), not explicit distance values (which is Metric MDS).