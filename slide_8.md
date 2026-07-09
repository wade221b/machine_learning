Ah, I see exactly what happened! In Markdown (the formatting language this chat
uses), a single line break is often just rendered as a space. To force things
onto a new line, you either need double line breaks or you need to format them
as a bulleted list. Because I didn't use bullet points for the checkboxes, your
markdown viewer clumped them all into one giant paragraph.

Here is the cleanly formatted version of all 15 questions using bullet points so
they render perfectly on separate lines.

Below that, I have updated the Master Prompt to specifically instruct the AI to
use bullet points for the options so this never happens again.

Part 1: Cleanly Formatted Practice Questions

Topic 1: Discriminative vs. Generative Models

Question 1 (3 Points) Which of the following statements correctly describe
generative and discriminative models?

  - ☐ Discriminative models attempt to model the joint probability distribution
    p(x, y).
  - ☐ Generative models can be used to synthesize new, artificial data points.
  - ☐ For a given classification task, a generative model typically requires
    fewer parameters to be estimated than a discriminative model.
  - ☐ Discriminative models focus on modeling the decision boundary between
    classes, i.e., p(y|x).

Topic 2: Quadratic Discriminant Analysis (QDA)

Question 2 (4 Points) Suppose you are applying Quadratic Discriminant Analysis
(QDA) to a dataset with classes c \in \{-1, +1\} where the data points
x \in \mathbb{R}^d. Which of the following statements are true?

  - ☐ QDA assumes that the data points for each class follow a uniform
    distribution.
  - ☐ The covariance matrix \Sigma must be calculated shared across all classes
    to form the quadratic decision boundary.
  - ☐ The Maximum Likelihood Estimate (MLE) for the class mean \mu is simply the
    arithmetic mean of the data points belonging to that class.
  - ☐ The total number of parameters to be estimated in a two-class QDA model
    scales quadratically with the number of dimensions d.

Question 3 (3 Points) When deriving the decision boundary for QDA by setting
p(y = -1 | x) = p(y = +1 | x), which of the following mathematical outcomes
occur?

  - ☐ The normalization constant \frac{1}{\sqrt{(2\pi)^d|\Sigma|}} cancels out
    entirely because it is independent of the class.
  - ☐ The log-likelihood involves the determinant of the covariance matrices
    |\Sigma_{-1}| and |\Sigma_{+1}|.
  - ☐ The resulting equation simplifies to the form x^T\Sigma x + v^Tx + t = 0.
  - ☐ The algorithm maps the data into a higher-dimensional space to find a
    linear boundary.

Topic 3: Linear Discriminant Analysis (LDA)

Question 4 (3 Points) Which of the following are true regarding Linear
Discriminant Analysis (LDA) compared to QDA?

  - ☐ LDA assumes that \Sigma = \Sigma_{-1} = \Sigma_{+1}.
  - ☐ LDA is a discriminative model, whereas QDA is a generative model.
  - ☐ Because the covariance matrices are assumed to be equal in LDA, the
    quadratic terms x^T\Sigma^{-1}x cancel out, resulting in a linear decision
    boundary.
  - ☐ The linear decision boundary found by LDA always coincides exactly with
    the boundary found by hard-margin Support Vector Machines.

Topic 4: Unsupervised Learning & k-means Objective

Question 5 (3 Points) Which of the following are true about the k-means
clustering algorithm and its objective function?

  - ☐ The goal is to find a discrete partition of the dataset that maximizes the
    within-cluster distances.
  - ☐ The standard k-means objective function is a non-convex optimization
    problem.
  - ☐ Finding the global optimal solution for the k-means objective is NP-hard.
  - ☐ The algorithm always constructs clusters that have non-linear, highly
    complex boundaries.

Topic 5: Lloyd's Algorithm and k-means variants

Question 6 (4 Points) Regarding Lloyd's algorithm (the standard k-means
algorithm heuristic), which of the following statements are correct?

  - ☐ The algorithm is guaranteed to terminate after a finite number of
    iterations.
  - ☐ In every iteration of the while loop, the objective function either
    decreases or stays the same.
  - ☐ The algorithm is guaranteed to find the global optimum if allowed to run
    for enough iterations.
  - ☐ The algorithm's final output heavily depends on the initial placement of
    the cluster centers.

Question 7 (3 Points) Consider the properties of k-means clustering and its
variants. Which of the following are true?

  - ☐ The partition induced by the k-means objective corresponds to a Voronoi
    partition of the space, meaning all cluster boundaries are linear.
  - ☐ k-means is highly robust to outliers because it uses the squared Euclidean
    distance ||x^{(i)} - \mu_j||^2.
  - ☐ k-median replaces the squared distance with standard Euclidean distance
    ||.||_2, making it more robust against outliers.
  - ☐ The kmeans++ initialization method selects all k initial centers
    completely uniformly at random from the dataset to ensure unbiasedness.

Topic 6: Learning Paradigms & Generative vs. Discriminative Characteristics

Question 8 (3 Points) Based on the introduction to supervised and unsupervised
learning, which of the following statements are correct?

  - ☐ Semi-supervised learning refers to a scenario where we have an unlabeled
    dataset but the goal is to predict p(y|x).
  - ☐ Dimensionality reduction and density estimation are both forms of
    unsupervised learning.
  - ☐ Attaching labels to data points is generally considered an inexpensive and
    trivial task in real-world scenarios.
  - ☐ Generative models usually require much more data to estimate accurately
    compared to discriminative models because they have a higher number of
    parameters.

Topic 7: Deep Dive into Quadratic Discriminant Analysis (QDA)

Question 9 (4 Points) When estimating the parameters \mu and \Sigma for QDA
using Maximum Likelihood Estimation (MLE), which of the following mathematical
statements are true according to the lecture?

  - ☐ The algorithm finds the parameters by minimizing the log-likelihood
    function.
  - ☐ The log-likelihood function is concave with respect to \mu and \Sigma.
  - ☐ Taking the derivative of the log-likelihood with respect to \mu and
    setting it to 0 yields the equation \sum_i \Sigma^{-1}(x^{(i)} - \mu) = 0.
  - ☐ The estimated joint distribution for QDA is modeled as
    p(x, y) = p(y|x) \cdot p(x).

Question 10 (3 Points) Consider the parameter complexity of QDA versus LDA for a
d-dimensional dataset with C=2 classes (classes \{-1, +1\}).

  - ☐ In QDA, the number of parameters to be estimated is exactly
    \frac{d(d+1)}{2} + 2d.
  - ☐ LDA requires estimating fewer parameters than QDA because LDA assumes
    \Sigma = \Sigma_{-1} = \Sigma_{+1}.
  - ☐ If you have very limited training data, QDA is generally preferred over
    LDA because its quadratic boundary prevents overfitting.
  - ☐ ChatGPT and DALL-E are provided in the slides as examples of
    discriminative models capable of generating synthetic data.

Topic 8: K-means Edge Cases and Computational Complexity

Question 11 (4 Points) Which of the following statements about the computational
complexity and optimality of k-means are true?

  - ☐ If we fix the dimension d to a constant (e.g., d=2), finding the globally
    optimal solution to the k-means problem becomes solvable in polynomial time.
  - ☐ Optimizing the k-means objective is known to have a polynomial smoothed
    complexity, which helps explain why it usually works well in practice.
  - ☐ By arranging 4 data points on a real 1D line with specific spacings
    (a, b, c), it is possible to construct a scenario where Lloyd's algorithm
    converges to a local optimum that is arbitrarily worse than the global
    optimum.
  - ☐ The kmeans++ initialization algorithm guarantees that the final objective
    value will be exactly the global minimum.

Topic 9: K-means Equivalence and Objective Functions

Question 12 (3 Points) Which of the following optimization problems are
mathematically equivalent to the standard k-means objective?

  - ☐ Finding cluster centers \mu_j such that the sum of the squared Euclidean
    distances from data points to their assigned centers is minimized.
  - ☐ Finding a discrete partition of the dataset such that the sum of squared
    distances between all pairs of points within the same cluster is minimized.
  - ☐ Maximizing the distance between the cluster centers \mu_1, ..., \mu_k.
  - ☐ Minimizing the L1 norm (||.||_1) of the distances from the data points to
    the cluster centers.

Topic 10: Practical Heuristics and K-means Variants

Question 13 (3 Points) In practice, Lloyd's algorithm can get stuck or produce
suboptimal results. Which of the following heuristics/practices were recommended
in the slides to improve clustering?

  - ☐ Because it is a highly non-convex problem, it is standard practice to run
    the algorithm many times with different initializations and keep the result
    with the lowest objective value.
  - ☐ If a cluster center "loses" all its data points during an iteration, the
    algorithm must terminate immediately and return an error.
  - ☐ A valid local search heuristic is to split a cluster that has a very bad
    objective function into two pieces and randomly remove another cluster
    center.
  - ☐ A valid local search heuristic is to merge two existing clusters and
    introduce a completely new cluster center elsewhere.

Question 14 (3 Points) Regarding the initialization step of k-means, which of
the following are true?

  - ☐ The most common (but naive) initialization is to uniformly at random
    choose arbitrary coordinates in the feature space as starting centers.
  - ☐ kmeans++ selects the first center uniformly at random from the data
    points.
  - ☐ kmeans++ selects subsequent centers x with a probability inversely
    proportional to their squared distance from the closest existing center.
  - ☐ With careful initialization (like kmeans++), one can achieve an expected
    objective value that is at most a factor of O(\log k) worse than the optimal
    solution.

Question 15 (2 Points) Which of the following correctly describes a variant of
the k-means algorithm?

  - ☐ k-median replaces the squared Euclidean distance ||.||_2^2 with the
    standard Euclidean distance ||.||_2, making it more robust against outliers.
  - ☐ Soft k-means removes hard assignments, assigning probabilities of
    belonging to clusters, and is mathematically related to Mixture of
    Gaussians.
  - ☐ Weighted k-means introduces weights for individual features (dimensions)
    to perform implicit dimensionality reduction.

Part 2: The New, Formatting-Safe Master Prompt

Copy and paste the text below into a new ChatGPT window whenever you want to
study a new set of lecture slides. I added a strict markdown formatting rule
(Rule 2b).

Copy below this line:

Act as an expert University Professor in Machine Learning. I am going to provide
you with a PDF (or text) of lecture slides. I need you to generate a rigorous,
comprehensive practice exam based strictly on the content of these slides.

Please follow these exact constraints to match my university's high-level
testing format:

1. Quantity and Coverage (CRITICAL):

  - Do not give me a short quiz. I need a high volume of questions (aim
    for 15-20 questions depending on the length of the slides).
  - You must extract testable material from almost every slide, ensuring no
    major concept, mathematical derivation, or theoretical limitation is left
    out.

2. Question Format & Markdown Rendering:

  - Format every question as a multiple-choice question.
  - CRITICAL RULE 2a: For every single question, the number of correct answers
    can be 0, 1, or more than 1. Ensure you actively design questions where ALL
    options might be wrong, or ALL options might be right, to truly test my
    confidence.
  - CRITICAL RULE 2b: To ensure proper Markdown rendering, format the
    multiple-choice options as a bulleted list. Each option MUST start with - ☐ 
    on a new line so they do not clump together in the markdown viewer.
  - Assign a point value to each question (e.g., 2 Points, 3 Points, 4 Points)
    based on its difficulty.
  - Group the questions by "Topic" based on the chronological flow of the
    lecture slides.

3. Content Rigor & Depth:

  - Do not just test basic trivia. Deeply test mathematical intuitions (e.g.,
    MLE derivations, objective functions), algorithm properties, computational
    complexities (e.g., NP-hardness, Big-O), and specific heuristics or edge
    cases mentioned in the slides.
  - If the slides contain a specific proof, example, or visual (like a 1D
    failure case for an algorithm), turn the mechanics of that example into a
    question.
  - Include tricky distractor options that represent common ML misconceptions
    (e.g., mixing up generative vs. discriminative traits, confusing L1/L2
    norms, misinterpreting min vs. max, or misunderstanding convergence
    guarantees).

4. Output Structure:

  - Part 1: The Test. Provide only the questions and the options formatted as
    bullet points (- ☐). Do not reveal the answers here.
  - Part 2: Answer Key & Explanations. Below the test, provide an answer key.
    For every single option (both right and wrong), write a 1-2 sentence
    detailed explanation of why it is correct or incorrect, explicitly
    referencing the concepts/formulas that would be on the slides. (e.g.,
    "Option 1 (False): Model X is generative, not discriminative because it
    models the joint distribution.")

Please acknowledge you understand these rules, and I will upload the slides for
you to process.
