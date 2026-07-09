# Part 1: The Test

**Instructions:** For each question, select all options that apply. A question may have 0, 1, or multiple correct answers. 

---

### Topic 1: Standard PCA Objectives and Mathematical Formulation

**Question 1 (2 Points):**
According to the slides on standard PCA approaches, which of the following statements correctly describe the fundamental mathematical objectives of PCA?
- [ ] PCA aims to find a linear projection on a low-dimensional space such that the overall variance of the resulting points is minimized.
- [ ] PCA aims to find a projection $\pi_S$ on an affine subspace $S$ such that the squared distance between the points and their projections is minimized.
- [ ] The max-variance criterion and the minimum squared reconstruction error criterion lead to completely different subspace solutions.
- [ ] Standard PCA can be computed by finding the eigendecomposition of the empirical covariance matrix.

**Question 2 (3 Points):**
Given a dataset of $n$ points $x^{(1)}, \dots, x^{(n)} \in \mathbb{R}^d$, we wish to project this data onto an $\ell$-dimensional space ($\ell < d$). Let $X$ be the $n \times d$ data matrix with centered data points as rows. Which of the following accurately reflect the formulation for the 1-dimensional case ($\ell = 1$)? 
- [ ] The optimization problem is given by $\max_{a \in \mathbb{R}^d} a^\top (XX^\top) a$ subject to $\|a\| = 1$.
- [ ] The optimization problem is equivalent to $\max_{a \in \mathbb{R}^d} \|X^\top a\|^2$ subject to $\|a\| = 1$.
- [ ] The optimal solution is the eigenvector $v$ corresponding to the smallest eigenvalue of the matrix $X^\top X$.
- [ ] We assume the data points are centered; if they are not, they are centered by setting $\tilde{x}^{(i)} = x^{(i)} - \bar{x}$ where $\bar{x} = \frac{1}{d} \sum_{i=1}^d x^{(i)}$.

**Question 3 (2 Points):**
Based strictly on the algorithm presented for standard PCA (Maximize Variance), which of the following is the exact definition of the sample covariance matrix $C$ constructed from the centered data matrix $X$?
- [ ] $C = \frac{1}{n} X X^\top$
- [ ] $C = \frac{1}{n-1} X^\top X$
- [ ] $C = X X^\top$
- [ ] $C = X^\top X$

---

### Topic 2: Eigendecompositions, Projections, and Error

**Question 4 (3 Points):**
Suppose you have computed the eigendecomposition $C = V \Lambda V^\top$ and defined $V_\ell$ as the matrix containing the first $\ell$ columns of $V$ (assuming eigenvalues in $\Lambda$ are decreasing). How is the mapped data represented in the original space (View 1)?
- [ ] $\hat{x}^{(i)} = V_\ell V_\ell^\top \tilde{x}^{(i)} + \bar{x}$
- [ ] $\hat{x}^{(i)} = V_\ell^\top \tilde{x}^{(i)}$
- [ ] $\hat{x}^{(i)} = P\tilde{x}^{(i)} - \bar{x}$ where $P = V_\ell V_\ell^\top$
- [ ] The mapped points span a $d$-dimensional affine subspace.

**Question 5 (3 Points):**
When choosing the parameter $\ell$, the slides establish a bound on the reconstruction error. If $\lambda_k$ denotes the $k$-th eigenvalue of the covariance matrix in decreasing order, which of the following is the correct theoretical upper bound for the reconstruction error $\sum_i \|x^{(i)} - \pi_\ell(x^{(i)})\|^2$?
- [ ] $\le \sum_{k=1}^\ell \lambda_k$
- [ ] $\le \sum_{k=\ell+1}^d \lambda_k$
- [ ] $\le \sum_{k=1}^d \lambda_k - \sum_{k=1}^\ell \lambda_k^2$
- [ ] $\le \prod_{k=\ell+1}^d \lambda_k$

---

### Topic 3: PCA Limitations and Practical Considerations

**Question 6 (2 Points):**
Which of the following describes the conditions under which standard PCA works best, or explains its failure modes?
- [ ] Standard PCA works best if the data comes from a Gaussian distribution.
- [ ] Standard PCA provides strong mathematical guarantees regarding what happens to individual data points, similar to the Johnson-Lindenstrauss lemma.
- [ ] Standard PCA is highly robust to outliers because it optimizes a local criterion for every data point.
- [ ] Outliers have a large effect on the result mainly due to the use of the $\|\cdot\|_2^2$ norm in the objective function.

**Question 7 (2 Points):**
If PCA is utilized as a preprocessing step for a supervised learning algorithm, what does the lecture explicitly recommend for setting the parameter $\ell$?
- [ ] Look at the largest eigenvalues and take the most "informative" ones using a scree plot.
- [ ] Always choose an $\ell$ that captures at least 95% of the total variance.
- [ ] Use cross-validation to set the parameter $\ell$.
- [ ] Choose $\ell$ such that the reconstruction error strictly equals zero.

---

### Topic 4: Kernel PCA Foundations

**Question 8 (4 Points):**
Let $X$ be the $n \times d$ data matrix of centered data points. The covariance matrix is $C = X^\top X$ and the kernel matrix is $K = X X^\top$. Which of the following statements regarding the relationship between the eigenvectors of $K$ and $C$ are true?
- [ ] If $v \neq 0$ is an eigenvector of $C$ with eigenvalue $\lambda$, then $a := Xv \in \mathbb{R}^n$ is an eigenvector of $K$ with eigenvalue $\lambda$.
- [ ] If $a$ is an eigenvector of $K$ with eigenvalue $\lambda$, then $v := X^\top a \in \mathbb{R}^d$ (if $v \neq 0$) is an eigenvector of $C$ with eigenvalue $\lambda$.
- [ ] $C$ is a $d \times d$ matrix and thus its eigendecomposition has $d$ eigenvalues.
- [ ] To ensure the derived eigenvector $v$ of $C$ (obtained from an eigenvector $a$ of $K$ where $\|a\|=1$) is a unit vector, it must be normalized by $1/\sqrt{\lambda}$.

**Question 9 (3 Points):**
Reviewing the proof that eigenvectors of $K$ imply eigenvectors of $C$, which of the following lines correctly represents a valid step in proving $Cv = \lambda v$?
- [ ] $Ka = \lambda a \implies XX^\top a = \lambda a$
- [ ] $XX^\top a = \lambda a \implies X^\top X \cdot X^\top a = \lambda X^\top a$
- [ ] $X^\top X \cdot X^\top a = \lambda X^\top a \implies C v = \lambda v$ (where $v := X^\top a$)
- [ ] $\|v\|^2 = \|X^\top a\|^2 = a^\top X X^\top a = a^\top K a$

---

### Topic 5: Kernel PCA Algorithm and Properties

**Question 10 (3 Points):**
In the Kernel PCA algorithm, how do we project the whole dataset onto an eigenvector $v$ of the covariance matrix without explicitly needing access to the data matrix $X$?
- [ ] $\pi_v(X) = K \frac{1}{\sqrt{\lambda}} a$
- [ ] $\pi_v(X) = C \frac{1}{\lambda} a$
- [ ] $\pi_v(X) = X X^\top \frac{1}{\lambda^2} a$
- [ ] We cannot do this without access to $X$, which is why the kernel trick fails for PCA.

**Question 11 (3 Points):**
When computing the centered kernel matrix $\tilde{K}$ in feature space, let $\mathbb{1}$ be the matrix used in the slides (representing a matrix with elements $1/n$). The correct formula for centering the kernel matrix is:
- [ ] $\tilde{K} = K - \mathbb{1}K - K\mathbb{1} - \mathbb{1}K\mathbb{1}$
- [ ] $\tilde{K} = K - \mathbb{1}K - K\mathbb{1} + \mathbb{1}K\mathbb{1}$
- [ ] $\tilde{K} = K - \frac{1}{n}K - K\frac{1}{n} + \frac{1}{n^2}K$
- [ ] $\tilde{K} = K - \bar{x}K - K\bar{x} + \bar{x}K\bar{x}$

**Question 12 (3 Points):**
Consider step 4 and step 5 of the Kernel PCA algorithm. We compute the eigendecomposition $\tilde{K} = A D A^\top$. Let $A_{:,k}$ denote the $k$-th column of $A$. Which of the following defines how the low-dimensional representation points $Y \in \mathbb{R}^{n \times \ell}$ are computed?
- [ ] Define $V_\ell$ with columns $A_{:,k} / \sqrt{\lambda_k}$ (for $k = 1, \dots, \ell$), and output $Y = \tilde{K} \cdot V_\ell$.
- [ ] Define $V_\ell$ with columns $A_{:,k} \cdot \sqrt{\lambda_k}$ (for $k = 1, \dots, \ell$), and output $Y = \tilde{K} \cdot V_\ell$.
- [ ] Define $V_\ell$ with columns $A_{:,k}$ (for $k = 1, \dots, \ell$), and output $Y = V_\ell \cdot \tilde{K}$.
- [ ] Output $Y = K \cdot A_{:,\ell}$ where only the $\ell$-th column is used.

**Question 13 (2 Points):**
In the Kernel PCA Toy Example presented in the slides, applying a Gaussian kernel ($\sigma = 1$) to a 2-dimensional dataset ($d=2$) allowed projecting the data into $\ell=3$ dimensions. Which of the following statements justifies why this is possible?
- [ ] The Gaussian kernel maps data to a higher-dimensional RKHS (Reproducing Kernel Hilbert Space).
- [ ] The slides state that "we implicitly work in the RKHS, which has $n$ dimensions", thus allowing $\ell > d$ (up to $n$).
- [ ] The original data inherently had 3 dimensions, but one was zeroed out before standard PCA.
- [ ] Kernel PCA ignores the parameter $\ell$ and always outputs $n$ dimensions regardless of the user's input.

**Question 14 (2 Points):**
Based strictly on the provided slides, what are explicitly listed as applications of PCA? 
- [ ] Noise filtering by removing components with low eigenvalues.
- [ ] Feature extraction to extract important features to improve machine learning models.
- [ ] Dimensionality reduction to reduce the number of features while retaining most information.
- [ ] Outlier generation using the Johnson-Lindenstrauss projections.

**Question 15 (2 Points):**
Which of the following visual scenarios were explicitly shown in the slides as cases where standard PCA "can behave very badly" (fails to capture the underlying structure)?
- [ ] Data forming a Gaussian blob.
- [ ] Data forming concentric circles.
- [ ] Data forming an "S" shape.
- [ ] Data forming completely linearly separable, straight lines.

**Question 16 (2 Points):**
Assume you have centered data points. You run standard PCA and Kernel PCA using a linear kernel. Which of the following is true?
- [ ] Kernel PCA with a linear kernel will produce an entirely different projection than standard PCA because the matrices have different dimensions ($n \times n$ vs $d \times d$).
- [ ] The non-zero eigenvalues computed from $K$ will be identical to the non-zero eigenvalues computed from $C$.
- [ ] Standard PCA cannot be kernelized because the covariance matrix relies on actual coordinates, which cannot be expressed as dot products.
- [ ] The reconstruction error in standard PCA and the projection step in Kernel PCA both inherently rely on the Pythagorean Theorem to minimize variance.


---
---


# Part 2: Answer Key & Explanations

### Topic 1: Standard PCA Objectives and Mathematical Formulation

**Question 1:**
- [ ] PCA aims to find a linear projection on a low-dimensional space such that the overall variance of the resulting points is minimized.
  *Incorrect.* Slide 10 explicitly states the idea is to find a linear projection such that the overall variance is as *large* as possible (maximized), not minimized.
- [x] PCA aims to find a projection $\pi_S$ on an affine subspace $S$ such that the squared distance between the points and their projections is minimized.
  *Correct.* Slide 18 presents Approach 2, stating the goal is to find a projection $\pi_S$ such that the squared distance between points and their projections is minimized.
- [ ] The max-variance criterion and the minimum squared reconstruction error criterion lead to completely different subspace solutions.
  *Incorrect.* Slide 19 states that "can prove that minimizing squared reconstruction error leads to the same solution as the one induced by the max-variance criterion".
- [x] Standard PCA can be computed by finding the eigendecomposition of the empirical covariance matrix.
  *Correct.* Slide 42 (Summary) explicitly states that Standard PCA "can be computed by an eigendecomposition of the empirical covariance matrix".

**Question 2:**
- [ ] The optimization problem is given by $\max_{a \in \mathbb{R}^d} a^\top (XX^\top) a$ subject to $\|a\| = 1$.
  *Incorrect.* Slide 13 shows the formulation as $\max_{a \in \mathbb{R}^d} a^\top (X^\top X) a$, not $X X^\top$. 
- [ ] The optimization problem is equivalent to $\max_{a \in \mathbb{R}^d} \|X^\top a\|^2$ subject to $\|a\| = 1$.
  *Incorrect.* Slide 13 formulates it as $\max_{a \in \mathbb{R}^d} \|X a\|^2$, not $\|X^\top a\|^2$. 
- [ ] The optimal solution is the eigenvector $v$ corresponding to the smallest eigenvalue of the matrix $X^\top X$.
  *Incorrect.* Slide 13 states the solution is the eigenvector $v$ to the *largest* eigenvalue $\lambda_{\max}$ of $X^\top X$.
- [ ] We assume the data points are centered; if they are not, they are centered by setting $\tilde{x}^{(i)} = x^{(i)} - \bar{x}$ where $\bar{x} = \frac{1}{d} \sum_{i=1}^d x^{(i)}$.
  *Incorrect.* Slide 11 defines centering via $\bar{x} = \frac{1}{n} \sum_{i=1}^n x^{(i)}$, summing over the $n$ data points, not the $d$ dimensions. 
*(Note: 0 options are correct for this question).*

**Question 3:**
- [ ] $C = \frac{1}{n} X X^\top$
  *Incorrect.* The slides define $C$ using $X^\top X$, not $X X^\top$, and omit the $1/n$ factor in their explicit matrix formulation on Slide 15.
- [ ] $C = \frac{1}{n-1} X^\top X$
  *Incorrect.* While statistically common, this is not the formula presented in the slides.
- [ ] $C = X X^\top$
  *Incorrect.* $X X^\top$ produces the $n \times n$ kernel matrix $K$ (Slide 29), not the covariance matrix $C$.
- [x] $C = X^\top X$
  *Correct.* Slide 15 explicitly defines the $d \times d$ sample covariance matrix as $C = X^\top X$.

---

### Topic 2: Eigendecompositions, Projections, and Error

**Question 4:**
- [x] $\hat{x}^{(i)} = V_\ell V_\ell^\top \tilde{x}^{(i)} + \bar{x}$
  *Correct.* Slide 15 states that in View 1 (mapping points back to the original space), $\hat{x}^{(i)} = P\tilde{x}^{(i)} + \bar{x}$ with $P = V_\ell V_\ell^\top$.
- [ ] $\hat{x}^{(i)} = V_\ell^\top \tilde{x}^{(i)}$
  *Incorrect.* This formula corresponds to View 2 ($z^{(i)}$), which projects the point into $\mathbb{R}^\ell$, not back to the original space $\mathbb{R}^d$ (Slide 15).
- [ ] $\hat{x}^{(i)} = P\tilde{x}^{(i)} - \bar{x}$ where $P = V_\ell V_\ell^\top$
  *Incorrect.* Slide 15 shows we must *add* the mean $\bar{x}$ back, not subtract it ($+ \bar{x}$).
- [ ] The mapped points span a $d$-dimensional affine subspace.
  *Incorrect.* Slide 15 explicitly notes that "they span only an $\ell$-dimensional affine subspace".

**Question 5:**
- [ ] $\le \sum_{k=1}^\ell \lambda_k$
  *Incorrect.* This sum represents the variance *retained*, not the reconstruction error bounded by the variance *discarded*.
- [x] $\le \sum_{k=\ell+1}^d \lambda_k$
  *Correct.* Slide 20 explicitly provides the formula $\sum_i \|x^{(i)} - \pi_\ell(x^{(i)})\|^2 \le \sum_{k=\ell+1}^d \lambda_k$.
- [ ] $\le \sum_{k=1}^d \lambda_k - \sum_{k=1}^\ell \lambda_k^2$
  *Incorrect.* The bound is simply the sum of the discarded eigenvalues; no squared eigenvalues are used in this bound on the slide.
- [ ] $\le \prod_{k=\ell+1}^d \lambda_k$
  *Incorrect.* The error is bounded by the sum of the discarded eigenvalues, not their product (Slide 20).

---

### Topic 3: PCA Limitations and Practical Considerations

**Question 6:**
- [x] Standard PCA works best if the data comes from a Gaussian distribution.
  *Correct.* Slide 23 explicitly states, "PCA works best if the data comes from a Gaussian".
- [ ] Standard PCA provides strong mathematical guarantees regarding what happens to individual data points, similar to the Johnson-Lindenstrauss lemma.
  *Incorrect.* Slide 22 explicitly states there are "no guarantees what happens to individual data points" and contrasts this with methods like Johnson-Lindenstrauss that *do* guarantee this.
- [ ] Standard PCA is highly robust to outliers because it optimizes a local criterion for every data point.
  *Incorrect.* Slide 22 states PCA is sensitive to outliers and optimizes a *global* criterion (sum over all data points).
- [x] Outliers have a large effect on the result mainly due to the use of the $\|\cdot\|_2^2$ norm in the objective function.
  *Correct.* Slide 22 notes that outliers can have a large effect "mainly due to the $\|\cdot\|_2^2$" (squared error).

**Question 7:**
- [ ] Look at the largest eigenvalues and take the most "informative" ones using a scree plot.
  *Incorrect.* While Slide 20 mentions this as a general "heuristic", Slide 21 explicitly gives a different instruction when PCA is used as a preprocessing step for supervised learning.
- [ ] Always choose an $\ell$ that captures at least 95% of the total variance.
  *Incorrect.* The slides never mention a 95% variance threshold rule.
- [x] Use cross-validation to set the parameter $\ell$.
  *Correct.* Slide 21 explicitly states: "if PCA is used as a preprocessing step for supervised learning, then use cross-validation to set the parameter $\ell$".
- [ ] Choose $\ell$ such that the reconstruction error strictly equals zero.
  *Incorrect.* The slides do not state this; setting error to exactly zero would require $\ell = d$, defeating dimensionality reduction (Slide 20-21).

---

### Topic 4: Kernel PCA Foundations

**Question 8:**
- [x] If $v \neq 0$ is an eigenvector of $C$ with eigenvalue $\lambda$, then $a := Xv \in \mathbb{R}^n$ is an eigenvector of $K$ with eigenvalue $\lambda$.
  *Correct.* Slide 32 proves this exactly: $Ka = XX^\top (Xv) = X(X^\top X)v = XCv = X(\lambda v) = \lambda (Xv) = \lambda a$.
- [x] If $a$ is an eigenvector of $K$ with eigenvalue $\lambda$, then $v := X^\top a \in \mathbb{R}^d$ (if $v \neq 0$) is an eigenvector of $C$ with eigenvalue $\lambda$.
  *Correct.* Slide 30 proves this exactly: $Ka = \lambda a \implies XX^\top a = \lambda a \implies X^\top X X^\top a = \lambda X^\top a \implies Cv = \lambda v$.
- [x] $C$ is a $d \times d$ matrix and thus its eigendecomposition has $d$ eigenvalues.
  *Correct.* Slide 33 explicitly states: "$C$ is a $d \times d$-matrix, so its eigendecomposition has $d$ eigenvalues".
- [x] To ensure the derived eigenvector $v$ of $C$ (obtained from an eigenvector $a$ of $K$ where $\|a\|=1$) is a unit vector, it must be normalized by $1/\sqrt{\lambda}$.
  *Correct.* Slide 31 proves that $\|v\|^2 = \lambda \|a\|^2$, so if $\|a\|=1$, $\|v\|=\sqrt{\lambda}$. Thus, to obtain a unit vector, it must be normalized by $1/\sqrt{\lambda}$.

**Question 9:**
- [x] $Ka = \lambda a \implies XX^\top a = \lambda a$
  *Correct.* This is the exact first implication shown in the proof of (1) on Slide 30, substituting $K = XX^\top$.
- [x] $XX^\top a = \lambda a \implies X^\top X \cdot X^\top a = \lambda X^\top a$
  *Correct.* This is the exact second implication on Slide 30, achieved by left-multiplying by $X^\top$.
- [x] $X^\top X \cdot X^\top a = \lambda X^\top a \implies C v = \lambda v$ (where $v := X^\top a$)
  *Correct.* This is the final step of the proof on Slide 30, substituting $C = X^\top X$ and $v = X^\top a$.
- [x] $\|v\|^2 = \|X^\top a\|^2 = a^\top X X^\top a = a^\top K a$
  *Correct.* These are the exact equality steps shown in the proof of (2) on Slide 31.

---

### Topic 5: Kernel PCA Algorithm and Properties

**Question 10:**
- [x] $\pi_v(X) = K \frac{1}{\sqrt{\lambda}} a$
  *Correct.* Slide 35 derives the projection of the whole dataset onto eigenvector $v$ without $X$ explicitly: $\pi_v(X) = Xv = XX^\top \frac{1}{\sqrt{\lambda}}a = K \frac{1}{\sqrt{\lambda}}a$.
- [ ] $\pi_v(X) = C \frac{1}{\lambda} a$
  *Incorrect.* The projection relies on the kernel matrix $K$, not $C$, as shown on Slide 35.
- [ ] $\pi_v(X) = X X^\top \frac{1}{\lambda^2} a$
  *Incorrect.* The scaling factor is $1/\sqrt{\lambda}$, not $1/\lambda^2$ (Slide 35).
- [ ] We cannot do this without access to $X$, which is why the kernel trick fails for PCA.
  *Incorrect.* Slide 35 explicitly demonstrates that the expression "does not depend on $X$ explicitly anymore", allowing the kernel trick to succeed.

**Question 11:**
- [ ] $\tilde{K} = K - \mathbb{1}K - K\mathbb{1} - \mathbb{1}K\mathbb{1}$
  *Incorrect.* The final term must be added, not subtracted (Slide 36).
- [x] $\tilde{K} = K - \mathbb{1}K - K\mathbb{1} + \mathbb{1}K\mathbb{1}$
  *Correct.* Slide 36 provides this exact formula for computing the centered kernel matrix.
- [ ] $\tilde{K} = K - \frac{1}{n}K - K\frac{1}{n} + \frac{1}{n^2}K$
  *Incorrect.* While $\mathbb{1}$ represents a matrix related to $1/n$, the slides formulate it strictly using matrix multiplication with $\mathbb{1}$ (Slide 36).
- [ ] $\tilde{K} = K - \bar{x}K - K\bar{x} + \bar{x}K\bar{x}$
  *Incorrect.* $\bar{x}$ is a vector in the original feature space; kernel centering relies on the constant matrix $\mathbb{1}$, not $\bar{x}$ directly in this formula (Slide 36).

**Question 12:**
- [x] Define $V_\ell$ with columns $A_{:,k} / \sqrt{\lambda_k}$ (for $k = 1, \dots, \ell$), and output $Y = \tilde{K} \cdot V_\ell$.
  *Correct.* Slide 36 outlines the algorithm exactly as defining $V_\ell$ with columns $A_{:,k} / \sqrt{\lambda_k}$ (Step 4) and computing representation $Y = \tilde{K} \cdot V_\ell$ (Step 5).
- [ ] Define $V_\ell$ with columns $A_{:,k} \cdot \sqrt{\lambda_k}$ (for $k = 1, \dots, \ell$), and output $Y = \tilde{K} \cdot V_\ell$.
  *Incorrect.* The eigenvectors must be divided by $\sqrt{\lambda_k}$ to be normalized, not multiplied by it (Slide 36).
- [ ] Define $V_\ell$ with columns $A_{:,k}$ (for $k = 1, \dots, \ell$), and output $Y = V_\ell \cdot \tilde{K}$.
  *Incorrect.* The columns must be scaled by $1/\sqrt{\lambda_k}$, and the matrix multiplication order is $\tilde{K} \cdot V_\ell$ (Slide 36).
- [ ] Output $Y = K \cdot A_{:,\ell}$ where only the $\ell$-th column is used.
  *Incorrect.* We use the $\ell$ largest components in matrix $V_\ell$, and we multiply by the *centered* kernel matrix $\tilde{K}$, not $K$ (Slide 36).

**Question 13:**
- [x] The Gaussian kernel maps data to a higher-dimensional RKHS (Reproducing Kernel Hilbert Space).
  *Correct.* Slide 38 implies this by stating we implicitly work in the RKHS which has $n$ dimensions, allowing us to find more components than the original $d=2$.
- [x] The slides state that "we implicitly work in the RKHS, which has $n$ dimensions", thus allowing $\ell > d$ (up to $n$).
  *Correct.* Slide 38 literally states "note: we implicitly work in the RKHS, which has $n$ dimensions" and "makes sense to choose $\ell=3$ (even though the original data set just had $d=2$)".
- [ ] The original data inherently had 3 dimensions, but one was zeroed out before standard PCA.
  *Incorrect.* The original data is described strictly as a "two-dimensional toy data set" ($d=2$) on Slide 37.
- [ ] Kernel PCA ignores the parameter $\ell$ and always outputs $n$ dimensions regardless of the user's input.
  *Incorrect.* Slide 36 shows that the algorithm explicitly takes $\ell$ as an input parameter and outputs $Y \in \mathbb{R}^{n \times \ell}$.

**Question 14:**
- [x] Noise filtering by removing components with low eigenvalues.
  *Correct.* Slide 26 lists "noise filtering: by removing the components with low eigenvalues, PCA can denoise data".
- [x] Feature extraction to extract important features to improve machine learning models.
  *Correct.* Slide 26 explicitly lists "feature extraction... to improve machine learning models".
- [x] Dimensionality reduction to reduce the number of features while retaining most information.
  *Correct.* Slide 26 explicitly lists this.
- [ ] Outlier generation using the Johnson-Lindenstrauss projections.
  *Incorrect.* Slide 22 mentions Johnson-Lindenstrauss as an *alternative* that guarantees individual data point bounds, not as an application of PCA for generating outliers.

**Question 15:**
- [ ] Data forming a Gaussian blob.
  *Incorrect.* Slide 23 shows PCA works *best* on Gaussian data.
- [x] Data forming concentric circles.
  *Correct.* Slide 24 shows concentric circles as a case where PCA behaves "very badly".
- [x] Data forming an "S" shape.
  *Correct.* Slide 24 visually presents an "S" shape dataset as a case where standard PCA behaves "very badly".
- [ ] Data forming completely linearly separable, straight lines.
  *Incorrect.* This is exactly the kind of linear structure PCA is designed to find, not a case where it behaves "very badly".

**Question 16:**
- [ ] Kernel PCA with a linear kernel will produce an entirely different projection than standard PCA because the matrices have different dimensions ($n \times n$ vs $d \times d$).
  *Incorrect.* Slides 28-33 show that there is a tight mathematical equivalency between $C$ and $K$. A linear kernel exactly reproduces standard PCA via the dual representation.
- [x] The non-zero eigenvalues computed from $K$ will be identical to the non-zero eigenvalues computed from $C$.
  *Correct.* Slide 33 explains the paradox of dimensions ($n$ vs $d$) and indicates you can convert the eigenvalues of $K$ to those of $C$, meaning their non-zero spectra match.
- [ ] Standard PCA cannot be kernelized because the covariance matrix relies on actual coordinates, which cannot be expressed as dot products.
  *Incorrect.* Slide 29 specifically shows how the kernel matrix is derived using dot products $K_{ij} = (x^{(i)})^\top x^{(j)}$, proving it *can* be kernelized.
- [ ] The reconstruction error in standard PCA and the projection step in Kernel PCA both inherently rely on the Pythagorean Theorem to minimize variance.
  *Incorrect.* Standard PCA reconstruction error relates to the Pythagorean Theorem (Slide 19), but it is used to *maximize* variance, not minimize it.