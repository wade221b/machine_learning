# Regression, Regularization & Model Selection — Cheat Sheet

Based on: *Polynomial Regression, Basis Functions, Overfitting, Cross-Validation, Regularization, Bias-Variance Tradeoff* lecture slides.

---

## 1. Polynomial Regression

**Name:** Polynomial Regression

**Explanation:**
Instead of fitting a straight line, we let our prediction function be a polynomial in $x$:
$$f(x) = w_d x^d + w_{d-1}x^{d-1} + \dots + w_1 x + w_0$$
We still want to minimize the squared loss between predictions $\hat y$ and true $y$:
$$l(y,\hat y) = \tfrac{1}{2}(y-\hat y)^2$$

**Key trick — turning it into linear regression:**
Even though $f(x)$ is non-linear in $x$, it is **linear in the weights $w$**. So if we map each input $x$ to a feature vector
$$\tilde x = (x^d, x^{d-1}, \dots, x, 1)^\top$$
then $f(x) = \tilde x^\top w$, and the whole problem becomes an ordinary **linear least-squares regression**:
$$\min_w \tfrac{1}{2}\|Xw - y\|^2$$
where each row of $X$ is one $\tilde x^{(i)}$.

**Formula (design matrix):**
$$X = \begin{pmatrix} (x^{(1)})^d & \cdots & x^{(1)} & 1 \\ \vdots & & \vdots & \vdots \\ (x^{(n)})^d & \cdots & x^{(n)} & 1 \end{pmatrix}, \quad y = \begin{pmatrix} y^{(1)} \\ \vdots \\ y^{(n)} \end{pmatrix}$$

**Multi-dimensional case:** With $m$ input features, a monomial of degree $d$ is $x_1^{\alpha_1}\cdots x_m^{\alpha_m}$ with $\sum_i \alpha_i = d$. Using all monomials up to degree $d$ gives $\binom{m+d}{d}$ features — this blows up fast and is impractical for large $m$.

**Pros:**
- Reuses all the machinery of linear regression (closed-form solution, gradient descent, convexity guarantees) for a much richer, non-linear function class.
- Simple, interpretable, easy to implement via a feature map.

**Cons:**
- Number of features explodes combinatorially with input dimension $m$ and degree $d$ — impractical for many features.
- High-degree polynomials are very prone to **overfitting** (wiggling to fit noise, see below).

**When to use it:** Small number of input dimensions, when you expect smooth/curved (not just linear) relationships in the data.

---

## 2. Basis Functions (General Feature Maps)

**Name:** Basis Function Regression / Generalized Linear Regression

**Explanation:**
The "polynomial trick" generalizes: instead of monomials, use *any* set of $d$ functions $f_1(x), \dots, f_d(x)$ (e.g. $\sin$, $\cos$, $\sqrt{\cdot}$, etc.) as your features:
$$f(x) = w_1 f_1(x) + w_2 f_2(x) + \dots + w_d f_d(x)$$
Map $x \mapsto (f_1(x), \dots, f_d(x))$, build the design matrix $X$ the same way as before, and solve the same least-squares problem $\min_w \tfrac{1}{2}\|Xw-y\|^2$.

**Pros:**
- Extremely flexible — with enough/varied basis functions you can approximate essentially **any** smooth function (polynomials, piecewise-linear functions, Fourier series, wavelets are all "uniform approximators," per the Weierstrass theorem, 1865).
- Still reduces to a simple, convex linear regression problem.

**Cons:**
- More basis functions = more capacity = more risk of overfitting (see below) — using tons of basis functions does **not** automatically "solve ML."

**When to use it:** Whenever you have domain knowledge about the shape of the relationship (periodic → use sin/cos, etc.), or want flexible non-linear regression while keeping a convex, easy-to-optimize problem.

---

## 3. Overfitting vs. Underfitting

**Name:** Overfitting / Underfitting

**Explanation:**
- **Underfitting:** model is too simple (e.g. a straight line for curved data) — it fails to capture the true signal. Both training and test error are high.
- **Overfitting:** model is too complex (e.g. a 9th-degree polynomial for a handful of points) — it fits the training data (including its noise) almost perfectly, but generalizes poorly to new/unseen data.
- **Good fit:** model complexity matches the true underlying signal — captures the pattern but not the noise.

**Visual intuition:** As model complexity increases, training error keeps decreasing (monotonically, even to zero), but test/validation error first decreases, then starts increasing again once the model starts fitting noise — this creates a U-shaped test-error curve.

**Pros/Cons:** N/A — this is a diagnostic concept, not a method.

**When to check for it:** Always, after training any model — compare training vs. validation error.

---

## 4. Train / Validation / Test Split & k-Fold Cross-Validation

**Name:** Data Splitting and k-Fold Cross-Validation

**Explanation:**
To detect over/underfitting reliably, split your data into three parts:
- **Training data:** used to fit the model parameters $w$.
- **Validation data:** used to estimate the generalization error and select model complexity/hyperparameters — **never used for training**.
- **Test data:** touched only once, at the very end, to report final performance — **never used for training or model selection**.

**k-Fold Cross-Validation:** When data is limited, instead of a single training/validation split:
1. Split training data into $k$ blocks (commonly $k=10$).
2. Use block 1 as validation, remaining $k-1$ blocks as training; train and record training/validation error.
3. Repeat, using each of the $k$ blocks as validation exactly once.
4. Average the training and validation errors over all $k$ runs.

**Pros:**
- Much more reliable estimate of generalization error than a single train/validation split, especially useful with limited data (uses all the data for both training and validation, just not simultaneously).

**Cons:**
- $k\times$ more compute (train the model $k$ times).

**When to use it:** Whenever you need to pick a hyperparameter (model complexity, regularization strength $\lambda$, etc.) and data is scarce.

**Full model-selection procedure:**
1. Compute a model-complexity-vs-error diagram using k-fold CV.
2. Pick the complexity with minimal *validation* error.
3. Retrain on **all** available training data (original training + validation) at that chosen complexity to get the final parameters $w^*$.

---

## 5. Regularization

**Name:** Regularization / Regularized Risk Minimization (RRM)

**Explanation:**
Instead of reducing model complexity by hand (e.g. lowering polynomial degree), we can keep a rich/complex function class but **constrain the size of the weights** $w$ — large, wild coefficients are usually a symptom of overfitting.

**Formula:**
Constrained form:
$$\min_w L(w) \quad \text{s.t.} \quad \|w\|_2^2 \le t$$
Equivalent unconstrained (Lagrangian) form:
$$\min_w L(w) + \frac{\lambda}{2}\|w\|_2^2$$

More generally, if $L(w) = \frac1n\sum_i l(y^{(i)}, \hat y^{(i)})$ is the **empirical risk**, then:
- $\min_w L(w)$ is called **Empirical Risk Minimization (ERM)**.
- $\min_w L(w) + \|w\|$ is called **Regularized Risk Minimization (RRM)**.

**Pros:**
- Directly fights overfitting without needing to redesign the feature/model class.
- Different norms give different useful properties (see Ridge vs. LASSO below).

**Cons:**
- Introduces a new hyperparameter $\lambda$ that must itself be tuned (typically via cross-validation, producing a "regularization path").
- Regularizers are usually **not** invariant to feature scaling — you must scale features first (see Feature Scaling below).

**When to use it:** Essentially always, whenever you have a rich model class and are worried about overfitting (which is almost always in practice).

---

## 6. Bias–Variance Tradeoff

**Name:** Bias-Variance Decomposition

**Explanation:**
The expected prediction error on unseen data can be mathematically decomposed into two competing sources of error:
- **Variance:** how much the learned model $f_n$ would change if trained on a *different* random training set of the same size — reflects sensitivity to the specific training sample.
- **Bias:** how far the *average* prediction (over many training sets) is from the true function $f^*$ — reflects systematic error from the model class being too restrictive.

**Formula (for squared loss):**
$$\mathbb{E}\left[(f_n(x)-f^*(x))^2\right] = \underbrace{\mathbb{E}\left[(f_n(x)-\mathbb{E}[f_n(x)])^2\right]}_{\text{variance}} + \underbrace{\left(\mathbb{E}[f_n(x)] - f^*(x)\right)^2}_{\text{bias}^2}$$

**Intuition:**
- **High bias, low variance:** simple models (e.g. linear fit) — consistently miss the true pattern (underfitting).
- **Low bias, high variance:** complex models (e.g. high-degree polynomial) — fit training data well on average but vary wildly across different training samples (overfitting).
- As model complexity grows: bias² decreases monotonically, variance increases monotonically; total error (their sum) is U-shaped, with a sweet spot in between.

**Pros/Cons:** N/A — a diagnostic/conceptual framework, not a method to apply directly (though it explains *why* regularization and model-selection work).

**When it matters:** Explains *why* neither "always use the simplest model" nor "always use tons of basis functions/maximum complexity" is correct — you must balance the two. Same decomposition idea extends (with different formulas) to other loss functions besides squared loss.

---

## 7. Ridge Regression (L2-Regularized Least Squares)

**Name:** Ridge Regression

**Explanation:**
Least-squares regression with an $\ell_2$ (Euclidean norm) regularizer on the weights.

**Formula:**
$$\min_w \frac{1}{2n}\|Xw-y\|_2^2 + \frac{\lambda}{2}\|w\|_2^2$$

**Pros:**
- Still has a closed-form solution (solvable via a linear system) since the loss stays smooth/quadratic — or can use gradient descent.
- Shrinks all coefficients smoothly toward zero, reducing variance and improving generalization/stability, especially when features are correlated.

**Cons:**
- Does **not** produce sparse solutions — coefficients shrink but rarely become exactly zero, so it doesn't perform feature selection.

**When to use it:** When you want to reduce overfitting/variance but keep (and use) all features, especially with many correlated features.

---

## 8. LASSO (L1-Regularized Least Squares)

**Name:** LASSO (Least Absolute Shrinkage and Selection Operator)

**Explanation:**
Motivating scenario: many features, few samples (e.g. gene expression data with $n\approx$ hundreds of patients but $d\approx$ thousands of genes) — least squares alone has infinitely many optimal solutions in this "underdetermined" regime. We'd like a **sparse** solution — most coefficients $w_i$ exactly zero — for interpretability (which genes actually matter?).

Ideally we'd directly constrain the number of nonzero coefficients ($\|w\|_0 \le t$), but this is **NP-hard**. Instead, we use the $\ell_1$-norm as the tightest convex relaxation of $\|\cdot\|_0$.

**Formula:**
$$\min_w \frac{1}{2n}\|Xw-y\|_2^2 + \lambda\|w\|_1$$

**Why $\ell_1$ gives sparsity (geometric intuition):** The $\ell_1$-norm "ball" has sharp corners on the coordinate axes; the loss contours are much more likely to first touch the constraint region exactly at a corner — where some coordinates are exactly zero — than the $\ell_2$-norm ball (smooth circle), which almost never touches at an axis.

**Pros:**
- Automatically performs **feature selection** (sparse $w$).
- Very useful when you have far more features than samples.

**Cons:**
- Not differentiable everywhere (the $\ell_1$ norm has kinks) — can't use plain gradient descent with line search; need subgradient methods or more specialized/efficient solvers (e.g. coordinate descent).
- Can behave erratically when features are highly correlated (tends to arbitrarily pick one of a correlated group).

**When to use it:** High-dimensional data with many irrelevant/redundant features, where you want both regularization *and* automatic feature selection/interpretability.

---

## 9. Elastic Net

**Name:** Elastic Net

**Explanation:**
A weighted combination of the LASSO ($\ell_1$) and Ridge ($\ell_2$) penalties, interpolating between the two.

**Formula:**
$$\min_w \frac{1}{2n}\|Xw-y\|_2^2 + \lambda\left(\alpha\|w\|_1 + \frac{1-\alpha}{2}\|w\|_2^2\right)$$
($\alpha=1$ recovers LASSO, $\alpha=0$ recovers Ridge.)

**Pros:**
- Gets sparsity like LASSO, but is more stable with correlated features (like Ridge) — tends to select or drop correlated groups of features together rather than arbitrarily.

**Cons:**
- Two hyperparameters to tune ($\lambda$ and $\alpha$) instead of one.

**When to use it:** High-dimensional data with correlated features (classic example: gene expression data) where pure LASSO would be too unstable.

---

## 10. Robust Regression (Handling Outliers)

**Name:** Robust Regression

**Explanation:**
Standard least-squares loss ($\ell_2^2$) squares the residuals, so a single large outlier can dominate the loss and badly skew the fit — this implicitly assumes Gaussian-distributed noise. If your data has outliers / non-Gaussian noise, use a loss that penalizes large residuals less severely.

**Formula:**
$$\min_w \frac{1}{n}\|Xw-y\|_1$$
(using the $\ell_1$-norm of the *residuals*, not of $w$ — note this is different from LASSO, which regularizes $w$).

**Alternative — Huber Loss:** A hybrid loss that behaves quadratically ($\ell_2^2$-like) for small residuals but linearly ($\ell_1$-like) for large residuals — combines the smoothness/optimization-friendliness of squared loss with the outlier-robustness of absolute loss.

**Pros:**
- Much less sensitive to outliers than squared loss.
- Can be combined with any regularizer.

**Cons:**
- The $\ell_1$ residual loss is non-differentiable at zero residual, so (like LASSO) needs subgradient or specialized methods.
- Huber loss requires an extra hyperparameter (the threshold where it switches from quadratic to linear behavior).

**When to use it:** Whenever your data likely contains outliers, or you don't trust the Gaussian-noise assumption behind ordinary least squares.

---

## 11. Feature Scaling

**Name:** Feature Scaling / Normalization

**Explanation:**
Before training (especially with regularization), all features should be roughly on the same numeric scale. Otherwise, features with naturally larger magnitudes will dominate the regularization penalty (and can slow down/destabilize gradient-based optimization) purely due to their units, not their actual importance.

**Formula (typical normalization procedure):**
1. **Center** each feature (column) by subtracting its mean:
$$X_{:,j}^{\text{centered}} = X_{:,j} - \bar x_j, \quad \bar x_j = \frac1n\sum_{i=1}^n X_{i,j}$$
2. **Scale** each column to unit norm:
$$X_{:,j}^{\text{scaled}} = \frac{X_{:,j}^{\text{centered}}}{\|X_{:,j}^{\text{centered}}\|_2}$$
(Alternatively, scale each feature into $[0,1]$ or $[-1,1]$.)

**Pros:**
- Makes regularization penalties fair across features (since regularizers are generally *not* invariant to feature scale).
- Improves interpretability of coefficients and often improves optimization behavior/convergence speed.

**Cons/Caveats:**
- **Critical:** Whatever scaling transform (and its parameters — mean, norm, etc.) is computed on the training data **must be reused unchanged on the test/validation data** — never recompute scaling parameters on test data, or you leak information / get inconsistent features.

**When to use it:** Essentially always before training regularized models (Ridge, LASSO, Elastic Net) or any gradient-based method with multiple features of different natural scales.

---

## Quick Comparison: Regularizers

| Method | Penalty | Sparse solution? | Differentiable? | Handles correlated features |
|---|---|---|---|---|
| Ridge | $\|w\|_2^2$ | No | Yes | Good (shrinks together) |
| LASSO | $\|w\|_1$ | Yes | No (needs subgradient) | Poor (picks one arbitrarily) |
| Elastic Net | mix of both | Yes | No (needs subgradient) | Good |

---

## Points to Watch Out for in a Master's-Level ML Exam

1. **Know the reduction trick cold:** polynomial regression → linear regression via the feature map $x \mapsto \tilde x$. Be ready to explicitly construct the design matrix $X$ for a given polynomial degree or basis function set, including the multi-dimensional monomial-counting formula $\binom{m+d}{d}$.
2. **Never confuse the three data splits** — training data trains parameters, validation data selects hyperparameters/model complexity, test data is touched exactly once at the end. This is a classic "what's wrong with this experimental setup" trick question.
3. **Be able to explain k-fold cross-validation step by step**, and know *why* it's preferred over a single split when data is scarce (uses all data for both roles, just not simultaneously; averages out the variance of a single random split).
4. **Understand the bias-variance decomposition derivation** and be able to state which knob (model complexity, regularization strength $\lambda$) moves you along the bias-variance curve and in which direction.
5. **Ridge vs. LASSO is a favorite exam topic:** know the formulas exactly, know that LASSO's $\ell_1$ penalty gives sparsity via its "corner" geometry (able to explain/sketch this geometric argument with the constraint-region vs. loss-contour picture), and know that LASSO is non-differentiable and thus needs subgradient methods.
6. **Remember $\|\cdot\|_0$ is NOT a norm** (it counts nonzero entries) and that exact sparse regression ($\|w\|_0 \le t$) is NP-hard — this is *why* we relax to $\ell_1$.
7. **Feature scaling is easy to forget but often tested:** know that regularizers are not scale-invariant, that you must scale features before regularized regression, and — critically — that test data must be scaled using the *training* data's scaling parameters, never recomputed.
8. **Connect concepts:** e.g., "Why does using too many basis functions not immediately solve ML?" requires linking basis functions → overfitting → high variance → bias-variance tradeoff → need for regularization/cross-validation.
9. **Elastic Net as an interpolation:** be ready to identify that setting $\alpha=1$ recovers LASSO and $\alpha=0$ recovers Ridge, and to explain why you might want the mix (sparsity + stability with correlated features).
10. **Robust regression vs. LASSO — don't mix these up.** Robust regression puts the $\ell_1$-norm on the **residuals** (loss function) to handle outliers; LASSO puts the $\ell_1$-norm on the **weights** (regularizer) to induce sparsity. Both are non-differentiable and solvable via subgradient methods, but they solve different problems.
