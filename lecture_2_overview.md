# Optimization for ML — Cheat Sheet

Based on: *Optimization for ML* lecture slides (gradient descent, matrix calculus, convexity, subgradients).

---

## 1. The General Optimization Problem

**Name:** Unconstrained minimization of a loss function

**Explanation:**
In ML, we want to find the weights $w$ that make our model's loss as small as possible:

$$\min_{w \in \mathbb{R}^d} L(w)$$

Ideally we'd just set the gradient to zero and solve, but this only works if that gives you a linear system that's actually solvable in a reasonable amount of time. For most real ML losses (neural nets, non-linear models, etc.) this is not possible, so we need iterative optimization methods.

**Pros / Cons:** N/A (this is the problem setup, not a method).

**When it's relevant:** Whenever $\nabla L(w) = 0$ can't be solved directly (non-linear in $w$, or too large a system).

---

## 2. Gradient Descent

**Name:** Gradient Descent (GD)

**Explanation:**
An iterative method that repeatedly moves the weights a small step in the direction *opposite* the gradient, since the gradient points toward the steepest **increase** and we want to **decrease** the loss.

**Formula:**
$$w^{(t+1)} = w^{(t)} - \gamma^{(t)} \cdot \nabla L(w^{(t)})$$

where $\gamma^{(t)}$ is the **step length / learning rate**.

**Pros:**
- Simple to implement, only needs first-order gradient information.
- Cheap per iteration.
- Guaranteed to converge to the *global* minimum if $L$ is convex and the step size is chosen appropriately.

**Cons:**
- Can get stuck in a local minimum on non-convex functions.
- Choosing the step size is tricky — too large and it diverges/oscillates, too small and it's painfully slow.
- Convergence speed depends heavily on the curvature (conditioning) of the loss.

**When to use it:** Default first choice for smooth (differentiable) loss functions, especially convex ones like least-squares regression.

---

## 3. Computing Gradients & Matrix Calculus

**Name:** Gradients / Matrix Calculus (denominator layout)

**Explanation:**
To run gradient descent you need $\nabla L(w)$ — the vector of partial derivatives of $L$ with respect to each weight. For a scalar-valued function $L(w): \mathbb{R}^d \to \mathbb{R}$:

$$\nabla L = \begin{pmatrix} \partial L/\partial w_1 \\ \partial L/\partial w_2 \\ \vdots \\ \partial L/\partial w_d \end{pmatrix}$$

For a vector-valued function $L(w): \mathbb{R}^d \to \mathbb{R}^m$, the derivative becomes a matrix (Jacobian), with two possible conventions:
- **Denominator layout:** columns index output, rows index input.
- **Numerator layout:** everything transposed relative to denominator layout.

**Useful identities:**
| Function | Gradient |
|---|---|
| $L(w) = \|w\|_2^2 = w^\top w$ | $\nabla L(w) = 2w$ |
| $L(w) = w^\top A w$ | $\nabla L(w) = Aw + A^\top w$ |
| $L(w) = \|Xw - y\|^2$ | $\nabla L(w) = 2X^\top(Xw - y)$ (up to a factor of 2 depending on normalization) |

**Pros:** Once you know the identities, gradients of common ML losses (linear/least-squares regression, ridge regression, quadratic forms) can be written down almost by inspection.

**Cons:** Easy to make sign/transpose errors; layout convention (numerator vs. denominator) matters and mixing them up gives wrong results.

**When to use it:** Any time you need to derive an update rule by hand. Alternatively, use a tool like **www.MatrixCalculus.org** to double check — but be careful to match its layout convention to yours.

---

## 4. Local vs. Global Minimum

**Name:** Local Minimizer vs. Global Minimizer

**Explanation:**
- A point $x^*$ is a **local minimizer** if it has the lowest loss among *nearby* points (within some small radius $\varepsilon$).
- A point $x^*$ is a **global minimizer** if it has the lowest loss among *all* feasible points.

**Formula:**
- Local: $\exists \varepsilon>0$ s.t. $\forall y$ with $\|x^*-y\|\le\varepsilon: f(x^*)\le f(y)$
- Global: $\forall y \in X: f(x^*) \le f(y)$

**Why it matters:** Plain gradient descent only guarantees it will stop at *a* critical point / local minimum — on a general (non-convex) surface, this might not be the best possible solution (there can also be saddle points, where the gradient is zero but it's neither a min nor max).

**When to worry about this:** Whenever your loss function is not known to be convex (e.g., deep neural networks).

---

## 5. Convex Sets and Convex Functions

**Name:** Convexity

**Explanation:**
A **convex set** is one where the straight line between any two points in the set stays inside the set (no "dents" or holes).

A **convex function** is one whose graph curves upward everywhere — the line segment connecting any two points on the graph lies *above or on* the graph. Intuitively: it looks like a bowl, so there's only one "bottom."

**Formulas:**
- Convex set: $\lambda x + (1-\lambda) y \in C$ for all $x,y \in C$, $\lambda \in [0,1]$
- Convex function: $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda) f(y)$

**Why it matters (key theorems):**
- **Every local minimum of a convex function is also a global minimum.** This is the single most important fact in this cheat sheet — it's *why* gradient descent is safe to use on convex losses.
- If $f$ is convex and differentiable, **every critical point** (where $\nabla f(x) = 0$) **is a global minimum**.

**Pros of working with convex losses:** No risk of getting trapped in a bad local minimum; optimization theory gives strong guarantees.

**Cons:** Many real-world ML losses (deep nets) are *not* convex, so these guarantees don't directly apply.

**When to check for convexity:** Before trusting that gradient descent (or any local method) finds the *true* optimum — e.g., linear/logistic/ridge regression losses are convex, most deep learning losses are not.

---

## 6. Testing Convexity via the Hessian (Positive Semidefiniteness)

**Name:** Second-order convexity test / Positive Semidefinite (PSD) matrices

**Explanation:**
Instead of checking the convexity definition directly, for twice-differentiable functions you can just check the **Hessian matrix** (matrix of second derivatives). If the Hessian is positive semidefinite everywhere, the function is convex.

**Formula:**
$$\nabla^2 f(x) = \begin{pmatrix} \partial^2 f/\partial x_1\partial x_1 & \cdots & \partial^2 f/\partial x_1 \partial x_d \\ \vdots & & \vdots \\ \partial^2 f/\partial x_d \partial x_1 & \cdots & \partial^2 f/\partial x_d \partial x_d \end{pmatrix}$$

$f$ is convex $\iff$ $\mathrm{dom}(f)$ is convex and $\nabla^2 f(x) \succeq 0 \ \forall x$ (Hessian is PSD everywhere).

**Definition of PSD:** A symmetric matrix $A$ is positive semidefinite ($A \succeq 0$) if $x^\top A x \ge 0$ for all $x$; positive definite ($A \succ 0$) if $x^\top A x > 0$ for all $x \ne 0$.

**Key fact:** A symmetric matrix is PSD $\iff$ all its **eigenvalues are $\ge 0$**. It's positive definite $\iff$ all eigenvalues are $> 0$.

**Geometric intuition (eigenvalues as curvature):** Along each eigenvector direction, the (quadratic) function behaves like a simple 1-D parabola $\lambda \cdot x^2$, where $\lambda$ is the corresponding eigenvalue.
- All eigenvalues positive → bowl shape (convex, unique minimum).
- One eigenvalue negative → saddle shape (not convex).
- Very different eigenvalue magnitudes → "elongated" bowl, meaning gradient descent zig-zags and converges slowly (poor conditioning).

**Worked example — Least Squares Regression:**
$$L(w) = \tfrac{1}{2}\|Xw-y\|_2^2, \quad \nabla L(w) = X^\top(Xw-y), \quad \nabla^2 L(w) = X^\top X$$
$X^\top X$ is always symmetric and PSD, so the least-squares loss is **always convex** — gradient descent is guaranteed to find the true global minimum.

**When to use it:** Whenever you have a twice-differentiable loss and want to formally verify convexity (e.g., justifying that gradient descent will work well for a given ML model).

---

## 7. Line Search Methods (Choosing the Step Length)

### 7a. Exact Line Search

**Explanation:** Given your current point and descent direction, exactly solve a 1-D optimization problem to find the step length $t$ that minimizes the loss along that direction.

**Formula:**
$$t = \arg\min_{s\ge 0} f(x + s \cdot \Delta x)$$

**Pros:** Makes the most "progress" possible for that step.

**Cons:** Solving a sub-optimization problem exactly at every iteration is often expensive or impossible in closed form.

**When to use:** Only when the 1-D sub-problem is cheap to solve exactly (e.g., quadratic losses).

### 7b. Backtracking Line Search

**Explanation:** A cheaper, practical approximation to exact line search. Start with a large step ($t=1$) and shrink it by a factor $\beta$ until the loss decreases "enough" (per the Armijo condition).

**Formula (Algorithm):**
```
Input: x, Δx, parameters α ∈ (0, 0.5), β ∈ (0, 1)
t = 1
while f(x + tΔx) > f(x) + α t ∇f(x)ᵀΔx:
    t = β t
```

**Pros:** Cheap, practical, works well in practice, doesn't require solving anything exactly.

**Cons:** Requires tuning $\alpha$, $\beta$; still needs multiple function evaluations per iteration.

**When to use:** The standard practical choice for step-size selection in (sub)gradient-based methods when exact line search is too expensive.

**Important caveat:** Line search (exact or backtracking) **only works for genuinely differentiable functions** — it fails for subgradient methods (see below), because the loss along the ray isn't guaranteed to decrease monotonically.

---

## 8. Subgradient Method (Non-Differentiable Losses)

**Name:** Subgradient Method

**Explanation:**
Many ML losses aren't differentiable everywhere (e.g., $\|w\|_1$, used in Lasso/sparse regression, has sharp corners). At non-differentiable points, we replace the gradient with a **subgradient**: any vector $g$ that stays "below" the function everywhere.

**Formula (definition of subgradient):**
$$f(y) \ge f(x) + g^\top (y-x) \quad \forall y \in \mathrm{dom}(f)$$
The full set of valid subgradients at $x$ is called the **subdifferential** $\partial f(x)$.

- Where $f$ is differentiable: subgradient = gradient (unique).
- Where $f$ is not differentiable: the subdifferential is the **convex hull** of gradients of nearby points.

**Update rule:**
$$w^{(t+1)} = w^{(t)} - \gamma^{(t)} \cdot g, \quad g \in \partial L(w^{(t)})$$

**Pros:** Extends gradient-based optimization to a much larger class of (convex but non-smooth) ML losses, e.g., L1-regularized problems.

**Cons:**
- **Not a descent method** — the loss can actually go *up* at some iterations, unlike gradient descent.
- **Line search cannot be used** (since progress isn't monotonic).
- Convergence is *much slower* than plain gradient descent.
- The size (norm) of the subgradient does **not** tell you how close you are to the optimum (unlike the gradient, which shrinks to zero near a smooth minimum).

**Step size:** A common choice is a diminishing step size, e.g. $\gamma^{(t)} = 1/\sqrt{t}$.

**When to use it:** When your convex loss has "kinks" / is not differentiable everywhere, e.g. L1 norm terms, hinge loss, etc.

---

## Quick Comparison Table

| Method | Needs differentiability? | Line search OK? | Convergence guarantee (convex case) | Speed |
|---|---|---|---|---|
| Gradient Descent | Yes | Yes | Global minimum | Fast(er) |
| Subgradient Method | No (works with subgradients) | No | Global minimum | Slow |

---

## Points to Watch Out for in a Master's-Level ML Exam

1. **Don't confuse local and global minima** — know the exact epsilon-ball definition of a local minimizer, and be able to state (and possibly prove/argue) the theorem that convexity guarantees local = global.
2. **Know the convexity definition itself** (the $\lambda f(x) + (1-\lambda) f(y)$ inequality) — you may be asked to prove a specific function convex directly from the definition, not just via the Hessian.
3. **Hessian/PSD test:** Be able to compute a Hessian, find/argue about its eigenvalues, and conclude PSD ⟺ convex. Also know PD vs. PSD (strict vs. non-strict inequality) and what that implies (uniqueness of minimizer).
4. **Practice matrix calculus derivations** by hand: $\nabla(w^\top A w)$, $\nabla \|Xw-y\|^2$, and similar — know the denominator vs numerator layout distinction and be consistent.
5. **Know why least-squares regression is convex** ($X^\top X \succeq 0$ always) — this is a classic proof/derivation to be asked to reproduce.
6. **Understand eigenvalues as "curvature along a direction"** — this connects linear algebra (eigen-decomposition) to optimization behavior (why ill-conditioned problems cause slow/zig-zagging convergence).
7. **Subgradient method subtleties are a common trick question:** remember it is *not* a descent method, cannot use line search, and the subgradient's magnitude does not indicate proximity to the optimum — these are the classic "gotchas."
8. **Step size / learning rate schedules** matter: know that gradient descent needs an "appropriately small" step for guaranteed convergence, and that subgradient methods typically need a *diminishing* step size (like $1/\sqrt{t}$) rather than a fixed one.
9. Be ready to connect concepts across the slide deck: e.g., "Why does gradient descent always find the true minimum for least squares?" requires chaining together (a) the gradient/Hessian computation, (b) the PSD argument, and (c) the convexity theorem.
