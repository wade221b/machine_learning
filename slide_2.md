# Master's Degree Practice Exam: Optimization for Machine Learning (Lecture 2)

## Part 1: The Test

### Topic 1: Gradient Descent Fundamentals
**Question 1: Gradient Properties (2 Points)**
In the context of optimizing a loss function $L(w)$, which of the following are true regarding the gradient $\nabla L$?
- [ ] The gradient $\nabla L(w)$ always points towards the global minimum of the function.
- [ ] The gradient $\nabla L(w)$ points in the direction of the greatest local increase of the function.
- [ ] At a critical point, the magnitude of the gradient is zero.
- [ ] For a function $L: \mathbb{R}^d \to \mathbb{R}$, the gradient is a vector in $\mathbb{R}^d$ composed of partial derivatives.

**Question 2: The Gradient Descent Update Rule (2 Points)**
Consider the update $w^{(t+1)} = w^{(t)} - \gamma^{(t)} \nabla L(w^{(t)})$. Which statements accurately describe the components?
- [ ] $\gamma^{(t)}$ represents the learning rate or step length.
- [ ] The term $-\nabla L(w^{(t)})$ ensures we move in the direction of steepest descent.
- [ ] If the learning rate $\gamma^{(t)}$ is too large, the algorithm may overshoot the minimum and diverge.
- [ ] Gradient descent is only applicable if the Hessian of the function is known.

### Topic 2: Matrix Calculus & Layouts
**Question 3: Derivatives of Quadratic Forms (3 Points)**
Given a weight vector $w \in \mathbb{R}^d$ and a symmetric matrix $A \in \mathbb{R}^{d \times d}$, what is the gradient of the quadratic form $L(w) = w^\top A w$?
- [ ] $\nabla L(w) = Aw$
- [ ] $\nabla L(w) = 2Aw$
- [ ] $\nabla L(w) = Aw + A^\top w$
- [ ] $\nabla L(w) = w^\top A$

**Question 4: Derivative Layout Conventions (2 Points)**
When computing the derivative of a vector-valued function $L(w) : \mathbb{R}^d \to \mathbb{R}^m$, which statements reflect the slides' discussion on conventions?
- [ ] The denominator layout and numerator layout are both valid but result in transposed matrices.
- [ ] In the denominator layout, the resulting matrix has dimensions $d \times m$.
- [ ] The slides recommend matrixcalculus.org as a tool for verifying results.
- [ ] Tensors make multi-dimensional calculus more difficult to standardize.

### Topic 3: Convexity Theory
**Question 5: Convex Sets (2 Points)**
According to the mathematical definition of a convex set $C$, which conditions must be met?
- [ ] For any $x, y \in C$, the midpoint $(x+y)/2$ must be in $C$.
- [ ] For any $x, y \in C$ and $\lambda \in$, the point $\lambda x + (1-\lambda)y$ must be in $C$.
- [ ] The set must contain its own boundary.
- [ ] A set with a "dent" or "cutout" can still be convex if the dent is smooth.

**Question 6: Convex Functions (3 Points)**
A function $f: \mathbb{R}^d \to \mathbb{R}$ is convex if:
- [ ] Its domain is a convex set.
- [ ] The line segment connecting any two points on the graph of the function lies on or above the graph.
- [ ] $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda) f(y)$ for all $\lambda \in$.
- [ ] It is twice differentiable everywhere in its domain.

**Question 7: Positive Semidefinite (PSD) Matrices (3 Points)**
A symmetric matrix $M$ is positive semidefinite (psd) if and only if:
- [ ] $x^\top M x \ge 0$ for all vectors $x$.
- [ ] All its eigenvalues are strictly positive ($> 0$).
- [ ] All its eigenvalues are non-negative ($\ge 0$).
- [ ] The matrix represents a function that has a saddle point.

**Question 8: The Hessian and Convexity (3 Points)**
For a twice-differentiable function $f$, what does the Hessian $\nabla^2 f(x)$ reveal about convexity?
- [ ] If $\nabla^2 f(x) \succeq 0$ (psd) at a single point $x$, the function is globally convex.
- [ ] The function is convex if and only if $\nabla^2 f(x) \succeq 0$ for all $x$ in a convex domain.
- [ ] The second-order Taylor approximation $f(x + \Delta x) \approx f(x) + \nabla f(x)^\top \Delta x + \frac{1}{2} \Delta x^\top \nabla^2 f(x) \Delta x$ is exact for quadratic functions.
- [ ] If the Hessian has even one negative eigenvalue, the function is not convex.

### Topic 4: Convergence & Line Search
**Question 9: Optimality in Convex Functions (2 Points)**
If $f$ is convex and differentiable, which of the following are guaranteed?
- [ ] Every local minimum is also a global minimum.
- [ ] Every critical point where $\nabla f(x) = 0$ is a global minimum.
- [ ] Gradient descent will always find the global minimum in a single step.
- [ ] The function cannot have any saddle points.

**Question 10: Step Size Rules & Line Search (3 Points)**
Regarding the selection of the learning rate $\gamma^{(t)}$:
- [ ] Exact Line Search finds the step length that minimizes the function along the search ray.
- [ ] Backtracking Line Search uses parameters $\alpha$ and $\beta$ to iteratively shrink the step size until a sufficient decrease condition is met.
- [ ] In Backtracking Line Search, the condition $f(x + t\Delta x) \le f(x) + \alpha t \nabla f(x)^\top \Delta x$ must be satisfied.
- [ ] Backtracking Line Search is more computationally expensive than Exact Line Search for all possible functions.

### Topic 5: Non-Differentiable Optimization
**Question 11: Subgradients and Subdifferentials (3 Points)**
When dealing with a non-differentiable convex function $f$ (like the $\ell_1$-norm):
- [ ] A subgradient $g$ at $x$ satisfies $f(y) \ge f(x) + g^\top(y-x)$ for all $y$.
- [ ] The subdifferential $\partial f(x)$ is the set of all subgradients at point $x$.
- [ ] At points where the function is differentiable, the subdifferential contains only the gradient.
- [ ] Subgradients exist only for points where the function is continuous but not differentiable.

**Question 12: The Subgradient Method (2 Points)**
How does the subgradient method differ from standard Gradient Descent?
- [ ] It is a descent method, meaning the objective value decreases at every step.
- [ ] It can be used with line search strategies to find the optimal step size.
- [ ] It requires a specific learning rate schedule, such as $\gamma^{(t)} = 1/\sqrt{t}$, to ensure convergence.
- [ ] The norm of the subgradient indicates exactly how close the algorithm is to the optimal solution.

### Topic 6: Application to Least Squares
**Question 13: Least Squares Optimization (2 Points)**
Consider the least squares loss $L(w) = \frac{1}{2} \|Xw - y\|_2^2$.
- [ ] The gradient is $\nabla L(w) = X^\top (Xw - y)$.
- [ ] The Hessian is $\nabla^2 L(w) = X^\top X$.
- [ ] Since $X^\top X$ is always positive semidefinite, the loss function is convex.
- [ ] Gradient descent is guaranteed to find the global minimum for this specific loss.

---

## Part 2: Answer Key & Explanations

### Topic 1
**Question 1: [ ] Opt 1, [X] Opt 2, [X] Opt 3, [X] Opt 4**
*   **Opt 1 Incorrect:** The gradient points in the direction of greatest *ascent* locally; it does not necessarily point toward the global minimum in non-convex or complex landscapes.
*   **Opt 2 Correct:** The slides state the gradient points in the "direction of greatest ascent".
*   **Opt 3 Correct:** A critical point is defined as a point where $\nabla f(x) = 0$.
*   **Opt 4 Correct:** The definition of $\nabla L$ shows it is a vector of partial derivatives in $\mathbb{R}^d$.

**Question 2: [X] Opt 1, [X] Opt 2, [X] Opt 3, [ ] Opt 4**
*   **Opt 1 Correct:** $\gamma^{(t)}$ is explicitly defined as the "step length (aka learning rate)".
*   **Opt 2 Correct:** We walk a "small step negative direction of gradient" to descend.
*   **Opt 3 Correct:** Choosing the step length is critical; the slides ask "How to pick step length?" and note that theory is needed to avoid issues.
*   **Opt 4 Incorrect:** GD only requires the gradient; knowing the Hessian (second derivative) is associated with more advanced methods like Newton's method, not standard GD.

### Topic 2
**Question 3: [ ] Opt 1, [X] Opt 2, [X] Opt 3, [ ] Opt 4**
*   **Opt 1 Incorrect:** This misses the transpose term.
*   **Opt 2 Correct:** For symmetric $A$, $Aw + A^\top w$ simplifies to $2Aw$.
*   **Opt 3 Correct:** The general matrix calculus identity provided is $\nabla(w^\top Aw) = Aw + A^\top w$.
*   **Opt 4 Incorrect:** This is a row vector; the gradient is defined as a column vector of partials.

**Question 4: [X] Opt 1, [X] Opt 2, [X] Opt 3, [ ] Opt 4**
*   **Opt 1 Correct:** The slides note that both layouts exist and one is the transpose of the other.
*   **Opt 2 Correct:** The definition of $L'$ in denominator layout for $L: \mathbb{R}^d \to \mathbb{R}^m$ is a $d \times m$ matrix.
*   **Opt 3 Correct:** The slides suggest "check out www.MatrixCalculus.org" but warn about layout conventions.
*   **Opt 4 Incorrect:** The slides state "everything will be easier when we talk about tensors".

### Topic 3
**Question 5: [ ] Opt 1, [X] Opt 2, [ ] Opt 3, [ ] Opt 4**
*   **Opt 1 Incorrect:** While the midpoint is included in the definition (where $\lambda=0.5$), the definition must hold for *all* points on the segment, not just the midpoint.
*   **Opt 2 Correct:** This is the verbatim mathematical definition of a convex set provided in the slides.
*   **Opt 3 Incorrect:** Convexity does not require a set to be closed (containing its boundary), though the domain for derivatives is often specified as open.
*   **Opt 4 Incorrect:** Any "dent" allows a line segment to exit the set, violating the definition.

**Question 6: [X] Opt 1, [X] Opt 2, [X] Opt 3, [ ] Opt 4**
*   **Opt 1 Correct:** Part of the formal definition is that "dom(f) is a convex set".
*   **Opt 2 Correct:** This is the visual/intuitive definition of a convex function.
*   **Opt 3 Correct:** This is the formal mathematical inequality for a convex function.
*   **Opt 4 Incorrect:** Convexity does not require differentiability (e.g., the absolute value function is convex but not differentiable at zero).

**Question 7: [X] Opt 1, [ ] Opt 2, [X] Opt 3, [ ] Opt 4**
*   **Opt 1 Correct:** This is the formal definition of psd ($A \succeq 0$).
*   **Opt 2 Incorrect:** This is the definition for positive *definite* ($A \succ 0$).
*   **Opt 3 Correct:** The theorem states a matrix is psd iff all eigenvalues are $\ge 0$.
*   **Opt 4 Incorrect:** If the Hessian is psd, the point is a minimum; a saddle point requires both positive and negative eigenvalues.

**Question 8: [ ] Opt 1, [X] Opt 2, [X] Opt 3, [X] Opt 4**
*   **Opt 1 Incorrect:** The Hessian must be psd "at every point x" in the domain to guarantee global convexity.
*   **Opt 2 Correct:** The theorem explicitly states $f$ is convex iff $\text{dom}(f)$ is convex and $\nabla^2 f(x) \succeq 0$ for all $x$.
*   **Opt 3 Correct:** For a quadratic, the higher-order terms in a Taylor series are zero, making the second-order approximation exact.
*   **Opt 4 Correct:** Since convexity requires *all* eigenvalues to be $\ge 0$, a single negative eigenvalue violates the condition.

### Topic 4
**Question 9: [X] Opt 1, [X] Opt 2, [ ] Opt 3, [ ] Opt 4**
*   **Opt 1 Correct:** A fundamental theorem states every local minimum of a convex function is a global minimum.
*   **Opt 2 Correct:** For convex, differentiable functions, "Every critical point is also a global minimum".
*   **Opt 3 Incorrect:** GD is an iterative process; convergence speed depends on the learning rate and function properties.
*   **Opt 4 Incorrect:** A convex function can have a region of zero curvature (flat), but a classic "saddle point" (descending in one direction, ascending in another) implies negative eigenvalues, which contradicts convexity.

**Question 10: [X] Opt 1, [X] Opt 2, [X] Opt 3, [ ] Opt 4**
*   **Opt 1 Correct:** Exact Line Search minimizes $f$ along the ray $\{x + s \Delta x\}$.
*   **Opt 2 Correct:** The algorithm iteratively sets $t = \beta t$ until the condition is met.
*   **Opt 3 Correct:** This is the "sufficient decrease" condition shown in the algorithm.
*   **Opt 4 Incorrect:** Exact Line Search is often much more expensive because it requires solving a separate optimization problem in every step, whereas Backtracking uses simple evaluations.

### Topic 5
**Question 11: [X] Opt 1, [X] Opt 2, [X] Opt 3, [ ] Opt 4**
*   **Opt 1 Correct:** This is the formal definition of a subgradient.
*   **Opt 2 Correct:** $\partial f(x)$ is defined as the "set of subgradients of f at x".
*   **Opt 3 Correct:** The slides state "subgradient = gradient when f is differentiable".
*   **Opt 4 Incorrect:** Subgradients can exist anywhere for convex functions, and they equal the gradient if it exists; they aren't "only" for non-differentiable points.

**Question 12: [ ] Opt 1, [ ] Opt 2, [X] Opt 3, [ ] Opt 4**
*   **Opt 1 Incorrect:** It is "not a descent method," meaning the function value can increase in some iterations.
*   **Opt 2 Incorrect:** "Line search strategies cannot be used!!!" for the subgradient method.
*   **Opt 3 Correct:** The slides specifically mention $\gamma^{(t)} = 1/\sqrt{t}$ as a step size that "will do the job".
*   **Opt 4 Incorrect:** Unlike GD, the "norm of subgradient does not tell you how close you are to the optimal solution".

### Topic 6
**Question 13: [X] Opt 1, [X] Opt 2, [X] Opt 3, [X] Opt 4**
*   **Opt 1 Correct:** Derived as $\nabla L(w) = X^\top(Xw - y)$.
*   **Opt 2 Correct:** The Hessian is given as $X^\top X$.
*   **Opt 3 Correct:** $X^\top X$ is noted as "symmetric and psd," proving the loss is convex.
*   **Opt 4 Correct:** Because it is convex and differentiable, "gradient descent always finds the true global minimum".
