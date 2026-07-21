# EXAM: Optimization for Machine Learning

## SECTION I: Multiple Choice Questions (Select the ONE best answer)
*Note: Distractors are designed to catch common misconceptions.*

**1. Matrix Calculus & Layouts**
You are minimizing a regularized loss function $L(w) = w^\top A w + b^\top w$, where $w, b \in \mathbb{R}^d$ and $A \in \mathbb{R}^{d \times d}$. However, unlike standard Least Squares, you are *not* guaranteed that matrix $A$ is symmetric. 
Assume denominator layout (as is standard in the slides). What is the gradient $\nabla L(w)$?
- [ ] A) $2Aw + b$
- [ ] B) $Aw + A^\top w + b$
- [ ] C) $2A^\top w + b^\top$
- [ ] D) $A^\top w + Aw + b^\top$

**2. Geometry of Gradient Descent (Eigenvalues)**
Recall the slide showing how a function behaves along eigenvectors like $1\cdot x^2$ and $3\cdot x^2$. Suppose you have a convex quadratic function $f(w) = \frac{1}{2} w^\top H w$. The Hessian $H$ has eigenvalues $\lambda_1 = 100$ and $\lambda_2 = 0.01$. You initialize gradient descent at a random point. Which of the following best describes the behavior of standard gradient descent with a fixed learning rate?
- [ ] A) It will converge extremely quickly because the high eigenvalue ($\lambda_1=100$) allows for large steps toward the minimum.
- [ ] B) It will suffer from severe zig-zagging because a learning rate small enough to avoid divergence along the $\lambda_1$ eigenvector will result in painfully slow progress along the $\lambda_2$ eigenvector.
- [ ] C) It will converge in exactly two steps if exact line search is used, regardless of the eigenvalues.
- [ ] D) It will fail to converge to a global minimum because the ratio of eigenvalues implies the matrix is not positive semi-definite.

**3. Convexity Edge Cases**
By the definitions provided in your slides, which of the following functions $f: \mathbb{R} \rightarrow \mathbb{R}$ is strictly convex, yet gradient descent (even with an infinitely small, perfect learning rate) will *never* find a global minimum?
- [ ] A) $f(x) = \max(0, x)$
- [ ] B) $f(x) = x^3$
- [ ] C) $f(x) = e^{-x}$
- [ ] D) $f(x) = |x|$

**4. Detecting Convexity via the Hessian**
You are given a twice-differentiable function $f(w_1, w_2)$. You compute the Hessian matrix evaluated at a specific point $x$ to be:

$$ \nabla^2 f(x) = \begin{pmatrix} 2 & 4 \\ 4 & 2 \end{pmatrix} $$

Based on this Hessian, what can you definitively conclude about the function's topography at point $x$?
- [ ] A) The function is locally strictly convex because the diagonal entries are positive ($2 > 0$).
- [ ] B) The function is locally convex because the trace of the matrix is positive ($4 > 0$).
- [ ] C) The function is locally a saddle point (neither convex nor concave) because the eigenvalues have mixed signs.
- [ ] D) The function is positive semi-definite (PSD) but not positive definite.

**5. Subgradients**
Let $f(w) = \max(-3w, 2w)$ for $w \in \mathbb{R}$. What is the subdifferential $\partial f(0)$?
- [ ] A) $[-2, 3]$
- [ ] B) $[-3, 2]$
- [ ] C) $\{0\}$
- [ ] D) $[-3, -2] \cup [2, 3]$

**6. The Subgradient Method & Line Search**
Your slides emphatically state with three exclamation marks: *"line search strategies cannot be used!!!"* for the subgradient method. Why is exact line search mathematically invalid for subgradients?
- [ ] A) Subgradients cannot be scaled by a scalar $t$; doing so breaks the convex hull property.
- [ ] B) The subgradient direction $-g$ is not strictly guaranteed to be a descent direction for every arbitrarily small step, meaning searching along that ray may not yield a smaller function value.
- [ ] C) The Taylor approximation requires a Hessian, which does not exist for non-differentiable functions.
- [ ] D) Backtracking line search will enter an infinite loop because $\alpha \in (0, 0.5)$ requires a differentiable gradient.

---

## SECTION II: Open-Ended / Synthesis Questions
*These questions require you to synthesize multiple slides and apply them to novel scenarios. Show your reasoning.*

**Open-Ended 1: The Geometry of the L1 Subdifferential**
Slide 40 states: "$\partial f(x)$ is the convex hull of the set $S(x)$" where:

$$ S(x) = \left\{ \lim_{n \to \infty} \nabla f(x_n) \;\middle|\; \lim_{n \to \infty} x_n = x \right\} $$

Consider the L1 norm in 2D space: $f(w_1, w_2) = |w_1| + |w_2|$.

**Part A:** Using the definition from the slide, rigorously derive the subdifferential at the origin $(0,0)$. *(Hint: approach the origin from the four quadrants).*

**Part B:** Imagine you are using the subgradient method to minimize this function, and you are currently at $(0,0)$. If you use a constant learning rate $\gamma = 0.1$, explain *geometrically* why the algorithm will fail to stay at the global minimum, and describe exactly what it will do instead. 

**Open-Ended 2: Broken Convexity Assumptions**
Slide 27 states: *"Suppose that $\text{dom}(f)$ is open... $f$ is convex if and only if $\text{dom}(f)$ is convex and we have $\forall x : \nabla^2 f(x) \succeq 0$."*

A fellow student argues: *"The domain being 'convex' and 'open' is just pedantic math jargon. If the Hessian is Positive Semi-Definite everywhere the function is defined, the function is convex, and Gradient Descent will find the global minimum."*

**Prove the student wrong.** 
Construct a specific, concrete mathematical example (or geometric scenario) where $\nabla^2 f(x) \succeq 0$ is true everywhere the function is defined, but because the domain is *not* a convex set, a local minimum exists that is *not* a global minimum, trapping Gradient Descent. 

<br><br><br><br><br><br><br><br>

---
---

# PROFESSOR'S ANSWER KEY & GRADING RUBRIC

## Section I: MCQ Answers

**1. B) $Aw + A^\top w + b$**
*Why:* Slide 55 proves that $\nabla (w^\top A w) = Aw + A^\top w$. Many students memorize $\nabla = 2Aw$, but that *only* holds when $A$ is symmetric (because $A = A^\top$). A tough professor will immediately check if you memorized the symmetric shortcut or if you know the fundamental matrix calculus rule. $\nabla(b^\top w) = b$.

**2. B) It will suffer from severe zig-zagging...**
*Why:* This tests your deep understanding of Slides 24-26. The ratio of the largest to smallest eigenvalue ($\lambda_{max}/\lambda_{min} = 10,000$) is the condition number. The gradient is overwhelmingly dominated by the $\lambda_1$ direction. If you set a learning rate large enough to move along the flat $\lambda_2$ valley, you will overshoot and diverge along the steep $\lambda_1$ walls. Therefore, you are forced to use a tiny learning rate, causing excruciatingly slow, zig-zag progress.

**3. C) $f(x) = e^{-x}$**
*Why:* Slide 15 talks about global minimizers, and Slide 13 defines convex functions. $e^{-x}$ has a strictly positive second derivative everywhere ($e^{-x} > 0$), so it is strictly convex. However, it only approaches $0$ as $x \to \infty$. It has no global minimum within the domain $\mathbb{R}$. Gradient descent will step endlessly toward infinity.

**4. C) The function is locally a saddle point...**
*Why:* Slide 21 states a symmetric matrix is PSD iff all its eigenvalues are $\ge 0$. Let's find the eigenvalues. The trace is $2+2=4$ (sum of eigenvalues) and the determinant is $(2 \times 2) - (4 \times 4) = 4 - 16 = -12$ (product of eigenvalues). Since the product of the eigenvalues is negative, one eigenvalue must be positive and the other negative. Therefore, it is neither PSD nor NSD; it is a saddle point.

**5. B) $[-3, 2]$**
*Why:* Slide 33 defines the subgradient as the "convex hull of gradients of nearby points." Approaching $0$ from the right ($w > 0$), the gradient is $2$. Approaching from the left ($w < 0$), the gradient is $-3$. The convex hull of $\{-3, 2\}$ is the continuous interval $[-3, 2]$.

**6. B) The subgradient direction $-g$ is not strictly guaranteed to be a descent direction...**
*Why:* Slide 34 explicitly notes "is not a descent method, i.e., function value does not decrease in every iteration." Because the negative subgradient might actually point slightly "uphill" relative to the level sets of the non-differentiable contour (often into the "V" shape of the contour, bouncing you across it), an exact line search looking to minimize $f$ along that ray will fail or return a step size of $0$. 

---

## Section II: Open-Ended Answers

**Open-Ended 1: L1 Subdifferential & Oscillation**

*Grading Rubric:*
*   **Part A (Derivation):** The student must identify the gradients in the four quadrants of $\mathbb{R}^2$ around $(0,0)$. 
    *   Quadrant 1 ($w_1>0, w_2>0$): $\nabla f = [1, 1]^\top$
    *   Quadrant 2 ($w_1<0, w_2>0$): $\nabla f = [-1, 1]^\top$
    *   Quadrant 3 ($w_1<0, w_2<0$): $\nabla f = [-1, -1]^\top$
    *   Quadrant 4 ($w_1>0, w_2<0$): $\nabla f = [1, -1]^\top$
    *   By Slide 40, the subdifferential is the convex hull of these four vectors. Geometrically, this forms a solid square in 2D space with vertices at $(1,1), (-1,1), (-1,-1), (1,-1)$. Thus, $\partial f(0,0) = \left\{ [g_1, g_2]^\top \mid g_1 \in [-1, 1], g_2 \in [-1, 1] \right\}$.
*   **Part B (Geometric failure):** If the algorithm reaches exactly $(0,0)$, it must pick a subgradient from the set above. Unless it magically picks exactly $[0,0]^\top$, it will pick a non-zero subgradient (e.g., $[1, 1]^\top$). With a constant learning rate of $\gamma = 0.1$, it will step to $w^{(t+1)} = [0,0]^\top - 0.1 \times [1,1]^\top = [-0.1, -0.1]^\top$. In the next step, it is in Quadrant 3, the gradient is definitively $[-1, -1]^\top$, and it steps back to $(0,0)$. It will oscillate endlessly between $(0,0)$ and the surrounding points. This demonstrates perfectly why Slide 37 requires a decaying learning rate like $\gamma^{(t)} = 1/\sqrt{t}$.

**Open-Ended 2: Broken Convexity Assumptions**

*Grading Rubric:*
*   **Core Concept:** The student must prove that a non-convex domain breaks the "local minima = global minima" guarantee of convex functions (Slide 15).
*   **Counter-Example:** Imagine a standard 3D bowl shape, $f(x,y) = x^2 + y^2$. The Hessian is:
    $$ \nabla^2 f(x) = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} $$
    which is strictly Positive Definite everywhere. 
*   **The Trap (Non-Convex Domain):** Now, define the domain to be a "C" shape or a horseshoe (e.g., cut out a massive circle from the middle of the domain so the origin $(0,0)$ is not included). 
*   If we place a point at the "tip" of one side of the horseshoe, any small step inside the valid domain goes UP in value. Therefore, it is mathematically a *local minimum* relative to its valid neighborhood. However, the true global minimum is at the other side of the horseshoe (or closer to the origin). Because the domain is not convex, the straight line between the local minimum and the global minimum passes outside the domain. Gradient Descent will get stuck at the local minimum on the boundary, unable to cross the "void" to reach the global minimum. The student is definitively proven wrong.
