# EXAM: Advanced Optimization for Machine Learning (Select ALL That Apply)

## SECTION I: Multiple Choice Multiple Answer (MCMA)
*Instructions: For each question, select **ALL** options that are mathematically correct. A question may have one, two, three, or all four options correct. Distractors are deeply tied to edge cases in the slides.*

**1. Convexity & Global Minima (Slides 12-16)**
Let $f: \mathbb{R}^d \rightarrow \mathbb{R}$ be a differentiable function defined on an open, convex domain. Which of the following statements are ALWAYS true?
- [ ] A) If $f$ is strictly convex, and there exists a point $x^*$ where $\nabla f(x^*) = 0$, then $x^*$ is the unique global minimum of $f$.
- [ ] B) If $f$ is convex, the set of all global minima of $f$ is guaranteed to be exactly one point. 
- [ ] C) If $f$ is convex, and $x_1, x_2$ are both global minimizers of $f$, then any point $x = \lambda x_1 + (1-\lambda)x_2$ for $\lambda \in [0,1]$ is also a global minimizer.
- [ ] D) If every critical point ($\nabla f(x) = 0$) of $f$ is a global minimum, then $f$ must be a convex function.

**2. Hessians, Eigenvalues, and GD Geometry (Slides 21, 24-27)**
Consider a twice-differentiable function $f(w) = \frac{1}{2} w^\top H w + b^\top w$. Let $\lambda_{max}$ and $\lambda_{min}$ be the largest and smallest eigenvalues of the Hessian $H$. Which of the following statements are true?
- [ ] A) $f(w)$ is a convex function if and only if all eigenvalues of $H$ are $\ge 0$.
- [ ] B) If you evaluate the Hessian of an arbitrary non-quadratic function at a specific point $x$ and find $\lambda_i > 0$ for all $i$, you have proven the function is globally convex.
- [ ] C) If $H$ is symmetric and Positive Semi-Definite (PSD) but has at least one eigenvalue $\lambda_i = 0$, the function cannot have a *unique* global minimum.
- [ ] D) Standard Gradient Descent will travel fastest along the eigenvector corresponding to $\lambda_{min}$ and slowest along the eigenvector corresponding to $\lambda_{max}$.

**3. Subgradients & Non-Differentiable Functions (Slides 32-40)**
Let $f(w)$ be a non-differentiable convex function. You are optimizing it using the Subgradient Method: $w^{(t+1)} = w^{(t)} - \gamma^{(t)} g$, where $g \in \partial f(w^{(t)})$. Which of the following are true?
- [ ] A) The subgradient method guarantees that $f(w^{(t+1)}) \le f(w^{(t)})$ for every single iteration, provided the learning rate $\gamma^{(t)}$ is chosen to be sufficiently small.
- [ ] B) Geometrically, for $f(w) = |w|$, the subdifferential at $w=0$ ($\partial f(0)$) represents the slopes of all possible lines that pass through the origin and strictly lower-bound the function everywhere.
- [ ] C) If the zero vector is an element of the subdifferential at point $x$ (i.e., $0 \in \partial f(x)$), then $x$ is a global minimum of $f$.
- [ ] D) A Backtracking Line Search (Algorithm 1 in the slides) is computationally heavy but will still mathematically guarantee a faster convergence for the subgradient method than a fixed schedule like $\gamma^{(t)} = \frac{1}{\sqrt{t}}$.

**4. Matrix Calculus & Least Squares (Slides 7-8, 30)**
Consider the Least Squares regression loss: $L(w) = \frac{1}{2}\|Xw - y\|_2^2$. Assume denominator layout. Which of the following statements are mathematically accurate?
- [ ] A) The Hessian of this loss function is $X^\top X$, which is always guaranteed to be a symmetric and positive semi-definite matrix, proving the problem is convex.
- [ ] B) The gradient of the loss is $\nabla L(w) = X^\top(Xw - y)$. If you change your convention to *numerator* layout, the gradient expression becomes $(Xw - y)^\top X$.
- [ ] C) If the columns of $X$ are perfectly collinear (linearly dependent), Gradient Descent will fail to converge to a global minimum because the loss function is no longer convex.
- [ ] D) By the rules of matrix calculus, $\nabla (w^\top A w) = 2Aw$ is ONLY true if $A$ is a symmetric matrix. 

---

## SECTION II: Synthesis Questions (Open-Ended)
*These require deep conceptual reasoning combining multiple slides.*

**Open-Ended 1: The Geometry of Line Search Failure**
Slide 34 emphatically states that for the Subgradient Method: *"line search strategies cannot be used!!!"* Slide 36 shows a contour plot titled "Line Search Fails" with a V-shaped non-differentiable valley.
Use mathematical and geometric reasoning to explain exactly *why* exact line search ($t = \text{argmin}_{s \ge 0} f(x + s \cdot \Delta x)$) fails in this scenario. What specifically happens to the direction vector $\Delta x$ (the negative subgradient) relative to the contour lines that causes the line search to break?

**Open-Ended 2: Positive Definite vs. Positive Semi-Definite in ML**
Slide 27 defines convexity using a Positive Semi-Definite (PSD) Hessian ($\nabla^2 f(x) \succeq 0$). Slide 30 applies this to Least Squares, noting $\nabla^2 L(w) = X^\top X$ is PSD.
Assume you are running Gradient Descent on Least Squares, but your data matrix $X$ results in an $X^\top X$ that is **PSD but NOT Positive Definite (PD)** (i.e., it has at least one eigenvalue of exactly $0$). 
1. Geometrically, what does the loss landscape look like?
2. Does a global minimum still exist?
3. Will standard Gradient Descent still converge to a global minimum, or will it fail? Explain why.

<br><br><br><br><br><br><br><br>

---
---

# PROFESSOR'S ANSWER KEY & GRADING RUBRIC

## Section I: MCQ Answers

**1. Correct Answers: A, C**
*   **A is True:** If a function is *strictly* convex, it has at most one global minimum. Since it's differentiable and domain is open, $\nabla f=0$ guarantees it is that unique minimum.
*   **B is False:** A convex function can have infinitely many global minima (e.g., a function with a flat bottom, like a plateau). It is not restricted to exactly one point unless it is *strictly* convex.
*   **C is True:** By the definition of a convex set/function (Slides 12-13), the set of global minimizers of a convex function is itself a convex set. Any linear combination of two minima is also a minimum.
*   **D is False:** This is a logical fallacy (converse error). $f(x) = x^3$ has a critical point at $x=0$, which is not a global minimum, so the premise fails. However, more formally, just because all critical points are global minima doesn't automatically mean the function satisfies the geometric inequality of convexity everywhere.

**2. Correct Answers: A, C**
*   **A is True:** Slide 27 explicitly states $f$ is convex iff $\forall x : \nabla^2 f(x) \succeq 0$ (which by Slide 21 means all eigenvalues $\ge 0$).
*   **B is False:** Finding strictly positive eigenvalues at *one specific point* $x$ only proves the function is *locally* strictly convex at that point, not globally.
*   **C is True:** An eigenvalue of $0$ means the function is perfectly flat (linear/constant) along the direction of that eigenvector. Therefore, it forms a "valley" or "trough" of minima, meaning the minimum is not unique. 
*   **D is False:** It is the exact opposite. Gradient descent moves *fastest* (steepest) along the eigenvector corresponding to the *largest* eigenvalue ($\lambda_{max}$), often causing overshoot/zigzagging, and crawls slowly along $\lambda_{min}$. (Slides 24-26).

**3. Correct Answers: B, C**
*   **A is False:** Slide 34 explicitly states: "is not a descent method, i.e., function value does not decrease in every iteration." A subgradient step can temporarily increase the loss, even with a small learning rate.
*   **B is True:** Slide 38 defines the subgradient mathematically as $f(y) \ge f(x) + g^\top(y - x)$. For $f(w)=|w|$ at $x=0$, this means $|y| \ge g \cdot y$. This holds true for any slope $g \in [-1, 1]$. Geometrically, these are lines through the origin that sit under the absolute value "V".
*   **C is True:** If $0 \in \partial f(x)$, plugging $g=0$ into the subgradient definition yields $f(y) \ge f(x) + 0$, which means $f(y) \ge f(x)$ for all $y$. This is the exact definition of a global minimum!
*   **D is False:** Slide 34 screams "line search strategies cannot be used!!!". Backtracking line search requires a differentiable function (Algorithm 1 uses $\nabla f(x)$). You cannot use it on subgradients.

**4. Correct Answers: A, B, D**
*   **A is True:** Slide 30 confirms $X^\top X$ is symmetric and PSD, hence Least Squares is always convex.
*   **B is True:** Switching to numerator layout transposes everything. $(X^\top(Xw - y))^\top = (Xw - y)^\top X$. 
*   **C is False:** If columns are collinear, $X^\top X$ has an eigenvalue of 0. It is still Positive Semi-Definite (just not Positive Definite). Therefore, the function is STILL convex (a flat valley). Gradient Descent will still happily converge to *one* of the infinitely many global minima.
*   **D is True:** Slide 55 proves $\nabla (w^\top A w) = Aw + A^\top w$. This only simplifies to $2Aw$ if $A = A^\top$ (symmetric). 

---

## Section II: Open-Ended Answers & Rubric

**Open-Ended 1: The Geometry of Line Search Failure**
*Grading Rubric:*
*   **Mathematical flaw:** A subgradient $g$ guarantees a global lower bound, but $-g$ is **not strictly a descent direction**. In standard GD, moving slightly in the $-\nabla f$ direction guarantees the function decreases. In subgradient methods, $-g$ might point slightly "uphill" relative to the local contour. 
*   **Geometric explanation (Slide 36):** Imagine a sharply kinked, non-differentiable valley (like a sharp zig-zag contour). If you are on the side of the valley, the negative subgradient might point *across* the valley rather than down the center of it. 
*   **Why Exact Line Search fails:** Exact line search looks for the minimum along the ray defined by $-g$. Because $-g$ might not be a descent direction, the function might strictly *increase* along that ray. In this case, the line search will return a step size of $t=0$, causing the algorithm to get stuck permanently, even though it is not at the global minimum. 

**Open-Ended 2: PSD vs. PD in Least Squares**
*Grading Rubric:*
*   **1. Loss Landscape (Geometry):** Because there is an eigenvalue of 0, the quadratic bowl is infinitely flat along the direction of the eigenvector associated with $\lambda=0$. Instead of a 3D bowl with a single bottom point, it looks like a half-pipe or a trough. 
*   **2. Existence of Global Minimum:** Yes, a global minimum still exists. In fact, infinitely many exist. Any point along the bottom of the "trough" yields the exact same minimum loss.
*   **3. Gradient Descent Behavior:** Gradient Descent **will still converge**. Because the function is convex (PSD), GD will roll down the sides of the trough. It will simply stop at whichever point on the bottom of the trough it hits first (this is determined by the initialization point and the span of the non-zero eigenvectors). It does not fail, it just selects one of the infinite valid solutions.
