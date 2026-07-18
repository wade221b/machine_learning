# Optimization for Machine Learning Cheatsheet

This cheatsheet provides a structured overview of the optimization methods and theoretical concepts for Machine Learning based on the lecture material from the Universität Hamburg.

---

## 1. General Machine Learning Optimization Problem

*   **Explanation**: In Machine Learning, training a model often boils down to minimizing an empirical loss function $L(w)$ over a vector of parameters $w \in \mathbb{R}^d$. While setting the analytical gradient to zero ($\nabla L(w) = 0$) works for simple models like standard linear regression, it is frequently impossible or computationally intractable for complex or non-linear systems. Hence, numerical optimization methods are required.
*   **Formula**: 
    $$\min_{w \in \mathbb{R}^d} L(w)$$
*   **When It Should Be Used**: Always, implicitly or explicitly, whenever a machine learning model is being trained using standard loss-minimization paradigms (e.g., training neural networks, support vector machines, or logistic regression models).
*   **Pros & Cons**:
    *   **Pros**: Provides a unified framework to formulate and learn optimal parameters directly from empirical data.
    *   **Cons**: Finding the true global minimum can be computationally expensive or mathematically impossible if the objective surface contains many local minima, saddle points, or non-differentiable regions.

---

## 2. Gradient Descent (GD)

*   **Explanation**: A first-order iterative optimization algorithm used to find the local minimum of a differentiable function. The algorithm computes the gradient vector $\nabla L(w)$ at the current position $w^{(t)}$, which points in the direction of steepest ascent, and takes a small step in the opposite (negative) direction to locate a lower function value.
*   **Formula**: 
    $$w^{(t+1)} = w^{(t)} - \gamma^{(t)} \cdot \nabla L(w^{(t)})$$
    *Where $\gamma^{(t)}$ denotes the step length / learning rate at iteration $t$.*
*   **When It Should Be Used**: When the loss function $L(w)$ is smooth and continuously differentiable, and full-batch computing of the gradient over the dataset is computationally viable.
*   **Pros & Cons**:
    *   **Pros**: Conceptually simple, easy to implement, and guaranteed to converge to the global minimum if the function is convex and the step size is chosen appropriately.
    *   **Cons**: Can become very slow on large datasets (requires computing gradients over all samples). It can also get trapped in local minima or saddle points for non-convex loss surfaces, or exhibit oscillating behavior if the learning rate is too high.

---

## 3. Mathematical Convexity (Sets & Functions)

*   **Explanation**: Convexity is a fundamental geometric property. A set is convex if any line segment connecting two points inside the set stays completely within it. A function is convex if the line segment connecting any two points on its graph sits above or on the graph itself. For a twice-differentiable function, convexity can be confirmed if its Hessian matrix $\nabla^2 f(x)$ is positive semidefinite (psd) across its domain.
*   **Formula**: 
    *   **Convex Set**: $\forall x, y \in C, \lambda \in [0, 1] \implies \lambda x + (1 - \lambda)y \in C$
    *   **Convex Function**: $f(\lambda x + (1 - \lambda)y) \le \lambda f(x) + (1 - \lambda)f(y)$
    *   **Second-Order Condition**: $\nabla^2 f(x) \succeq 0 \quad (\forall x \in \text{dom}(f))$
*   **When It Should Be Used**: Used primarily during theoretical model design and analysis to provide rigid mathematical convergence guarantees. Many robust ML estimators (e.g., Least Squares Regression, Logistic Regression, Linear SVMs) are explicitly framed as convex problems.
*   **Pros & Cons**:
    *   **Pros**: **Critical Theorem**: Every local minimum of a convex function is also a global minimum, and every critical point ($\nabla f(x) = 0$) is a global minimum. This removes the risk of getting trapped in poor local solutions.
    *   **Cons**: Many highly powerful modern ML structures, such as deep neural networks, are inherently non-convex, meaning convexity analysis cannot be applied directly.

---

## 4. Line Searches (Step Size Selection)

*   **Explanation**: Methods used to adaptively pick an appropriate step length $\gamma^{(t)}$ at each iteration instead of using a constant hyperparameter value. 
    *   **Exact Line Search** computes the absolute optimal step length along the search direction. 
    *   **Backtracking Line Search** is an algorithmic heuristic that starts with a default step size and iteratively shrinks it by a factor $\beta$ until a sufficient decrease condition (Armijo condition) is satisfied.
*   **Formula (Backtracking Line Search)**:
    *   *Input parameters*: $\alpha \in (0, 0.5)$, $\beta \in (0, 1)$, initial $t = 1$.
    *   *Loop condition*: While $f(x + t\Delta x) > f(x) + \alpha t \nabla f(x)^\top \Delta x$, update $t \leftarrow \beta t$.
*   **When It Should Be Used**: When standard gradient descent exhibits poor convergence, oscillations, or divergence due to poorly chosen static learning rates.
*   **Pros & Cons**:
    *   **Pros**: Mathematically guarantees stable descent at each step, preventing divergence and accelerating convergence near steep valleys.
    *   **Cons**: Requires evaluating the loss function multiple times per gradient step, increasing the computational overhead per iteration. **Completely fails and cannot be used with non-differentiable functions (e.g., subgradient methods).**

---

## 5. Subgradient Method

*   **Explanation**: An extension of gradient descent designed for non-differentiable convex functions (e.g., optimization involving absolute values like $L_1$-regularization / Lasso: $\|w\|_1$). When a function has a sharp "kink", a unique gradient does not exist. The subgradient method substitutes the gradient vector with any subgradient $g$ taken from the subdifferential set $\partial f(x)$ (which represents the convex hull of the limits of gradients of nearby smooth points).
*   **Formula**: 
    $$w^{(t+1)} = w^{(t)} - \gamma^{(t)} \cdot g$$
    *Where $g \in \partial f(w^{(t)})$ is a subgradient, and step length is often scheduled dynamically, e.g., $\gamma^{(t)} = \frac{1}{\sqrt{t}}$.*
*   **When It Should Be Used**: When optimizing non-differentiable convex functions commonly found in sparse modeling or robust loss metrics (e.g., $L_1$ norms, hinge loss).
*   **Pros & Cons**:
    *   **Pros**: Robustly handles non-smooth convex objectives and retains global convergence guarantees when paired with an appropriate diminishing learning rate schedule.
    *   **Cons**: It is **not** a strict descent method (the function value can temporarily increase between iterations), meaning standard line searches fail. Convergence guarantees are significantly slower than standard gradient descent, and the norm of the subgradient does not indicate proximity to the optimal solution.
