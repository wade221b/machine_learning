
---

### Part 2: The Ultimate ML Convexity Test
Since this is for a Machine Learning exam, I have specifically chosen examples that represent **weights, regularization, probabilities, and constraints** used in real ML models (like SVMs, Neural Networks, and Logistic Regression).

**Instructions:** 
Read the math, translate it to a physical 2D shape in your head, and apply the line segment test. 
*Grab a piece of paper, write down Convex or Not Convex for 1 through 10, and then scroll down to check your answers!*

1.  **The Line:** $\{w \in \mathbb{R}^2 \mid w_1 + w_2 = 1\}$
2.  **The Halfspace:** $\{w \in \mathbb{R}^2 \mid w_1 + w_2 \le 1\}$
3.  **The L-Infinity Norm:** $\{w \in \mathbb{R}^2 \mid \max(|w_1|, |w_2|) \le 1\}$ *(Also written as $\|w\|_\infty \le 1$)*
4.  **The "Outside":** $\{w \in \mathbb{R}^2 \mid \|w\|_2 \ge 1\}$
5.  **The L0 "Norm" (Sparsity):** $\{w \in \mathbb{R}^2 \mid \text{Count of non-zero features} \le 1\}$
6.  **The Donut:** $\{w \in \mathbb{R}^2 \mid 1 \le \|w\|_2 \le 2\}$
7.  **The Probability Simplex:** $\{p \in \mathbb{R}^3 \mid p_1 + p_2 + p_3 = 1, \text{ and all } p \ge 0\}$
8.  **Positive Quadrant:** $\{w \in \mathbb{R}^2 \mid w_1 \ge 0 \text{ and } w_2 \ge 0\}$
9.  **Intersection:** A set formed by taking the overlapping region of two solid circles.
10. **Union:** A set formed by putting two separate solid circles together into one group.

***
**(STOP! Do not scroll until you have your answers!)**
***
<br><br><br><br><br><br>

### The Answers & Explanations

**1. $\{w \in \mathbb{R}^2 \mid w_1 + w_2 = 1\}$**
*   **Shape:** An infinite, infinitely thin, straight diagonal line (like $y = -x + 1$). 
*   **Test:** Connect two points on a straight line. The segment is just part of the line itself. 
*   **Verdict: CONVEX.** *(Crucial for ML: This is a hyperplane. The decision boundary of an SVM or Perceptron is always a convex set).*

**2. $\{w \in \mathbb{R}^2 \mid w_1 + w_2 \le 1\}$**
*   **Shape:** Everything *below* and including the diagonal line from Question 1. It's a solid, infinite half of the universe.
*   **Test:** Pick two points in the solid "bottom half" of the universe. The line connecting them can't possibly curve up and cross the boundary.
*   **Verdict: CONVEX.**

**3. $\{w \in \mathbb{R}^2 \mid \max(|w_1|, |w_2|) \le 1\}$**
*   **Shape:** A solid, filled-in **Square** bounded by -1 and 1 on the X and Y axes. 
*   **Test:** Pick two points inside a solid square. The line connecting them stays inside the square. 
*   **Verdict: CONVEX.** *(This is the L-infinity norm).*

**4. $\{w \in \mathbb{R}^2 \mid \|w\|_2 \ge 1\}$**
*   **Shape:** Imagine a piece of paper with a circle cut out of the middle. This set is the paper. It is everything *outside* the circle.
*   **Test:** Pick a point on the far left of the paper, and a point on the far right. Draw a line. It passes exactly through the empty hole in the middle. 
*   **Verdict: NOT CONVEX.** *(This is why maximizing distance is mathematically much harder than minimizing it).*

**5. The L0 "Norm": Count of non-zero features $\le 1$**
*   **Shape:** If only 1 feature can be non-zero, you are either on the X-axis ($[w_1, 0]$) OR the Y-axis ($[0, w_2]$). The shape is a giant, infinitely thin **"+" (a cross)**.
*   **Test:** Pick point A at $[0, 5]$ (top of the cross). Pick point B at $[5, 0]$ (right arm of the cross). Connect them. The line segment cuts diagonally through empty space. 
*   **Verdict: NOT CONVEX.** *(This is a massive concept in ML. L0 regularization gives you perfect feature selection, but it is Not Convex, meaning gradient descent fails. We use L1 regularization (solid diamond) instead because it acts similarly but IS convex!)*

**6. The Donut: $1 \le \|w\|_2 \le 2$**
*   **Shape:** A solid ring/donut. 
*   **Test:** Pick a point on the left side of the donut meat, and the right side of the donut meat. The line goes straight through the empty donut hole.
*   **Verdict: NOT CONVEX.**

**7. The Probability Simplex: $p_1 + p_2 + p_3 = 1$, all $\ge 0$**
*   **Shape:** A flat, solid, triangular plate in 3D space, connecting the points (1,0,0), (0,1,0), and (0,0,1). 
*   **Test:** It's a solid flat triangle. Connecting any two points stays on the flat triangle. 
*   **Verdict: CONVEX.** *(This is the output space of a Softmax layer in neural networks! Because it's convex, optimizing Softmax outputs is mathematically smooth and easy).*

**8. Positive Quadrant: $w_1 \ge 0, w_2 \ge 0$**
*   **Shape:** The top-right quarter of a 2D graph. A solid, infinite 90-degree corner.
*   **Test:** Connect any two points in the top right. You can't accidentally enter the negative zones.
*   **Verdict: CONVEX.**

**9. Intersection of Two Convex Sets**
*   **Shape:** A Venn diagram overlap. Imagine the shape of an eye. 
*   **Test:** The overlapping region has flat or bulging-out walls. The line stays inside. 
*   **Verdict: CONVEX.** *(Golden Rule of ML Optimization: The intersection of ANY number of convex sets is ALWAYS convex!)*

**10. Union of Two Convex Sets**
*   **Shape:** Two separate bowling balls sitting on the floor. 
*   **Test:** Pick Point A in ball 1, and Point B in ball 2. The line connecting them travels through the empty air between the balls.
*   **Verdict: NOT CONVEX.** *(Golden Rule of ML Optimization: The union of convex sets is generally NOT convex).*

How did you do? If you can visualize the shape, you can conquer any convexity question they throw at you!
