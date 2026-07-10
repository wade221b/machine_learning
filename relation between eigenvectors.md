The formula **$v := \frac{1}{\sqrt{\lambda}} X^\top a$** is the mathematical bridge that allows Kernel PCA to work. It translates an eigenvector found in the "Kernel space" back into an eigenvector in the "feature space," while making sure its length is exactly 1.

To understand why this makes sense, we have to look at a neat linear algebra trick called **"Dual PCA,"** which proves that the eigenvectors of the Covariance matrix ($C$) and the Kernel matrix ($K$) are intimately linked.

Here is the step-by-step breakdown of why that specific formula is used.

---

### Step 1: The Goal and the Problem
In standard PCA, you want to find the eigenvectors (let's call them $v$) of the covariance matrix $C$.
Assuming your data matrix $X$ is centered, the covariance matrix is proportional to $X^\top X$.
So, you are trying to solve this eigenvalue equation:
$$(X^\top X)v = \lambda v$$

**The Problem in Kernel PCA:** In Kernel PCA, the data is mapped to a massively high-dimensional (sometimes infinite) space. You cannot explicitly compute or store $X^\top X$. Therefore, you cannot find $v$ directly.

### Step 2: The "Trick" (Explaining the $X^\top a$ part)
Instead of dealing with $X^\top X$, we use the Kernel trick to create the Kernel Matrix $K = X X^\top$. This matrix represents the similarities between data points and is easy to compute, regardless of dimensions.

We calculate the eigenvectors of $K$. Let's call one of these eigenvectors $a$, with eigenvalue $\lambda$.
$$K a = \lambda a$$
Since $K = X X^\top$, we substitute it in:
$$(X X^\top) a = \lambda a$$

Here is the magic step. Multiply both sides of that equation on the left by $X^\top$:
$$X^\top(X X^\top a) = X^\top(\lambda a)$$
$$(X^\top X)(X^\top a) = \lambda (X^\top a)$$

Look closely at the structure of that final equation. It looks exactly like our goal equation $(X^\top X)v = \lambda v$, where **$v = X^\top a$**.

**Conclusion 1:** If you know the eigenvector $a$ of the Kernel matrix, you can find the eigenvector $v$ of the Covariance matrix simply by multiplying $X^\top$ by $a$. This tells you the *direction* is correct.

### Step 3: The Normalization (Explaining the $\frac{1}{\sqrt{\lambda}}$ part)
We know the direction is $v = X^\top a$. However, a principal component vector **must have a length of 1** (i.e., its magnitude $||v|| = 1$). Let's check the length of our new vector $X^\top a$.

The squared length of a vector is the vector multiplied by its transpose:
$$||X^\top a||^2 = (X^\top a)^\top (X^\top a)$$
$$= a^\top X X^\top a$$

We know that $X X^\top$ is our Kernel matrix $K$:
$$= a^\top K a$$

Because $a$ is an eigenvector of $K$, we know $K a = \lambda a$:
$$= a^\top (\lambda a)$$
$$= \lambda (a^\top a)$$

The slide states you must first ensure $a$ is normalized to length 1 (so $a^\top a = 1$). Therefore:
$$= \lambda (1) = \lambda$$

**Conclusion 2:** The squared length of the vector $(X^\top a)$ is $\lambda$. This means its actual length is **$\sqrt{\lambda}$**.

### Summary: Putting it together
If we just use $v = X^\top a$, our eigenvector will be too long (by a factor of $\sqrt{\lambda}$). To fix this and force the vector to have a length of exactly 1, we must divide it by its length.

$$v_{normalized} = \frac{\text{Direction}}{\text{Length}} = \frac{X^\top a}{\sqrt{\lambda}}$$

**In plain English:** The $X^\top a$ part magically translates the Kernel eigenvector into a Covariance eigenvector. The $\frac{1}{\sqrt{\lambda}}$ part shrinks it down so it is a proper, standardized unit vector.