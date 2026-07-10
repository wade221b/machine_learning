To derive the formula for $v$ on line 5, we look at the relationship between the covariance matrix $C$ and the Gram/Kernel matrix $K$ in dual Principal Component Analysis (PCA).
Here is the step-by-step proof.
1. Define the Matrices and Setup
Let $X$ be the centered data matrix of size $n \times d$ (where $n$ is the number of data points and $d$ is the number of features).
The covariance matrix (up to a scaling factor) is given by:
$$C = X^\top X$$
The kernel matrix (or Gram matrix) is given by:
$$K = X X^\top$$
We are given that $a$ is a normalized eigenvector of $K$ with eigenvalue $\lambda$. By definition, this satisfies the eigenvalue equation:

$$K a = \lambda a \implies (X X^\top) a = \lambda a$$
2. Relate the Eigenvectors of $K$ and $C$
We want to find the eigenvector $v$ of the covariance matrix $C$, which must satisfy:

$$C v = \lambda v \implies (X^\top X) v = \lambda v$$
Let's take the eigenvalue equation for $K$ and multiply both sides from the left by $X^\top$:

$$X^\top (X X^\top a) = X^\top (\lambda a)$$
Using the associative property of matrix multiplication, we can regroup the terms:

$$(X^\top X) (X^\top a) = \lambda (X^\top a)$$
Notice that this matches the form $C \cdot \text{vector} = \lambda \cdot \text{vector}$. This proves that the vector $X^\top a$ is an eigenvector of $C$ with the exact same eigenvalue $\lambda$.
Therefore, we can write:

$$v \propto X^\top a$$
3. Normalize to Unit Length
The slide states that $v$ must have unit length ($\|v\| = 1$), and we are given that $a$ is already normalized ($\|a\| = 1$, meaning $a^\top a = 1$).
Let's calculate the squared norm of the unnormalized vector $X^\top a$:

$$\|X^\top a\|^2 = (X^\top a)^\top (X^\top a)$$
Using the transpose property $(AB)^\top = B^\top A^\top$:

$$\|X^\top a\|^2 = a^\top X X^\top a$$
Since $X X^\top a = \lambda a$ (from our first equation), we can substitute it in:

$$\|X^\top a\|^2 = a^\top (\lambda a) = \lambda (a^\top a)$$
Because $a$ is a unit vector ($a^\top a = 1$):

$$\|X^\top a\|^2 = \lambda$$
Taking the square root gives the norm (length) of the vector:

$$\|X^\top a\| = \sqrt{\lambda}$$
Conclusion
To turn $X^\top a$ into a unit eigenvector $v$, we simply divide it by its length $\sqrt{\lambda}$:

$$v = \frac{1}{\sqrt{\lambda}} X^\top a$$
