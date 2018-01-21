---
layout: post
mathjax:true
---

## Chapter 2:

#### Matrix & Vector Properties
- The matrix product $\textbf{C} = \textbf{AB}$ is defined by $$\textbf{C}_{i,j} = \sum_{k} \textbf{A}_{i,k}\textbf{B}_{k,j}$$
    - This equation basically says that the $(i,j)$ th element of $\textbf{C}$ is equal to the dot product of the $i$th row of $\textbf{A}$ and the $j$th column of $\textbf{B}$. 
    - Matrix multiplication is not commutative: $AB \neq BA$ in general
    - It is associative: $(\textbf{AB})\textbf{C} = \textbf{A}(\textbf{BC})$ and distributive: $\textbf{A}(\textbf{B} + \textbf{C}) = \textbf{AB} + \textbf{BC}$.
- **Hadamard product** is an element wise product of 2 matrices.
$$
\begin{bmatrix}
1 & 2 \\
3 & 4 
\end{bmatrix}
\circ
\begin{bmatrix}
1 & 2 \\
3 & 4 
\end{bmatrix}
=
\begin{bmatrix}
1 & 4 \\
9 & 16
\end{bmatrix}
$$
- The **dot product** is commutative, $\textbf{x}^T\textbf{y} = \textbf{y}^T\textbf{x}$

- We can express a system of linear equations as a matrix multiplication $\textbf{Ax}=\textbf{b}$

- If $\textbf{A}$ is invertible, then $\textbf{A}\textbf{A}^{-1} = \textbf{I}$. Multiplying a matrix by its **inverse** yields the identity matrix.
    - for $\textbf{A}^{-1}$ to exist, $\textbf{Ax}=\textbf{b}$ must have one and only one solution for each $\textbf{b}$

#### Span and Linear Dependence
- The **span** of vectors is all points obtainable by **linear combination** of a set of vectors 
- A linear cobnation of a set of vectors is given by scaling each vector and taking the sum of the result: $\sum_i c_i\textbf{v}_i$.
- if $\textbf{Ax}=\textbf{b}$ then $\textbf{b}$ is in the column-span, or **range** of the columns of $\textbf{A}$,
- In order for $\textbf{Ax} = \textbf{b}$ to have a solution for $x$, $b$ must be in the column-span of $A$. In order for $Ax = b$ to have a solution for $x$,$b$ must be in the column-span of $A$. This means that some linear combination of the vectors given by $A$' columns must be equal to $b$. The elemements of $x$, then, tell us how much to scale each of $A$'s column vectors by to get $b$.
    - in the case of $n \geq m$ we also must ensure that the columns are **linearly independent**, which means that no column is a linear combination of the other vectors in the set
    - also. we need to ensure that there are at most $m$ columns, as this ensures there is one and only one solution. This means that $\textbf{A}$ must be **square**  and non-singular
    - If $\hat{x}$ is a solution to $Ax = b$ then $y$ is also a solution if $A(\hat{x} + y) = b \rightarrow{} Ay = 0$ has any non-trivial solutions. This means that the null-space of $A$ has nontrivial vectors and $Ax = b$ has infinitely many solutions.


#### Norms
- used to measure the size of a vector
- the $L^p$ norm is defined by $\left\vert\vert\textbf{x}\right\vert\vert_p =  (\sum_i\left\vert{x}_i\right\vert^p)^{\frac{1}{p}}$
- any function that ensures
    - $f(\textbf{x}) \implies \textbf{x}=0$
    - $f(\textbf{x} + \textbf{y}) \leq f(\textbf{x}) + f(\textbf{y})$
    - $\forall \alpha \in \mathbb{R} = \left\vert\alpha\right\vert f(\textbf{x})$
- $L^2$ norm is also known as the **Euclidean norm**, although it's square is usually easier to work with and be simply calculated as $\textbf{x}^T\textbf{x}$
- We can use the $L_1$ norm when we want to distinguish between the elements that are close to $0$ and are actually $0$. Whenever an element of $x$ moves by some $\epsilon$ from $0$, the $L_1$ norm grows by exactly $\epsilon$.
- The max norm $\Vert{x}_\infty$  corresponds to the maximum element of $x$. 
- Frobenius norm: The $L^2$ norm applied to matrices: $\sqrt{\sum_{i,j}A_{i,j}^2}$

#### Special Matrices
- **diagonal** if and only if $D_{i, j} =0 \forall i \neq j$, can be written as $diag(\textbf{v})$
- **symmetric** matrix is any matrix equal to it's transpose $\textbf{A} = \textbf{A}^T$
- **unit vectors** are any vector with 1 $L^2$ norm 
- **orthogonal** vectors are any two vectors s.t. $\textbf{x}^T\textbf{y} = 0$
- **orthonormal** vectors are both orthogonal and of unit norm
- **orthogonal** matrix is a square matrix whose rows and columns are mutually orthonormal. An orthonormal matrix has the property $AA^T = A^TA = I = A^{-1}A$. 
- A linear map $f(x) = Ax$ has many properties when $A$ is orthogonal:
  - inner products are preserved: $(Ax)^T(Ay) = x^Ty$ 
  - norms are preserved: $\Vert{Ax} = \Vert{x}$
  - distances are preserved: $\Vert{Ax - y} = \Vert{x - y}$


#### Eigendecomposition
- we can decompose matrices into a set of eigenvectors and eigenvalues
    - An **eigenvector** of $\textbf{A}$ is any vector $\textbf{v}$ s.t. $\textbf{Av} = \lambda\textbf{v}$ - which means $\textbf{v}$ is only rescaled
    - $\lambda$ is the **eigenvalue** for that eigenvector
    - Let $A$ be a matrix with $n$ linearly independent eigenvectors ${v_1...v_n}$, and let the corresponding eigenvalues be ${\lambda_1 ... \lambda_n}$. Then the eigendecomposition of $A$ is given by $$A = Vdiag(\lambda)V^{-1}$$ where $V$ is the matrix obtained by letting the eigenvectors be the columns of the matrix $V$, and letting $diag(\lambda)$ be a diagonal matrix who's entries are given by the vector $\lambda$. 
    - If the matrix $A$ is symmetric then $V$ will be an orthogonal matrix. Since the inverse of an orthogolal matrix is it's transpose, we can rewrite the eigendecomposition for symmetric matrices as $A = Vdiag(\lambda)V^T$. 
- The eigendecomposition is unique only if all the eigenvalues are unique. 
- If a matrix has $0$ as an eigenvalue, that means there exists a vector $v$ such that $Av = 0$ and the matrix is singular and cannot be inverted.
- **Positive definite** - a matrix whose eigenvalues are all positive
- **positive semidefinite** - a matrix whose eigenvalues aer all positive or 0. For a PSD matrix we have $z^TAz \geq 0 $. 
- **negative definite** - a matrix with all negative eigenvalues
- **negative semidefinite** - a matrix with all negative or 0 eigenvalues

#### Optimization with Eigenvalues and Eigenvectors

- Often come up in maximizing some function of a matrix (i.e. PCA)
- The solution to the optimization problem $max_x x^TAx$ subject to $\Vert{x} = 1$ is $x_1$, the eigenvector corresponding to the largest eigenvalue.  

#### Singular Value Decomposition
- factorize a matrix into singular vectors and singular values
- We have $A = UDV^T$
- Here, $U$ and $V$ are orthogonal matrices and $D$ is a diagonal matrix. 
- $U$ corrseponds to the **left singular vectors** of $A$ which are the eigenvectors of $AA^T$. Similarly, $V$ corresponds to the **right singular vectors** of $A$ which are the eigenvectors of $A^TA$. 
- $D$'s diagonal entries correspond to $A$'s singular values, which are the square roots of the eigenvalues of $AA^T$. 
- The SVD is useful to compute the **Moore-Penrose Psuedoinverse** of nonsquare matrices. 

#### Trace & Determinant
- The trace operator gives the sum of all diagonal entries of a matrix: $Tr(A) = \sum_i A_{i,i}$. The trace is invariant to transpose. 
- The Frobenius norm can be expressed in terms of the trace: $\Vert{A}_F = \sqrt{Tr(AA^T)}$


- The determinant maps matrices to real-valued scalars. The determinant of a matrix is equal to the product of it's eigenvalues. It can be thought of a measure of how the matrix expand/contracts space.
- The trace is the sum of all the eigenvalues, while the determinant is their product. 
