# Superoperator representations

## `vec` and `unvec`

$$ A = [a_{ij}] = \begin{pmatrix}  
a_{11} & a_{12} & \ldots & a_{1m} \\\\
a_{21} & a_{22} & \ldots & a_{2m}\\\\ 
\vdots &   & \ddots & \vdots\\\\ 
a_{m1} & a_{m2} & \ldots & a_{mm} 
\end{pmatrix}$$
where $i$ is a row and $j$ is a column index.

We define `vec` to be column stacking

$ {\rm vec}(A) = (a_{11},a_{21},\ldots,a_{m1},a_{12},\ldots,a_{mm})^T$

were $T$ denotes a transpose. Clearly an inverse operation, `unvec` can be defined so that 

$ {\rm unvec}\big ( {\rm vec}(A) \big ) = A$.

Similarly we can define a row vectorization to be row stacking

$|A\rangle \rangle := {\rm vec_r}(A) = (a_{11}, a_{12}, \ldots, a_{1m}, a_{21},\ldots, a_{mm})^T.$

Note that ${\rm vec}(A) = {\rm vec_r}(A^T)$. In any case we will **not** use this row convention.

For matricies $A,B,C$ some useful results related to vectorization are

${\rm vec}(ABC) = (C^T\otimes A) {\rm vec}(B)\quad$ [Eq.1]

$ {\rm vec}([A,X])= (I\otimes A - A^T\otimes I) {\rm vec}(X)$

${\rm vec}(ABC) = (I\otimes AB) {\rm vec}( C ) = (C^T B^T\otimes I) {\rm vec}(A)$

${\rm vec}(AB) = (I\otimes A) {\rm vec}(B) = (B^T\otimes I) {\rm vec}(A)$

## Kraus to Superoperator

Consider the set of Kraus operators $\{ M_k \}$. The corresponding quantum operation is $\mathcal E (\rho) = \sum_k M_k \rho M_k^\dagger $.

Using the vec operato (see Eq. 1) this implies a superoperator

$\mathcal E = \sum_k (M_k^\dagger)^T \otimes M_k = \sum_k M_k^* \otimes M_k$

## Kraus to Choi

Define $ | \eta \rangle = \frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}|i,i \rangle $

One can show that 

$|A\rangle \rangle = {\rm vec}(A) = \sqrt{d} (I\otimes A) |\eta\rangle$

The Choi state is 

$\begin{align}
\mathcal C &= I\otimes \mathcal E |\eta \rangle \langle \eta| \\\\
&=\sum_i (I \otimes M_i) |\eta \rangle \langle \eta  | ( I \otimes M_i^\dagger)\\\\
& = \frac{1}{d} \sum_i {\rm vec}(M_i)  {\rm vec} (M_i) ^\dagger \\\\
& = \frac{1}{d} \sum_i |M_i\rangle \rangle \langle\langle M_i | 
\end{align}$
