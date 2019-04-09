# Superoperator representations

This document summarizes our conventions for the different superoperator representations. We show how to apply the channels to states in these represenations and how to convert channels between a subset of representations.

At the bottom of the document you can find a list of references if you need more information. These references explain, for example, when how to determine if a channel is unital or completely postive in the different representations.

## `vec` and `unvec`
Consider an $m\times m$ matrix
$$ A = [a_{ij}] = \begin{pmatrix}  
a_{11} & a_{12} & \ldots & a_{1m} \\\\
a_{21} & a_{22} & \ldots & a_{2m}\\\\ 
\vdots &   & \ddots & \vdots\\\\ 
a_{m1} & a_{m2} & \ldots & a_{mm} 
\end{pmatrix}$$
where $i$ is a row and $j$ is a column index.

We define `vec` to be column stacking

$$ |A\rangle \rangle :={\rm vec}(A) = (a_{11},a_{21},\ldots,a_{m1},a_{12},\ldots,a_{mm})^T \quad (1) $$

were $T$ denotes a transpose. Clearly an inverse operation, `unvec` can be defined so that 

$$ {\rm unvec}\big ( {\rm vec}(A) \big ) = A.$$

Similarly we can define a row vectorization to be row stacking $ {\rm vec_r}(A) = (a_{11}, a_{12}, \ldots, a_{1m}, a_{21},\ldots, a_{mm})^T$. Note that ${\rm vec}(A) = {\rm vec_r}(A^T)$. In any case we will **not** use this row convention.

### Matrix multiplication in vectorized form

For matricies $A,B,C$ 

$$\begin{align}
{\rm vec}(ABC) = (C^T\otimes A) {\rm vec}(B), \quad (2)
\end{align}$$
which is sometimes called Roth's lemma.

Eq. 2 is useful in representing quantum operations on mixed quantum states. For example consider 
$$ \rho' = U \rho U^\dagger.$$
We can use Eq. 1 to write this as

$$ {\rm vec}(\rho') = \{(U^\dagger)^T \otimes U \} {\rm vec}(\rho)
= (U^*\otimes U) |\rho\rangle\rangle$$
so 
$$ |\rho'\rangle \rangle = \mathcal U |\rho\rangle\rangle,
$$
where $\mathcal U = U^*\otimes U$. The nice thing about this is the operator (the state) has become a vector and the superoperator (the left right action of $U$) has become an operator. 


Some other useful results related to vectorization are

$ {\rm vec}([A,X])= (I\otimes A - A^T\otimes I) {\rm vec}(X)$

${\rm vec}(ABC) = (I\otimes AB) {\rm vec}( C ) = (C^T B^T\otimes I) {\rm vec}(A)$

${\rm vec}(AB) = (I\otimes A) {\rm vec}(B) = (B^T\otimes I) {\rm vec}(A)$. 


## The $n$-qubit Pauli basis 
 The $n$-qubit Pauli basis is denoted $\mathcal P^{\otimes n} $ where $\mathcal  P = \{ I, X, Y, Z \}$ are the usual Pauli matricies. It is an operator baisis for the $d = 2^n$ dimensional Hilbert space. There are $d^2 = 4^n$ operators. If one divides all the operators by $\sqrt{d}$ the basis is orthonormal with respect to the Hilbert-Schmidt inner product.
 
It is often convenient to index the $d^2$ operators with a single label, e.g. $P_0=I^{\otimes n},\, \ldots,\, P_{d^2-1}= Z^{\otimes n}$  (or $P_1=I^{\otimes n}$ if you like 1 indexing). In anycase, as these operators are Hermitian and unitary they obey $P_i^2=I^{\otimes n}$.
 
To be explicit, for two qubits $d=4$ and we have 16 operators e.g. $\{II,IX,IY,IZ,XI,XX,XY,...,ZZ\}$ were $II$ should be intepreted as $I\otimes I$ etc. 

## Quantum channels in the Kraus decomposition (or operator-sum representation)
A completely positive map on the state $\rho$ can be written using a set of Kraus operators $\{ M_k \}$ as


$$\rho' =\mathcal E (\rho) = \sum_{k=1}^N M_k \rho M_k^\dagger, $$
where $\rho'$ is the state at the output of the channel.

If $\sum_k M_k^\dagger M_k= I $ the map is trace preserving. It turns out that $N\le d^2$ where $d$ is the Hilbert space dimension e.g. $d=2^n$ for $n$ qubits.


## Kraus to process matrix (aka $\chi$ or chi matrix)
We choose to represent the process matrix in the Pauli basis. So we expand each of the Kraus operators in the $n$ qubit Pauli basis 

$M_k = \sum^{d^2}_{j=1}c_{kj}\,P_j$ 

where $\mathcal P_j \in \mathcal P ^{\otimes n}$.

Now the channel $\mathcal E$ can be written as

$\mathcal E (\rho) = \sum_{i,j=1}^{d^2} \chi_{i,j} P_i\rho P_j ,$

where 

$$\chi_{i,j} = \sum_k c_{k,i} c_{k,j}^*$$ 

is an element of the process matrix $\chi$  of size $d^2 \times d^2$. The if the channel is CP the process matrix is a Hermitian and positive semidefinite. 


## Kraus to Pauli Transfer matrix
We begin by defining the Pauli vector representation of the state $\rho$ 

$$ |\rho \rangle \rangle = \sum_j c_j |P_j\rangle \rangle$$

where $P_j \in \mathcal P^{\otimes n}$ and $c_j = (1/d) \langle\langle P_j|\rho \rangle\rangle$.

The Pauli-Liouville or Pauli transfer matrix representation of the channel $\mathcal E$ is denoted by $R_{\mathcal E}$. The matrix elements are

$$(R_{\mathcal E})_{i,j} = \frac 1 d {\rm Tr}[P_i \mathcal E(P_j)].$$

Trace preservation implies $(R_{\mathcal E})_{0,j} = \delta_{0,j}$, i.e. the first row is one and all zeros. Unitality implies $(R_{\mathcal E})_{i,0} = \delta_{i,0}$, the first column is one and all zeros.

In this representation the channel is applied to the state by multiplication

$$|\rho' \rangle \rangle = R_{\mathcal E} |\rho \rangle \rangle$$


## Kraus to Superoperator
We already saw an example of this in the setion on `vec`-ing. There we had uintary evolutiuon which only requires one Kraus operator. Lets generalized that to many Kraus opeerators.  

Consider the set of Kraus operators $\{ M_k \}$. The corresponding quantum operation is $\mathcal E (\rho) = \sum_k M_k \rho M_k^\dagger $.

Using the vec operator (see Eq. 1) this implies a superoperator

$\mathcal E = \sum_k (M_k^\dagger)^T \otimes M_k = \sum_k M_k^* \otimes M_k.$

## Kraus to Choi

Define $ | \eta \rangle = \frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}|i,i \rangle $

One can show that 

$|A\rangle \rangle = {\rm vec}(A) = \sqrt{d} (I\otimes A) |\eta\rangle$.

The Choi state is 

$\begin{align}
\mathcal C &= I\otimes \mathcal E (|\eta \rangle \langle \eta|) \\\\
&=\sum_i (I \otimes M_i) |\eta \rangle \langle \eta  | ( I \otimes M_i^\dagger)\\\\
& = \frac{1}{d} \sum_i {\rm vec}(M_i)  {\rm vec} (M_i) ^\dagger \\\\
& = \frac{1}{d} \sum_i |M_i\rangle \rangle \langle\langle M_i |. 
\end{align}$

An often qouted equivalent expression is

$\begin{align}
\mathcal C &= I\otimes \mathcal E (|\eta \rangle \langle \eta|) \\\\
&=\sum_{ij} |i\rangle \langle j| \otimes  \mathcal E (|i \rangle \langle j | ).
\end{align}$


## Choi to Pauli transfer matrix
First we normalize the Choi representation

$$\begin{align}
\rho_{\mathcal E}=\frac 1 d \mathcal C = \frac 1 d \sum_{ij} |i\rangle \langle j| \otimes  \mathcal E (|i \rangle \langle j | )
\end{align}$$ 

Then the matrix elements of the Pauli transfer matrix representation of the channel can be obtained from the Choi state using

$$(R_{\mathcal E})_{i,j} ={\rm Tr}[ \rho_{\mathcal E} \, P_j^T \otimes P_i].$$


## Pauli transfer matrix to Choi
We obtain the normalized Choi matrix using the 

$$ \rho_{\mathcal E} = \frac{1}{d^2}\sum_{i,j=1}^{d^2} (R_{\mathcal E})_{i,j}  \, P_j^T \otimes P_i.$$


## Process / $\chi$ matrix to Pauli transfer matrix

$$(R_{\mathcal E})_{i,j} = \frac 1 d \sum_{k,l}\chi_{k,l} {\rm Tr}[ P_i P_k P_j P_l].$$


## Examples

### Unitary Channels or Gates
As an example we look two single qubit channels $R_z(\theta) = \exp(-i \theta Z/2)$ and $H$. The Hadamard is is a nice channel to examine as it transforms $X$ and $Z$ to each other
$$\begin{align}
H Z H^\dagger &=X\\\\
H X H^\dagger &= Z
\end{align}$$ 
which can be easily seen in some of the channel representations.

**Kraus** 

As the channel is unitary there is only one Kraus operators used in the operator sum representation. However we express them in the Pauli basis to make some of the below manipulations easier
$$\begin{align}
R_z(\theta) &= \cos(\theta/2) I - i \sin(\theta/2) Z\\\\
&= \begin{pmatrix}  
e^{-i\theta/2} & 0 \\\\
0 & e^{i\theta /2}
\end{pmatrix}
\\\\
H &= \frac{1}{\sqrt{2}} (X+Z)\\\\
&=\frac{1}{\sqrt{2}} 
 \begin{pmatrix}  
1 & 1 \\\\
1 & -1
\end{pmatrix}
\end{align}$$

**Process or $\chi$ matrix**

$$ \chi(R_z) = [\chi_{ij}] = \frac 1 2\begin{pmatrix}  
1+\cos(\theta) & 0 & 0 & i \sin(\theta) \\\\
0 & 0 & 0 & 0\\\\ 
0 & 0  & 0 & 0\\\\ 
-i\sin(\theta) & 0 & 0 & 1-\cos(\theta) 
\end{pmatrix}$$

$$ \chi(H) = [\chi_{ij}] = \frac 1 2\begin{pmatrix}  
0 & 0 & 0 & 0 \\\\
0 & 1 & 0 & 1\\\\ 
0 & 0 & 0 & 0\\\\ 
0 & 1 & 0 & 1 
\end{pmatrix}$$

**Pauli Transfer matrix**
$$
R_{R_z(\theta)}= [(R_{R_z(\theta)})_{i,j}] =
\begin{pmatrix}  
1 & 0 & 0 & 0 \\\\
0 & \cos(\theta) & -\sin(\theta) & 0 \\\\ 
0 & \sin(\theta) & \cos(\theta) & 0 \\\\ 
0 & 0 & 0 & 1 
\end{pmatrix}$$

$$
R_{H}= [(R_{H})_{i,j}] =
\frac 1 2\begin{pmatrix}  
1 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 1 \\\\ 
0 & 0 & -1 & 0 \\\\ 
0 & 1 & 0 & 0
\end{pmatrix}$$

**Superoperator**
$$ \mathcal R_z(\theta) =  R_z(\theta)^*\otimes  R_z(\theta)=
\begin{pmatrix}  
1 & 0 & 0 & 0 \\\\
0 & e^{i\theta} & 0 & 0\\\\ 
0 & 0  & e^{-i\theta} & 0\\\\ 
0 & 0 & 0 & 1 
\end{pmatrix} 
$$

$$ \mathcal H = H^*\otimes H=\frac 1 2
\begin{pmatrix}  
1 & 1 & 1 & 1 \\\\
1 & -1 & 1 & -1\\\\ 
1 & 1  & -1 &-1\\\\ 
1 & -1 & -1 & 1 
\end{pmatrix} 
$$

**Choi**

$$\begin{align}
\mathcal C_{R_z} &= \frac 1 2  |R_z(\theta)\rangle\rangle\langle\langle R_z(\theta)|\\\\
&=\frac 1 2
\begin{pmatrix}  
1 & 0 & 0 & e^{-i\theta} \\\\
0 & 0 & 0 & 0\\\\ 
0 & 0 & 0 & 0\\\\ 
e^{i\theta} & 0 & 0 & 1 
\end{pmatrix}
\end{align}$$

$$\begin{align}
\mathcal C_H &= \frac 1 2  |H\rangle\rangle\langle\langle H|\\\\
&=\frac 1 2
\begin{pmatrix}  
1  & 1  &  1 & -1 \\\\
1  & 1  &  1 & -1\\\\ 
1  & 1  &  1 & -1\\\\ 
-1 & -1 & -1 &  1 
\end{pmatrix}
\end{align}$$



### Pauli Channels
Pauli channels are nice because they are diagonal in two representations and they have the *depolarlizing channel* as a speical case. 

In the operator sum representation a single qubit Pauli channel is defined as  
$$\mathcal E(\rho) = (1-p_x-p_y-p_z) I \rho I + p_x X\rho X + p_y Y \rho Y + p_z Z \rho Z$$
where $p_x,p_y,p_z\ge 0$ and $p_x+p_y+p_z\le 1$.

If we define $p' = p_x+p_y+p_z$ then

$$\mathcal E(\rho) = (1-p') I \rho I + p_x X\rho X + p_y Y \rho Y + p_z Z \rho Z$$

The Pauli channel specializes to the depolarizing channel when 
$$ p' = \frac 3 4 p \quad {\rm and}\quad p_x=p_y=p_z = p
$$
for $0\le p \le 1$.
 
**Kraus** 

The Kraus operators used in the operator sum representation are
$$\begin{align}
M_0 &= \sqrt{1-p'}I \\\\
M_1 &= \sqrt{p_x}X \\\\
M_2 &= \sqrt{p_y'}Y \\\\
M_3 &= \sqrt{p_z}Z.
\end{align}$$

**Process or $\chi$ matrix**

$$ \chi = [\chi_{ij}] = \begin{pmatrix}  
(1-p') & 0 & 0 & 0 \\\\
0 & p_x & 0 & 0\\\\ 
0 & 0  & p_y & 0\\\\ 
0 & 0 & 0 & p_z 
\end{pmatrix}$$

**Pauli Transfer matrix**
$$
R_{\mathcal E}= [(R_{\mathcal E})_{i,j}] =
\begin{pmatrix}  
1 & 0 & 0 & 0 \\\\
0 & 1-2(p_y+p_z) & 0 & 0 \\\\ 
0 & 0 & 1-2(p_x+p_z) & 0 \\\\ 
0 & 0 & 0 & 1-2(p_x+p_y) 
\end{pmatrix}$$

**Superoperator**
$$(1-p')
\begin{pmatrix}  
1 & 0 & 0 & 0 \\\\
0 & 1 & 0 & 0\\\\ 
0 & 0  & 1 & 0\\\\ 
0 & 0 & 0 & 1 
\end{pmatrix} + 
p_x
\begin{pmatrix}  
0 & 0 & 0 & 1\\\\
0 & 0 & 1 & 0\\\\ 
0 & 1 & 0 & 0\\\\ 
1 & 0 & 0 & 0 
\end{pmatrix}+ 
p_y
\begin{pmatrix}  
0 & 0 & 0 & 1\\\\
0 & 0 & -1 & 0\\\\ 
0 & -1 & 0 & 0\\\\ 
1 & 0 & 0 & 0 
\end{pmatrix}+ 
p_z
\begin{pmatrix}  
1 & 0 & 0 & 0\\\\
0 & -1 & 0 & 0\\\\ 
0 & 0 & -1 & 0\\\\ 
0 & 0 & 0 & 1 
\end{pmatrix}
$$
So 
$$
\begin{pmatrix}  
(1-p')+p_z & 0 & 0 & p_x+p_y \\\\
0 & (1-p')-p_z & p_x-p_y & 0\\\\ 
0 & p_x-p_y  & (1-p')-p_z & 0\\\\ 
p_x +p_y & 0 & 0 & (1-p')+p_z 
\end{pmatrix} $$

**Choi**

$$\begin{align}
\mathcal C &= \frac 1 2 ( |M_0\rangle\rangle\langle\langle M_0|+|M_1\rangle\rangle\langle\langle M_1|+|M_2\rangle\rangle\langle\langle M_2|+|M_3\rangle\rangle\langle\langle M_3|)\\\\
&= \frac 1 2
\begin{pmatrix}  
(1-p')+p_z & 0 & 0 & (1-p')-p_z \\\\
0 & p_x+p_y & p_x-p_y & 0\\\\ 
0 & p_x-p_y  & p_x+p_y & 0\\\\ 
(1-p')-p_z & 0 & 0 & (1-p')+p_z 
\end{pmatrix}
\end{align}$$

### Amplitude Damping or the $T_1$ channel
Amplitude damping is an energy dissipation (or relaxation) process. If a qubit it in it's excited state $|1\rangle$ it may emit energy, a photon, and transition to the ground state $|0\rangle$. In device physics an experiment that measures the decay over some time $t$, with functional form $\exp(-\Gamma t)$, is known as a $T_1$ experiment (where $T_1 = 1/\Gamma$).

From the perspective of quantum channels the amplitude damping channel is interesting as is an example of a non-unital channel i.e. one that does not have the identity matrix as a fixed point $\mathcal E_{AD} (I) \neq I$.

**Kraus** 

The Kraus operators are
$$\begin{align}
M_0 &=   \sqrt{I - \gamma \sigma_+\sigma_-}
= \begin{pmatrix}  
1 & 0 \\\\
0 & \sqrt{1-\gamma}
\end{pmatrix}
\\\\
M_1&=\sqrt{\gamma}\sigma_- 
=\begin{pmatrix}  
0 & \sqrt{\gamma} \\\\
0 & 0
\end{pmatrix}
\end{align}$$
where $\sigma_- = (\sigma_+)^\dagger= \frac 1 2 (X +i Y) =|0\rangle \langle 1| $. To relate this channel to a $T_1$ process we make the decay rate time dependant $\gamma(t) = \exp(-\Gamma t)$.

**Process or $\chi$ matrix**

$$ \chi(AD) = [\chi_{ij}] = \frac 1 4\begin{pmatrix}  
(1+\sqrt{1-\gamma})^2 & 0       & 0        & \gamma \\\\
0 						  & \gamma  & -i\gamma & 0\\\\ 
0 						  & i\gamma & \gamma   & 0\\\\ 
\gamma 				  & 0  & 0        & (-1+\sqrt{1-\gamma})^2
\end{pmatrix}$$

**Pauli Transfer matrix**
$$
R_{AD}= [(R_{AD})_{i,j}] =
\begin{pmatrix}  
1 & 0 & 0 & 0 \\\\
0 & \sqrt{1-\gamma} & 0 & 0 \\\\ 
0 & 0 & \sqrt{1-\gamma} & 0 \\\\ 
\gamma & 0 & 0 & 1-\gamma 
\end{pmatrix}$$

**Superoperator**
$$
\begin{pmatrix}  
1 & 0 & 0 & \gamma \\\\
0 & \sqrt{1-\gamma} & 0 & 0\\\\ 
0 & 0  & \sqrt{1-\gamma} & 0\\\\ 
0 & 0 & 0 & 1-\gamma 
\end{pmatrix}
$$


**Choi**

$$\begin{align}
\mathcal C &= \frac 1 2 ( |M_0\rangle\rangle\langle\langle M_0|+|M_1\rangle\rangle\langle\langle M_1|)\\\\
&=\frac 1 2
\begin{pmatrix}  
1 & 0 & 0 & \sqrt{1-\gamma} \\\\
0 & 0 & 0 & 0\\\\ 
0 & 0  & \gamma & 0\\\\ 
\sqrt{1-\gamma} & 0 & 0 & 1-\gamma 
\end{pmatrix}
\end{align}$$



## References

[IGST] Introduction to Quantum Gate Set Tomography   
Greenbaum,   
arXiv:1509.02921, (2015)  
https://arxiv.org/abs/1509.02921     

[QN] Quantum Nescimus. Improving the characterization of quantum systems from limited information  
Harper,  
PhD thesis University of Sydney, 2018  
https://ses.library.usyd.edu.au/handle/2123/17896 

[GRAPTN] Tensor networks and graphical calculus for open quantum systems  
Wood et al.,  
Quant. Inf. Comp. 15, 0579-0811 (2015)  
https://arxiv.org/abs/1111.6950 

[MATQO] On the Matrix Representation of Quantum Operations  
Nambu et al.,  
arXiv: 0504091 (2005)  
https://arxiv.org/abs/quant-ph/0504091

[DUAL] On duality between quantum maps and quantum states  
Zyczkowski et al.,  
Open Syst. Inf. Dyn. 11, 3 (2004)  
https://dx.doi.org/10.1023/B:OPSY.0000024753.05661.c2  
https://arxiv.org/abs/quant-ph/0401119 
