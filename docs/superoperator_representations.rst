Superoperator representations
=============================

This document summarizes our conventions for the different superoperator
representations. We show how to apply the channels to states in these
representations and how to convert channels between a subset of
representations. By combining these conversion methods you can convert
between any of the channel representations.

This document is **not** intended to be a tutorial or a comprehensive
review. At the bottom of the document there is a list of references with
more information. This document was influenced by [IGST]_ and we recommend
reading [GRAPTN]_ to gain deeper understanding. (The references are listed
at the bottom of this document.) Additionally these references explain,
for example how to determine if a channel is unital or completely
positive in the different representations.

``vec`` and ``unvec``
---------------------

Consider an :math:`m\times m` matrix

.. math::

    A = [a_{ij}] = \begin{pmatrix}  
   a_{11} & a_{12} & \ldots & a_{1m} \\\\
   a_{21} & a_{22} & \ldots & a_{2m}\\\\ 
   \vdots &   & \ddots & \vdots\\\\ 
   a_{m1} & a_{m2} & \ldots & a_{mm} 
   \end{pmatrix}

where :math:`i` is a row and :math:`j` is a column index.

We define ``vec`` to be column stacking

.. math::  |A\rangle \rangle :={\rm vec}(A) = (a_{11},a_{21},\ldots,a_{m1},a_{12},\ldots,a_{mm})^T \quad (1) 

were :math:`T` denotes a transpose. Clearly an inverse operation,
``unvec`` can be defined so that

.. math::  {\rm unvec}\big ( {\rm vec}(A) \big ) = A.

Of course ``unvec()`` generally depends on the dimensions of :math:`A`,
which are not recoverable from ``vec(A)``. We often focus on square A,
but for generality, we require the dimensions for :math:`A`, defaulting
to the square root of the dimension of ``vec(A)``. Column stacking
corresponds to how matrices are stored in memory for column major
storage conventions.

Similarly we can define a row vectorization to be row stacking :math:`{\rm vec_r}(A) = (a_{11}, a_{12}, \ldots, a_{1m}, a_{21},\ldots, a_{mm})^T`
. Note that :math:`{\rm vec}(A) = {\rm vec_r}(A^T)`. In any case we will
**not** use this row convention.

Matrix multiplication in vectorized form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For matrices :math:`A,B,C`

.. math::

   \begin{align}
   {\rm vec}(ABC) = (C^T\otimes A) {\rm vec}(B), \quad (2)
   \end{align}

which is sometimes called Roth's lemma.

Eq. 2 is useful in representing quantum operations on mixed quantum
states. For example consider

.. math::  \rho' = U \rho U^\dagger.

We can use Eq. 1 to write this as

.. math::

    {\rm vec}(\rho') = \{(U^\dagger)^T \otimes U \} {\rm vec}(\rho)
   = (U^*\otimes U) |\rho\rangle\rangle

so

.. math::

    |\rho'\rangle \rangle = \mathcal U |\rho\rangle\rangle,

where :math:`\mathcal U = U^*\otimes U`. The nice thing about this is
the operator (the state) has become a vector and the superoperator (the
left right action of :math:`U`) has become an operator.

Some other useful results related to vectorization are

$ {}([A,X])= (IA - A^TI) {}(X)$

:math:`{\rm vec}(ABC) = (I\otimes AB) {\rm vec}( C ) = (C^T B^T\otimes I) {\rm vec}(A)`

:math:`{\rm vec}(AB) = (I\otimes A) {\rm vec}(B) = (B^T\otimes I) {\rm vec}(A)`.

Matrix operations on Bipartite matrices: Reshuffling, SWAP, and tranposition
----------------------------------------------------------------------------

This section is based on the Wood et al. presentation in [GRAPTN]_.

As motivation for this section consider the Kraus representation
theorem. It shows that a quantum channel can be represented as a partial
trace over a unitary operation on a larger Hilbert space. Actually the
unitary is on a bipartite Hilbert space.

When representing quantum channels one insight is used many times.

Consider two Hilbert spaces :math:`\mathbb H_A` and :math:`\mathbb H_B`
with dimensions :math:`d_A` and :math:`d_B` respectively. An abstract
quantum process matrix :math:`\mathcal Q` lives in the combined
(bipartite) space of :math:`\mathbb H_A \otimes \mathbb H_B` so
:math:`\mathcal Q` is a :math:`d_A^2\times d_B^2` matrix.

We can represent the process as a tensor with components

.. math:: \mathcal Q_{m,\mu;n,\nu} = \langle m, \mu |\mathcal Q |n,\nu \rangle 

where :math:`|n,\nu\rangle = |n\rangle \otimes |\nu\rangle`,
:math:`m,n\in \{0,\ldots, d_A-1\}`,
:math:`\mu,\nu\in \{0,\ldots, d_B-1\}` and all vectors are in the
standard basis.

With respect to these indices some useful operations are [GRAPTN]_:

Transpose :math:`T`:
:math:`\mathcal Q_{m,\mu;n,\nu} \mapsto \mathcal Q_{n,\nu;m,\mu},`

SWAP:
:math:`\mathcal Q_{m,\mu;n,\nu} \mapsto \mathcal Q_{\mu,m;\nu,n},`

Row-reshuffling :math:`R_r`:
:math:`\mathcal Q_{m,\mu;n,\nu} \mapsto \mathcal Q_{m,n;\mu,\nu},`

Col-reshuffling :math:`R`:
:math:`\mathcal Q_{m,\mu;n,\nu} \mapsto \mathcal Q_{\nu,\mu;n,m}.`

The importance of understanding reshuffling can be understood as
understanding the relationship between

.. math:: {\rm vec}(G)\otimes {\rm vec}(\Gamma) \quad {\rm and} \quad  {\rm vec}(G\otimes\Gamma)

where :math:`G` and :math:`\Gamma` are matrices that act on
:math:`\mathbb H_A` and :math:`\mathbb H_B` respectively, as explained
in [VECQO]_.

A note on numerical implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most linear algebra (or tensor) libraries have the ability to ``reshape``
a matrix and ``swapaxes`` (or sometimes it is called ``permute_dims``).

If you are trying to reshuffle indices usually the first job is to
write your matrix in tensor form. This requires reshaping a
:math:`d_A^2\times d_B^2` matrix into a
:math:`d_A\times d_A\times d_B \times d_B` tensor. Next you
``permute_dims`` or ``swapaxes``. Often :math:`d_A = d_B` so we
``reshape`` to a Matrix that has the same dimensions as the original
:math:`d_A^2\times d_A^2` matrix.

The :math:`n`-qubit Pauli basis
-------------------------------

The :math:`n`-qubit Pauli basis is denoted
:math:`\mathcal P^{\otimes n} ` where
:math:`\mathcal  P = \{ I, X, Y, Z \}` are the usual Pauli matrices. It
is an operator basis for the :math:`d = 2^n` dimensional Hilbert space
and there are :math:`d^2 = 4^n` operators in
:math:`\mathcal P^{\otimes n} `. If one divides all the operators by
:math:`\sqrt{d}` the basis is orthonormal with respect to the
Hilbert-Schmidt inner product.

It is often convenient to index the :math:`d^2` operators with a single
label, e.g.
:math:`P_1=I^{\otimes n},\, \ldots,\, P_{d^2}= Z^{\otimes n}` (or
:math:`P_0=I^{\otimes n}` if you like zero indexing). In anycase, as
these operators are Hermitian and unitary they obey
:math:`P_i^2=I^{\otimes n}`.

To be explicit, for two qubits :math:`d=4` and we have 16 operators e.g.
:math:`\{II, IX, IY, IZ, XI, XX, XY, ..., ZZ\}` were :math:`II` should
be interpreted as :math:`I\otimes I` etc. The single index would be
:math:`\{P_1, P_2, P_3, P_4, P_5, P_6, P_7, ..., P_{16}\}`.

Quantum channels in the Kraus decomposition (or operator-sum representation)
----------------------------------------------------------------------------

A completely positive map on the state :math:`\rho` can be written using
a set of Kraus operators :math:`\{ M_k \}` as

.. math:: \rho' =\mathcal E (\rho) = \sum_{k=1}^N M_k \rho M_k^\dagger, 

where :math:`\rho'` is the state at the output of the channel.

If :math:`\sum_k M_k^\dagger M_k= I ` the map is trace preserving. It
turns out that :math:`N\le d^2` where :math:`d` is the Hilbert space
dimension e.g. :math:`d=2^n` for :math:`n` qubits. Kraus operators are
not necessarily unique, sometimes there is a unitary degree of freedom
in the Kraus representation.

Kraus to :math:`\chi` matrix (aka chi or process matrix)
--------------------------------------------------------

We choose to represent the :math:`\chi` matrix in the Pauli basis. So we
expand each of the Kraus operators in the :math:`n` qubit Pauli basis

:math:`M_k = \sum^{d^2}_{j=1}c_{kj}\,P_j`

where :math:`\mathcal P_j \in \mathcal P ^{\otimes n}`.

Now the channel :math:`\mathcal E` can be written as

:math:`\mathcal E (\rho) = \sum_{i,j=1}^{d^2} \chi_{i,j} P_i\rho P_j ,`

where

.. math:: \chi_{i,j} = \sum_k c_{k,i} c_{k,j}^*

is an element of the process matrix :math:`\chi` of size
:math:`d^2 \times d^2`. If the channel is CP the :math:`\chi` matrix is
a Hermitian and positive semidefinite.

The :math:`\chi` matrix can be related to the (yet to be defined) Choi
matrix via a change of basis. Typically the Choi matrix is defined in
the computational basis, while the :math:`\chi` matrix uses the Pauli
basis. Moreover, they may have different normalization conventions.

In this light, after reviewing the Kraus to Choi conversion it is simple
to see that the above is equivalent to first defining

.. math::


   |c_{k}\rangle\rangle = U_{c2p}{\rm vec}(M_k) 

then

.. math::


   \chi = \sum_k |c_{k}\rangle\rangle \langle\langle c_k|.

Kraus to Pauli-Liouville matrix (Pauli transfer matrix)
-------------------------------------------------------

We begin by defining the Pauli vector representation of the state
:math:`\rho`

.. math::  |\rho \rangle \rangle = \sum_j c_j |P_j\rangle \rangle

where :math:`P_j \in \mathcal P^{\otimes n}` and
:math:`c_j = (1/d) \langle\langle P_j|\rho \rangle\rangle`.

The Pauli-Liouville or Pauli transfer matrix representation of the
channel :math:`\mathcal E` is denoted by :math:`R_{\mathcal E}`. The
matrix elements are

.. math:: (R_{\mathcal E})_{i,j} = \frac 1 d {\rm Tr}[P_i \mathcal E(P_j)].

Trace preservation implies
:math:`(R_{\mathcal E})_{0,j} = \delta_{0,j}`, i.e. the first row is one
and all zeros. Unitality implies
:math:`(R_{\mathcal E})_{i,0} = \delta_{i,0}`, the first column is one
and all zeros.

In this representation the channel is applied to the state by
multiplication

.. math:: |\rho' \rangle \rangle = R_{\mathcal E} |\rho \rangle \rangle.

Kraus to Superoperator (Liouville)
----------------------------------

We already saw an example of this in the section on ``vec``-ing. There we
re-packaged conjugation by unitary evolution into the action of a matrix
on a vec'd density operator. Unitary evolution is simply the case of a
single Kraus operator, so we generalize this by taking a sum over all
Kraus operators.

Consider the set of Kraus operators :math:`\{ M_k \}`. The corresponding
quantum operation is

.. math:: \mathcal E (\rho) = \sum_k M_k \rho M_k^\dagger

Using the vec operator (see Eq. 1) this implies a superoperator

.. math:: \mathcal E = \sum_k (M_k^\dagger)^T \otimes M_k = \sum_k M_k^* \otimes M_k,

which acts as :math:`\mathcal E |\rho\rangle \rangle` using Equation 2.

**Note** In quantum information a superoperator is an abstract concept.
The object above is a concrete representation of the abstract concept in
a particular basis. In the NMR community this particular construction is
called the Liouville representation. The Pauli-Liouville representation
is attained from Liouville representation by a change of basis, so the
similarity in naming makes sense.

Kraus to Choi
-------------

Define $ \| = \_{i=0}^{d-1}\|i,i $

One can show that

:math:`|A\rangle \rangle = {\rm vec}(A) = \sqrt{d} (I\otimes A) |\eta\rangle`.

The Choi state is

.. math::

   \begin{align}
   \mathcal C &= I\otimes \mathcal E (|\eta \rangle \langle \eta|) \\\\
   &=\sum_i (I \otimes M_i) |\eta \rangle \langle \eta  | ( I \otimes M_i^\dagger)\\\\
   & = \frac{1}{d} \sum_i {\rm vec}(M_i)  {\rm vec} (M_i) ^\dagger \\\\
   & = \frac{1}{d} \sum_i |M_i\rangle \rangle \langle\langle M_i |. 
   \end{align}

An often quoted equivalent expression is

:math:`\begin{align} \mathcal C &= I\otimes \mathcal E (|\eta \rangle \langle \eta|) \\\\ &=\sum_{ij} |i\rangle \langle j| \otimes  \mathcal E (|i \rangle \langle j | ). \end{align}`

:math:`\chi` matrix to Pauli-Liouville matrix
---------------------------------------------

.. math:: (R_{\mathcal E})_{i,j} = \frac 1 d \sum_{k,l}\chi_{k,l} {\rm Tr}[ P_i P_k P_j P_l].

Superoperator to Pauli-Liouville matrix
---------------------------------------

The standard basis on :math:`n` qubits is called the computational
basis. It is essentially all the strings
:math:`|c_1\rangle=|0..0\rangle` through to
:math:`|c_{\rm max}\rangle = |1...1\rangle`. To convert between a
superoperator and the Pauli-Liouville matrix representation we need to
do a change of basis from the computational basis to the Pauli basis.
This is achieved by the unitary

.. math::  U_{c2p}= \sum_{k=1}|c_k\rangle\langle\langle P_k|.

The we have

.. math::  R_{\mathcal E} =  U_{c2p} \mathcal E U_{c2p}^\dagger.

Superoperator to Choi
---------------------

The conversion from the superoperator to a Choi matrix
:math:`\mathcal C` is simply a (column) reshuffling operation

.. math::  \mathcal C = R(\mathcal E).

It turns out that $ E = R(C)$ which means that
:math:`\mathcal E= R(R(\mathcal E))`.

Pauli-Liouville matrix to Superoperator
---------------------------------------

To convert between the Pauli-Liouville matrix and the superoperator
representation we need to to a change of basis from the Pauli basis to
the computational basis. This is achieved by the unitary

.. math::  U_{p2c}= \sum_{k=1}|P_k\rangle\rangle \langle k|,

which is simply :math:`U_{c2p}^\dagger`.

The we have

.. math:: \mathcal E =  U_{p2c}R_{\mathcal E}U_{p2c}^\dagger.

Pauli-Liouville to Choi
-----------------------

We obtain the normalized Choi matrix using the expression

.. math::  \rho_{\mathcal E} = \frac{1}{d^2}\sum_{i,j=1}^{d^2} (R_{\mathcal E})_{i,j}  \, P_j^T \otimes P_i.

Choi to Kraus
-------------

This is simply the reverse of the Kraus to Choi procedure.

Given the Choi matrix :math:`\mathcal C` we find its eigenvalues
:math:`\{\lambda_i\}` and vectors :math:`\{|M_i\rangle\rangle \}`. Then
the Kraus operators are

.. math::  M_i = \sqrt{\lambda_i}\, {\rm unvec}\big (|M_i\rangle\rangle\big),

For numerical implementation one usually puts a threshold on the
eigenvalues, say :math:`\lambda> 10^{-10}`, to prevent numerical
instabilities.

Choi to Pauli-Liouville
-----------------------

First we normalize the Choi representation

.. math::

   \begin{align}
   \rho_{\mathcal E}=\frac 1 d \mathcal C = \frac 1 d \sum_{ij} |i\rangle \langle j| \otimes  \mathcal E (|i \rangle \langle j | )
   \end{align}

Then the matrix elements of the Pauli-Liouville matrix representation of
the channel can be obtained from the Choi state using

.. math:: (R_{\mathcal E})_{i,j} ={\rm Tr}[ \rho_{\mathcal E} \, P_j^T \otimes P_i].

Choi to Superoperator
---------------------

The conversion from a Choi matrix :math:`\mathcal C` to a superoperator
is simply a (column) reshuffling operation

.. math::  \mathcal E = R(\mathcal C).

It turns out that $ C = R(E)$ which means that
:math:`\mathcal C= R(R(\mathcal C))`.

Examples: One qubit channels
----------------------------

Some observations:

-  The Choi matrix of a unitary process always has rank 1.
-  The superoperator / Liouville representation of a unitary process is
   always full rank.
-  The eigenvalues of a Choi matrix give you an upper bound to the
   probability a particular (canonical) Kraus operator will occur
   (generally that probability depends on the state). This is helpful
   when sampling Kraus operators (you can test for which occurred
   according to the order of these eigenvalues).
-  The :math:`\chi` matrix (in the Pauli basis) is very convenient for
   computing the result of Pauli twirling or Clifford twirling the
   corresponding process.

Unitary Channels or Gates
~~~~~~~~~~~~~~~~~~~~~~~~~

As an example we look at two single qubit channels
:math:`R_z(\theta) = \exp(-i \theta Z/2)` and :math:`H`. The Hadamard is
is a nice channel to examine as it transforms :math:`X` and :math:`Z` to
each other

.. math::

   \begin{align}
   H Z H^\dagger &=X\\\\
   H X H^\dagger &= Z
   \end{align}

which can be easily seen in some of the channel representations.

**Kraus**

As the channel is unitary there is only one Kraus operator used in the
operator sum representation. However we express them in the Pauli basis
to make some of the below manipulations easier

.. math::

   \begin{align}
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
   \end{align}

**:math:`\chi` matrix (process)**

.. math::

    \chi(R_z) = [\chi_{ij}] = \frac 1 2\begin{pmatrix}  
   1+\cos(\theta) & 0 & 0 & i \sin(\theta) \\\\
   0 & 0 & 0 & 0\\\\ 
   0 & 0  & 0 & 0\\\\ 
   -i\sin(\theta) & 0 & 0 & 1-\cos(\theta) 
   \end{pmatrix}

.. math::

    \chi(H) = [\chi_{ij}] = \frac 1 2\begin{pmatrix}  
   0 & 0 & 0 & 0 \\\\
   0 & 1 & 0 & 1\\\\ 
   0 & 0 & 0 & 0\\\\ 
   0 & 1 & 0 & 1 
   \end{pmatrix}

**Pauli-Liouville matrix**

.. math::


   R_{R_z(\theta)}= [(R_{R_z(\theta)})_{i,j}] =
   \begin{pmatrix}  
   1 & 0 & 0 & 0 \\\\
   0 & \cos(\theta) & -\sin(\theta) & 0 \\\\ 
   0 & \sin(\theta) & \cos(\theta) & 0 \\\\ 
   0 & 0 & 0 & 1 
   \end{pmatrix}

.. math::


   R_{H}= [(R_{H})_{i,j}] =
   \frac 1 2\begin{pmatrix}  
   1 & 0 & 0 & 0 \\\\
   0 & 0 & 0 & 1 \\\\ 
   0 & 0 & -1 & 0 \\\\ 
   0 & 1 & 0 & 0
   \end{pmatrix}

**Superoperator**

.. math::

    \mathcal R_z(\theta) =  R_z(\theta)^*\otimes  R_z(\theta)=
   \begin{pmatrix}  
   1 & 0 & 0 & 0 \\\\
   0 & e^{i\theta} & 0 & 0\\\\ 
   0 & 0  & e^{-i\theta} & 0\\\\ 
   0 & 0 & 0 & 1 
   \end{pmatrix} 

.. math::

    \mathcal H = H^*\otimes H=\frac 1 2
   \begin{pmatrix}  
   1 & 1 & 1 & 1 \\\\
   1 & -1 & 1 & -1\\\\ 
   1 & 1  & -1 &-1\\\\ 
   1 & -1 & -1 & 1 
   \end{pmatrix} 

**Choi**

.. math::

   \begin{align}
   \mathcal C_{R_z} &= \frac 1 2  |R_z(\theta)\rangle\rangle\langle\langle R_z(\theta)|\\\\
   &=\frac 1 2
   \begin{pmatrix}  
   1 & 0 & 0 & e^{-i\theta} \\\\
   0 & 0 & 0 & 0\\\\ 
   0 & 0 & 0 & 0\\\\ 
   e^{i\theta} & 0 & 0 & 1 
   \end{pmatrix}
   \end{align}

.. math::

   \begin{align}
   \mathcal C_H &= \frac 1 2  |H\rangle\rangle\langle\langle H|\\\\
   &=\frac 1 2
   \begin{pmatrix}  
   1  & 1  &  1 & -1 \\\\
   1  & 1  &  1 & -1\\\\ 
   1  & 1  &  1 & -1\\\\ 
   -1 & -1 & -1 &  1 
   \end{pmatrix}
   \end{align}

Pauli Channels
~~~~~~~~~~~~~~

Pauli channels are nice because they are diagonal in two representations
and they have the *depolarizing channel* as a special case.

In the operator sum representation a single qubit Pauli channel is
defined as

.. math:: \mathcal E(\rho) = (1-p_x-p_y-p_z) I \rho I + p_x X\rho X + p_y Y \rho Y + p_z Z \rho Z

where :math:`p_x,p_y,p_z\ge 0` and :math:`p_x+p_y+p_z\le 1`.

If we define :math:`p' = p_x+p_y+p_z` then

.. math:: \mathcal E(\rho) = (1-p') I \rho I + p_x X\rho X + p_y Y \rho Y + p_z Z \rho Z.

The Pauli channel specializes to the depolarizing channel when

.. math::

    p' = \frac 3 4 p \quad {\rm and}\quad p_x=p_y=p_z = p

for :math:`0\le p \le 1`.

**Kraus**

The Kraus operators used in the operator sum representation are

.. math::

   \begin{align}
   M_0 &= \sqrt{1-p'}I \\\\
   M_1 &= \sqrt{p_x}X \\\\
   M_2 &= \sqrt{p_y}Y \\\\
   M_3 &= \sqrt{p_z}Z.
   \end{align}

**:math:`\chi` matrix (process)**

.. math::

    \chi = [\chi_{ij}] = \begin{pmatrix}  
   (1-p') & 0 & 0 & 0 \\\\
   0 & p_x & 0 & 0\\\\ 
   0 & 0  & p_y & 0\\\\ 
   0 & 0 & 0 & p_z 
   \end{pmatrix}

**Pauli-Liouville matrix**

.. math::


   R_{\mathcal E}= [(R_{\mathcal E})_{i,j}] =
   \begin{pmatrix}  
   1 & 0 & 0 & 0 \\\\
   0 & 1-2(p_y+p_z) & 0 & 0 \\\\ 
   0 & 0 & 1-2(p_x+p_z) & 0 \\\\ 
   0 & 0 & 0 & 1-2(p_x+p_y) 
   \end{pmatrix}

**Superoperator**

.. math::

   (1-p')
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

So

.. math::


   \begin{pmatrix}  
   (1-p')+p_z & 0 & 0 & p_x+p_y \\\\
   0 & (1-p')-p_z & p_x-p_y & 0\\\\ 
   0 & p_x-p_y  & (1-p')-p_z & 0\\\\ 
   p_x +p_y & 0 & 0 & (1-p')+p_z 
   \end{pmatrix} 

**Choi**

.. math::

   \begin{align}
   \mathcal C &= \frac 1 2 ( |M_0\rangle\rangle\langle\langle M_0|+|M_1\rangle\rangle\langle\langle M_1|+|M_2\rangle\rangle\langle\langle M_2|+|M_3\rangle\rangle\langle\langle M_3|)\\\\
   &= \frac 1 2
   \begin{pmatrix}  
   (1-p')+p_z & 0 & 0 & (1-p')-p_z \\\\
   0 & p_x+p_y & p_x-p_y & 0\\\\ 
   0 & p_x-p_y  & p_x+p_y & 0\\\\ 
   (1-p')-p_z & 0 & 0 & (1-p')+p_z 
   \end{pmatrix}
   \end{align}

Amplitude Damping or the :math:`T_1` channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Amplitude damping is an energy dissipation (or relaxation) process. If a
qubit it in its excited state :math:`|1\rangle` it may emit energy, a
photon, and transition to the ground state :math:`|0\rangle`. In device
physics an experiment that measures the decay over some time :math:`t`,
with functional form :math:`\exp(-\Gamma t)`, is known as a :math:`T_1`
experiment (where :math:`T_1 = 1/\Gamma`).

From the perspective of quantum channels the amplitude damping channel
is interesting as is an example of a non-unital channel i.e. one that
does not have the identity matrix as a fixed point
:math:`\mathcal E_{AD} (I) \neq I`.

**Kraus**

The Kraus operators are

.. math::

   \begin{align}
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
   \end{align}

where
:math:`\sigma_- = (\sigma_+)^\dagger= \frac 1 2 (X +i Y) =|0\rangle \langle 1| `.
To relate this channel to a :math:`T_1` process we make the decay rate
time dependant :math:`\gamma(t) = \exp(-\Gamma t)`.

**:math:`\chi` matrix (process)**

.. math::

    \chi(AD) = [\chi_{ij}] = \frac 1 4\begin{pmatrix}  
   (1+\sqrt{1-\gamma})^2 & 0       & 0        & \gamma \\\\
   0                         & \gamma  & -i\gamma & 0\\\\ 
   0                         & i\gamma & \gamma   & 0\\\\ 
   \gamma                & 0  & 0        & (-1+\sqrt{1-\gamma})^2
   \end{pmatrix}

**Pauli-Liouville matrix**

.. math::


   R_{AD}= [(R_{AD})_{i,j}] =
   \begin{pmatrix}  
   1 & 0 & 0 & 0 \\\\
   0 & \sqrt{1-\gamma} & 0 & 0 \\\\ 
   0 & 0 & \sqrt{1-\gamma} & 0 \\\\ 
   \gamma & 0 & 0 & 1-\gamma 
   \end{pmatrix}

**Superoperator**

.. math::


   \begin{pmatrix}  
   1 & 0 & 0 & \gamma \\\\
   0 & \sqrt{1-\gamma} & 0 & 0\\\\ 
   0 & 0  & \sqrt{1-\gamma} & 0\\\\ 
   0 & 0 & 0 & 1-\gamma 
   \end{pmatrix}

**Choi**

.. math::

   \begin{align}
   \mathcal C &= \frac 1 2 ( |M_0\rangle\rangle\langle\langle M_0|+|M_1\rangle\rangle\langle\langle M_1|)\\\\
   &=\frac 1 2
   \begin{pmatrix}  
   1 & 0 & 0 & \sqrt{1-\gamma} \\\\
   0 & 0 & 0 & 0\\\\ 
   0 & 0  & \gamma & 0\\\\ 
   \sqrt{1-\gamma} & 0 & 0 & 1-\gamma 
   \end{pmatrix}
   \end{align}

Examples: Two qubit channels
----------------------------

This section will not be as comprehensive we only consider two channels
and two representations the operator sum representation (Kraus) and the
superoperator representation.

| **Kraus**
| The two channels we consider are:

(1) A unitary channel on one qubit

    .. math:: \mathcal U_{IZ}(\rho) = U_{IZ} \rho U_{IZ}^\dagger 

    with Kraus operator :math:`U_{IZ} = I\otimes Z = IZ`.

(2) A dephasing channel on one qubit

    .. math::  \mathcal E_{IZ}(\rho) = (1-p)II \rho II + p IZ \rho IZ,

    with Kraus operators :math:`M_0=\sqrt{1-p}II` and
    :math:`M_1= \sqrt{p}IZ`.

| **Superoperator**
| The superoperator representations for both channels are

.. math::

   \mathcal U_{IZ} = U_{IZ}^* \otimes U_{IZ} =
   {\rm diag}(1, -1, 1, -1, -1, 1, -1, 1, 1, -1,  1, -1, -1, 1, -1,  1)

| and
| 

.. math::

   \begin{align}
   \mathcal E_{IZ} &=
   (1-p)\,{\rm diag}(1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1,  1)+ \\\\
   &\quad p \,{\rm diag}(1, -1, 1, -1, -1, 1, -1, 1, 1, -1,  1, -1, -1, 1, -1, 1).
   \end{align}

References
----------

.. [IGST] Introduction to Quantum Gate Set Tomography.
    Greenbaum.
    arXiv:1509.02921, (2015).
    https://arxiv.org/abs/1509.02921

.. [QN] Quantum Nescimus. Improving the characterization of quantum systems from limited information.
    Harper.
    PhD thesis University of Sydney, 2018.
    https://ses.library.usyd.edu.au/handle/2123/17896

.. [GRAPTN] Tensor networks and graphical calculus for open quantum systems. 
    Wood et al.
    Quant. Inf. Comp. 15, 0579-0811 (2015).
    https://arxiv.org/abs/1111.6950

.. [SVDMAT] Singular value decomposition and matrix reorderings in quantum information theory.
    Miszczak.
    Int. J. Mod. Phys. C 22, No. 9, 897 (2011).
    https://dx.doi.org/10.1142/S0129183111016683
    https://arxiv.org/abs/1011.1585

.. [VECQO] Vectorization of quantum operations and its use.
    Gilchrist et al., arXiv:0911.2539, (2009).
    https://arxiv.org/abs/0911.2539

.. [MATQO] On the Matrix Representation of Quantum Operations.
    Nambu et al.
    arXiv: 0504091 (2005).
    https://arxiv.org/abs/quant-ph/0504091

.. [DUAL] On duality between quantum maps and quantum states.
    Zyczkowski et al.
    Open Syst. Inf. Dyn. 11, 3 (2004).
    https://dx.doi.org/10.1023/B:OPSY.0000024753.05661.c2
    https://arxiv.org/abs/quant-ph/0401119
