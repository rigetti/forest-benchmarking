# Conventions: labeling of tensor product structure of qubits

In this note book we explore the different ways to label qubit Hilbert spaces in a tensor product. This becomes important when doing QCVV as different labeling of qubits can create confusion about which state or process is the ideal.

It is largely motivated by the compelling arguments given in [Simith17]. Unfortunately not everyone in the world uses this convetion so we need to be explicit about it here.


[SMITH17]  Someone shouts, "|01000>!" Who is Excited?  
			 	Robert S. Smith  
				https://arxiv.org/abs/1711.02086  
				

# Smith ordering

Suppose you order the labels of quantum systems in the following way:

$$H_n :=\mathcal H_n \otimes \mathcal H_{n-1} \otimes \ldots \otimes \mathcal H_1 \otimes \mathcal H_0.$$


Then if you add a new system, Smith argues that you should extend the Hilbert space to the left: $\mathcal H_{n+1} \otimes H_n$.

# Standard ordering

A fairly typical ordering seen in the literature and various code bases is
$$\mathcal H_0 \otimes \mathcal H_1 \otimes \ldots \otimes \mathcal H_{n-1} \otimes \mathcal H_n.$$



# Abstract CNOT action
We can define the action of a CNOT by how it acts on the control and target qubit independent of the labels of the control and target. 
First we define   
$|0\rangle = (1,0)^T$,  
$|1\rangle = (0,1)^T$,  
$\Pi_0 = |0\rangle \langle 0|$,  
$\Pi_1 = |1\rangle \langle 1|$,  
$I = |0\rangle \langle 0| + |1\rangle \langle 1|$, and   
$X = |1\rangle \langle 0| + |0\rangle \langle 1|$ 

Then with out any explicit ordering we can write down the definition of a CNOT

$${\rm CNOT(control, target)}:= \Pi_{0,{\rm control}} \otimes I_{\rm target}+\Pi_{1,{\rm control}} \otimes X_{\rm target}$$

## In Smith ordering

Now we consider a two qubit Hilbert space $ \mathcal H_1 \otimes \mathcal H_0$.

$$ {\rm CNOT}(0,1) = 
\begin{pmatrix} 
1& 0& 0& 0\\\\
0& 0& 0& 1\\\\
0& 0& 1& 0\\\\
0& 1& 0& 0\\\\
\end{pmatrix}
$$

$$ {\rm CNOT}(1,0) = \begin{pmatrix} 
1& 0& 0& 0\\\\
0& 1& 0& 0\\\\
0& 0& 0& 1\\\\
0& 0& 1& 0\\\\
\end{pmatrix}
$$

## In standard ordering
$$ {\rm CNOT}(0,1) = 
\begin{pmatrix} 
1& 0& 0& 0\\\\
0& 1& 0& 0\\\\
0& 0& 0& 1\\\\
0& 0& 1& 0\\\\
\end{pmatrix}
$$

$$ {\rm CNOT}(1,0) = 
\begin{pmatrix} 
1& 0& 0& 0\\\\
0& 0& 0& 1\\\\
0& 0& 1& 0\\\\
0& 1& 0& 0\\\\
\end{pmatrix}
$$
