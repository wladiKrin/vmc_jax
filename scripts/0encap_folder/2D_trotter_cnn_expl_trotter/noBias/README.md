# Simulating IBM's "quantum utility" circuit with NQS

## The circuit
We denote the number of qubits by $N$.

* Initial condition: $|\psi_0\rangle=|0\rangle^{\otimes N}$
* $X$-gates: $\hat U_X=\prod_i \exp\big(-i\theta_h\hat X_i\big)$
* $ZZ$-gates: $\hat U_{ZZ}=\prod_{\langle i,j\rangle} \exp\big(-i\theta_J\hat Z_i\hat Z_j\big)$

with fixed $\theta_J=\frac{\pi}{4}$.

One layer of the circuit is given by
$$\hat U = \hat U_{ZZ}\hat U_X$$

## Implementation with NQS

### Exact treatment of the $ZZ$-gates

The $ZZ$-gates can be applied exactly if the wave function is encoded in the $\hat Z$-basis:

$$\hat U_{ZZ}|\psi\rangle=\sum_s\psi(s)\hat U_{ZZ}|s\rangle=\sum_s\psi(s)e^{-i\theta_J \sum_{\langle i,j\rangle}s_is_j}|s\rangle$$

### The initial state

The initial condition: $|\psi_0\rangle=|0\rangle^{\otimes N}$ is hard to deal with using NQS with the computational $\hat Z$-basis. But we have

$$|\tilde\psi\rangle=
\hat U_X|\psi_0\rangle=
\bigotimes_i\bigg[\exp\big(-i\theta_h\hat X_i\big)|0\rangle\bigg]
=\bigotimes_i\bigg[\cos(\theta_h)|0\rangle-i\sin\theta_h|1\rangle\bigg]$$

We can find $|\tilde\psi\rangle$ as the ground state of
$$\hat H_{\text{init}}=\sum_i\big(\cos(2\theta_h)\hat Z_i+\sin(2\theta_h)\hat Y_i\big)\ .$$

### Variational $X$-gates
$X$-gates have to be applied variationally. To get the action of $\hat U_X$, we perform the TDVP with $\hat H_X=\sum_i\hat X_i$ up to time $t=\theta_h$.

### Overall procedure
1. Find $|\tilde\psi\rangle$, the ground state of $\hat H_{\text{init}}$.
2. Construct $|\psi_1\rangle=\hat U|\psi_0\rangle\equiv \hat U_{ZZ}|\tilde\psi\rangle$.
3. Perform TDVP to get $|\tilde\psi\rangle=\hat U_X|\psi_n\rangle$.
4. Construct $|\psi_{n+1}\rangle=\hat U_{ZZ}|\tilde\psi\rangle$. Continue with 3, unless desired depth is reached.