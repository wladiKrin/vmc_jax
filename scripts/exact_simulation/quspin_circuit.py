from quspin.operators import hamiltonian, exp_op # operators
from quspin.basis import spin_basis_general

import numpy as np


def get_unitary(mat, angle):

    ev, V = np.linalg.eigh(mat)

    return V.dot( np.diag(np.exp(-1.j * angle * ev)).dot(V.conj().T) )

L=6
theta_J = np.pi/4
theta_h = np.pi/8
circuitDepth = 5

X_steps = 20

basis = spin_basis_general(L)

ZZ = [ ["zz", [[1., i, i+1] for i in range(L-1)]] ]
ham_ZZ = hamiltonian(ZZ,[],dtype=np.float64,basis=basis)

X = [ ["x", [[1., i] for i in range(L)]] ]
ham_X = hamiltonian(X,[],dtype=np.float64,basis=basis)

X_obs = [ hamiltonian([ ["x", [[1, i]]]],[],dtype=np.float64,basis=basis) for i in range(L) ]
Z_obs = [ hamiltonian([ ["z", [[1, i]]]],[],dtype=np.float64,basis=basis) for i in range(L) ]

ZZ_mat = ham_ZZ.toarray()
X_mat = ham_X.toarray()

U_ZZ = get_unitary(ham_ZZ.toarray(), theta_J)
U_X = get_unitary(ham_X.toarray(), theta_h)
U_X_step = get_unitary(ham_X.toarray(), theta_h/X_steps)


U = U_X.dot(U_ZZ)
U = U_ZZ.dot(U_X)

psi0 = np.ones(basis.Ns, dtype=np.complex128)
psi0 /= np.linalg.norm(psi0)


psi0 = np.zeros(basis.Ns, dtype=np.complex128)
psi0[-1] = 1.

psi0 = U_X.dot(psi0)
psi0 = U_ZZ.dot(psi0)

psi_t = psi0.copy()

#psi_t = np.zeros((basis.Ns, circuitDepth+1), dtype=np.complex128)
#psi_t[:,0] = psi0

obsX = [[0.] + [np.real(o.expt_value(psi_t)) for o in X_obs]]
obsZ = [[0.] + [np.real(o.expt_value(psi_t)) for o in Z_obs]]
for i in range(circuitDepth):

    #psi_t[:,i+1] = U.dot(psi_t[:,i]).copy()

    for t in np.arange(X_steps) / X_steps:

        psi_t = U_X_step.dot(psi_t)

        obsX.append([(i) + t + 1./X_steps] + [np.real(o.expt_value(psi_t)) for o in X_obs])
        obsZ.append([(i) + t + 1./X_steps] + [np.real(o.expt_value(psi_t)) for o in Z_obs])

    psi_t = U_ZZ.dot(psi_t)

obsX.append([(circuitDepth)] + [np.real(o.expt_value(psi_t)) for o in X_obs])
obsZ.append([(circuitDepth)] + [np.real(o.expt_value(psi_t)) for o in Z_obs])
obsX=np.array(obsX)
obsZ=np.array(obsZ)

np.savetxt("exact_X_L=%d.txt" % (L), obsX)
np.savetxt("exact_Z_L=%d.txt" % (L), obsZ)

#print(psi_t[:,1] / np.exp(1.j*np.angle(psi_t[0,1])))

#print("Z - ", np.real(Z_obs[0].expt_value(psi_t)))
#print("X -", np.real(X_obs[0].expt_value(psi_t)))
