import numpy as np
import pandas as pd
from quspin.basis import spin_basis_general
from quspin.operators import exp_op, hamiltonian  # operators
from quspin.tools.measurements import ED_state_vs_time

# Physical paramters:
L = 10 # inp["system"]["Lx"]

J = -1.
dt = 0.01
tmax = 2

s = np.arange(L)  # sites [0,1,2,....]
x = s % L  # x positions for sites
T_x = (x + 1) % L 
Z = -(s + 1)  # spin inversion

basis = spin_basis_general(L,)

print("Size of symm. 2D H-space: {Ns:d}".format(Ns=basis.Ns))

###### setting up operators in hamiltonian ######

# Jzz_2d = [[-1.0, xy_to_id(x,y,Lx), xy_to_id((x+1)%Lx,y,Lx)] for x in range(Lx-1) for y in range(Ly)] + [
Y = [[1., i] for i in range(L)]
ZZ = [[J, i, i+1] for i in range(L-1)]
H = hamiltonian([["zz", ZZ], ["y", Y]], [], basis=basis, dtype=np.complex128)

X_obs = hamiltonian([["x", [[1./L, i] for i in range(L)]]],[],dtype=np.float64,basis=basis)
Z_obs = hamiltonian([["z", [[1./L, i] for i in range(L)]]],[],dtype=np.float64,basis=basis)
ZZ_obs = hamiltonian([["zz", [[1./L, i, T_x[i]] for i in range(L-1)] ]],[],dtype=np.float64,basis=basis)

psi0 = np.ones(basis.Ns, dtype=np.complex128)
psi0 /= np.linalg.norm(psi0)

E, V = H.eigh()

times = np.linspace(0.0,2.0,41)

psi = ED_state_vs_time(psi0, E, V, times , iterate=True)

# psi_full = np.array(basis.get_vec(psi[:,0]).todense())[:,0]
# print(psi_full.shape)
#
# dfPsi = pd.DataFrame( {
#     "psiR": np.real(psi_full),
#     "psiI": np.imag(psi_full),
# })
# dfPsi.to_csv("./psi0_L=%d_J=%f.csv" % (L,J), sep=' ')

obsX = [] 
obsZ = []
obsZZ = []
for i, psi_t in enumerate(psi):
    obsX.append(np.real(X_obs.expt_value(psi_t)))
    obsZ.append(np.real(Z_obs.expt_value(psi_t)))
    obsZZ.append(np.real(ZZ_obs.expt_value(psi_t)))

obsX=np.array(obsX)
obsZ=np.array(obsZ)
obsZZ=np.array(obsZZ)

print(obsZ)

dfTDVP = pd.DataFrame( {
    "time":       times,
    "xPol":       obsX,
    "zPol":       obsZ,
    "zz":         obsZZ,
})

dfTDVP.to_csv("./exact_L=%d_J=%f.csv" % (L,J), sep=' ')
dfPsi = pd.DataFrame( {
    "psi": psi[-1],
})
dfPsi.to_csv("./psi_L=%d_J=%f.csv" % (L,J), sep=' ')

# np.savetxt("exact_X_Lx=%d_Ly=%d.txt" % (Lx,Ly), obsX)
# np.savetxt("exact_Z_Lx=%d_Ly=%d.txt" % (Lx,Ly), obsZ)
# np.savetxt("exact_ZZ_Lx=%d_Ly=%d.txt" % (Lx,Ly), obsZZ)

#print(psi_t[:,1] / np.exp(1.j*np.angle(psi_t[0,1])))

#print("Z - ", np.real(Z_obs[0].expt_value(psi_t)))
#print("X -", np.real(X_obs[0].expt_value(psi_t)))
