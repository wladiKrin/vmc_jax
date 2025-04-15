import os
import json
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
# from nets.zz_wrapper import BlochWrapperCpx, CpxRBM, LinkList, ZZWrapper
import flax
import flax.linen as nn
from nets.RBMCNN import CpxRBMCNNLog

import jVMC
import jVMC.operator as op
import jVMC.mpi_wrapper as mpi
import jVMC.nets.activation_functions as act_funs
import pandas as pd
from jVMC.util import ground_state_search, measure

print(jax.local_devices())

def xy_to_id(x,y,L):
    return int(x + L * y)

def norm_fun(v, df=lambda x: x):
    return jnp.real(jnp.conj(jnp.transpose(v)).dot(df(v)))

inp = None
with open("input.json", 'r') as f:
    inp = json.load(f)

# Physical paramters:
Lx = inp["system"]["Lx"]
Ly = inp["system"]["Ly"]

J = -1.
h = -2.*J
deltat = 0.25
theta_J = deltat*J
theta_h = deltat*h
theta_init = jnp.pi / 18

i = int(os.environ["ENCAP_PROCID"])

circuitDepth = inp["simulation"]["circuitDepth"] 
tol = inp["simulation"]["tol"][1]
pinvTol = inp["simulation"]["pinvTol"]

# Network parameters
filtersize  = inp["net"]["filtersize"]
numChannels  = inp["net"]["num_channels"]
numSamples = inp["net"]["num_samples"]

# param_name = "CpxRBM_HT_Lx=%d_Ly=%d_h=%.3f_depth=%d_numChannels=%d_filtersize=%d_numSamples=%d" % (Lx,Ly,theta_h,circuitDepth,numChannels, filtersize,numSamples)
param_name = "CpxRBMCNN_HT_exact_Lx=%d_Ly=%d_h=%.3f_depth=%d_tol=%e_pinvTol=%e_numChannels=%d_filtersize=%d" % (Lx,Ly,theta_h,circuitDepth,tol,pinvTol,numChannels,filtersize)

dt = 1e-3
maxDt = 1e-0


net = CpxRBMCNNLog(
    F=(filtersize,filtersize), 
    channels=(numChannels,), 
    strides=(1,1),
    #actFun=(act_funs.log_poly5,),
    firstLayerBias=False,
    bias=False,
    Lx=Lx,
    Ly=Ly,
)
psi = jVMC.vqs.NQS(net, seed=1234)
psi(jnp.ones((1,1,Lx,Ly), dtype=int))

# Set up hamiltonian for initial state (after one application of X-gates)
hamiltonianGS = jVMC.operator.BranchFreeOperator()
for x in range(Lx):
    for y in range(Ly):
        hamiltonianGS.add(op.scal_opstr(jnp.cos(2*(theta_h+theta_init)),  (op.Sz(xy_to_id(x,y,Lx)), )))
        hamiltonianGS.add(op.scal_opstr(-jnp.sin(2*(theta_h+theta_init)), (op.Sy(xy_to_id(x,y,Lx)), )))

# Set up hamiltonian for one-qubit gates
hamiltonianX  = jVMC.operator.BranchFreeOperator()
hamiltonianZZ = jVMC.operator.BranchFreeOperator()
for x in range(Lx):
    for y in range(Ly):
        hamiltonianX.add(op.scal_opstr(-1.,  (op.Sx(xy_to_id(x,y,Lx)), )))

        hamiltonianZZ.add(op.scal_opstr(-1., (op.Sz(xy_to_id(x,y,Lx)), op.Sz(xy_to_id((x+1)%Lx,y,Lx)))))
        hamiltonianZZ.add(op.scal_opstr(-1., (op.Sz(xy_to_id(x,y,Lx)), op.Sz(xy_to_id(x,(y+1)%Ly,Lx)))))


# Set up observables
observables = {
    "Z": jVMC.operator.BranchFreeOperator(),
    "X": jVMC.operator.BranchFreeOperator(),
    "ZZ": jVMC.operator.BranchFreeOperator(),
}

for x in range(Lx):
    for y in range(Ly):
        observables["X"].add(op.scal_opstr(1. / (Lx*Ly), (op.Sx(xy_to_id(x,y,Lx)), )))
        observables["Z"].add(op.scal_opstr(1. / (Lx*Ly), (op.Sz(xy_to_id(x,y,Lx)), )))
        observables["ZZ"].add(op.scal_opstr(1. / (Lx*Ly), (op.Sz(xy_to_id(x,y,Lx)), op.Sz(xy_to_id((x+1)%Lx,y,Lx)))))
        observables["ZZ"].add(op.scal_opstr(1. / (Lx*Ly), (op.Sz(xy_to_id(x,y,Lx)), op.Sz(xy_to_id(x,(y+1)%Ly,Lx)))))

# Set up exact sampler
sampler = jVMC.sampler.ExactSampler(psi, (Lx*Ly))
# sampler = jVMC.sampler.MCSampler(psi, (Lx*Ly,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
#                                  numChains=25, sweepSteps=Lx*Ly,
#                                  numSamples=numSamples, thermalizationSweeps=25)

# Initial state search
print("** Ground state search")

tdvpEquation = jVMC.util.TDVP(sampler, snrTol=1,
                            pinvTol=1e-6,
                            rhsPrefactor=1., diagonalShift=10,
                            makeReal="real", diagonalizeOnDevice=True)

# ground_state_search(
#     psi, 
#     hamiltonianGS, 
#     tdvpEquation, 
#     sampler, 
#     numSteps=500, 
#     varianceTol=1e-8 * Lx**2 * Ly**2, 
#     stepSize=1e-2, 
#     observables=None,
# ) #, outp=outp)
# np.save("./parameters_GS", np.array(psi.get_parameters()))

# Get initial parameters
psi.set_parameters(np.load("./parameters_GS.npy"))

obs = measure(observables, psi, sampler)

# Time evolution
print("** Time evolution")
tdvpEquation = jVMC.util.TDVP(sampler, rhsPrefactor=-1.j, pinvTol=pinvTol)

# Set up stepper
stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=1e-4, tol=tol, maxStep=maxDt)

data = []
data.append([0, 
    0, # obs["energy"]["mean"][0], 
    0, # obs["energy"]["variance"][0], 
    0, # obs["energy"]["MC_error"][0], 
    obs["Z"]["mean"][0],
    obs["Z"]["variance"][0], 
    obs["Z"]["MC_error"][0], 
    obs["ZZ"]["mean"][0],
    obs["ZZ"]["variance"][0], 
    obs["ZZ"]["MC_error"][0], 
    obs["X"]["mean"][0],
    obs["X"]["variance"][0], 
    obs["X"]["MC_error"][0], 
    0, 0, 0],
)

print("GS Z-polarization: ", obs["Z"])
print("GS ZZ-correlation: ", obs["ZZ"])

for circuitStep in range(circuitDepth):

    print("### Circuit step  %d\n" % (circuitStep))
    # Diagonal part
    print("Diagonal part")
    t = 0
    while t < abs(theta_J):
        stepper.maxStep = min(maxDt, abs(theta_J-t+1e-8))
        stepper.dt = min(stepper.dt, abs(theta_J-t))

        tic = time.perf_counter()
        print(">  t = %f\n" % (t))

        # Measure observables
        obs = measure(observables, psi, sampler)

        # TDVP step
        dp, dt = stepper.step(t, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonianZZ, psi=psi, 
                                numSamples=numSamples, # outp=outp, 
                                normFunction=partial(norm_fun, df=tdvpEquation.S_dot))

        psi.set_parameters(dp)
        t += dt
        energy = tdvpEquation.get_energy_mean()
        energyVar = tdvpEquation.get_energy_variance()
        print("   Time step size: dt = %f" % (dt))
        tdvpErr, tdvpRes = tdvpEquation.get_residuals()
        print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))

        ## Write observables
        print("saving time: ", np.abs((t-dt)/theta_J) + 2*circuitStep)
        data.append([np.abs((t-dt)/theta_J) + 2*circuitStep, #(T-dt) / theta_h, 
            np.sum(np.real(energy)),
            np.sum(np.real(energyVar)),
            0,
            obs["Z"]["mean"][0],
            obs["Z"]["variance"][0], 
            obs["Z"]["MC_error"][0], 
            obs["ZZ"]["mean"][0],
            obs["ZZ"]["variance"][0], 
            obs["ZZ"]["MC_error"][0], 
            obs["X"]["mean"][0],
            obs["X"]["variance"][0], 
            obs["X"]["MC_error"][0], 
            tdvpErr, tdvpRes, dt],
        )

        npdata   = np.array(data)

        dfTDVP = pd.DataFrame( {
            "time":       npdata[:, 0],
            "energy":     npdata[:, 1],
            "energy_var": npdata[:, 2],
            "energy_MC":  npdata[:, 3],
            "zPol":       npdata[:, 4],
            "zPol_var":   npdata[:, 5],
            "zPol_MC":    npdata[:, 6],
            "zz":         npdata[:, 7],
            "zz_var":     npdata[:, 8],
            "zz_MC":      npdata[:, 9],
            "xPol":       npdata[:, 10],
            "xPol_var":   npdata[:, 11],
            "xPol_MC":    npdata[:, 12],
            "tdvpErr":    npdata[:, 13],
            "tdvpRes":    npdata[:, 14],
            "dt":         npdata[:, 15],
        })

        dfTDVP.to_csv("./data_"+param_name+".csv", sep=' ')

        print("    Energy = ", energy)
        print("    Energy variance = ", energyVar)

        toc = time.perf_counter()
        print("   == Total time for this step: ", (toc - tic))
    np.save("./parameters_after_ZZ_circuitStep_" + str(circuitStep), np.array(psi.get_parameters()))

    # stepper.dt = 1e-4

    # Off-diagonal part
    print("Off-diagonal part")
    t=0
    while t < theta_h:
        stepper.maxStep = min(maxDt, theta_h-t+1e-8)
        stepper.dt = min(stepper.dt, theta_h-t)

        tic = time.perf_counter()
        print(">  t = %f\n" % (t))

        # Measure observables
        obs = measure(observables, psi, sampler)

        # TDVP step
        dp, dt = stepper.step(t, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonianX, psi=psi, 
                                numSamples=numSamples, # outp=outp, 
                                normFunction=partial(norm_fun, df=tdvpEquation.S_dot))

        psi.set_parameters(dp)
        t += dt
        energy = tdvpEquation.get_energy_mean()
        energyVar = tdvpEquation.get_energy_variance()
        print("   Time step size: dt = %f" % (dt))
        tdvpErr, tdvpRes = tdvpEquation.get_residuals()
        print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))

        ## Write observables
        print("saving time: ", np.abs((t-dt)/theta_h) + 2*circuitStep+1)
        data.append([np.abs((t-dt)/theta_h) + 2*circuitStep+1, #(T-dt) / theta_h, 
            np.sum(np.real(energy)),
            np.sum(np.real(energyVar)),
            0,
            obs["Z"]["mean"][0],
            obs["Z"]["variance"][0], 
            obs["Z"]["MC_error"][0], 
            obs["ZZ"]["mean"][0],
            obs["ZZ"]["variance"][0], 
            obs["ZZ"]["MC_error"][0], 
            obs["X"]["mean"][0],
            obs["X"]["variance"][0], 
            obs["X"]["MC_error"][0], 
            tdvpErr, tdvpRes, dt],
        )

        npdata   = np.array(data)

        dfTDVP = pd.DataFrame( {
            "time":       npdata[:, 0],
            "energy":     npdata[:, 1],
            "energy_var": npdata[:, 2],
            "energy_MC":  npdata[:, 3],
            "zPol":       npdata[:, 4],
            "zPol_var":   npdata[:, 5],
            "zPol_MC":    npdata[:, 6],
            "zz":         npdata[:, 7],
            "zz_var":     npdata[:, 8],
            "zz_MC":      npdata[:, 9],
            "xPol":       npdata[:, 10],
            "xPol_var":   npdata[:, 11],
            "xPol_MC":    npdata[:, 12],
            "tdvpErr":    npdata[:, 13],
            "tdvpRes":    npdata[:, 14],
            "dt":         npdata[:, 15],
        })

        dfTDVP.to_csv("./data_"+param_name+".csv", sep=' ')
        
        # Write network parameters
        # outp.write_network_checkpoint(T / theta_h, psi.get_parameters())

        print("    Energy = ", energy)
        print("    Energy variance = ", energyVar)

        toc = time.perf_counter()
        print("   == Total time for this step: ", (toc - tic))
    np.save("./parameters_after_X_circuitStep_" + str(circuitStep), np.array(psi.get_parameters()))
