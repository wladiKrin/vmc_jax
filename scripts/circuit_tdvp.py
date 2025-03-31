import json
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import my_tdvp
import numpy as np
from nets.zz_wrapper import BlochWrapperCpx, CpxRBM, LinkList, ZZWrapper

import jVMC
import jVMC.operator as op
from jVMC.util import ground_state_search, measure


def xy_to_id(x,y,L):
    return int(x + L * y)

def norm_fun(v, df=lambda x: x):
    return jnp.real(jnp.conj(jnp.transpose(v)).dot(df(v)))

inp = None
if len(sys.argv) > 1:
    # if an input file is given
    with open(sys.argv[1], 'r') as f:
        inp = json.load(f)
else:

    if mpi.rank == 0:
        print("Error: No input file given.")
        exit()

#wdir = inp["general"]["working_directory"]

# Physical paramters:
Lx = inp["system"]["Lx"]
Ly = inp["system"]["Lx"]

J = -1
h = -2*J
dt = 0.25
theta_J = dt*J
theta_h = dt*h
theta_init = jnp.pi / 18

circuitDepth = inp["simulation"]["circuitDepth"] 

# Network parameters
numHidden  = inp["net"]["num_hidden"]
numSamples = inp["net"]["num_samples"]

# Initialize output manager
outp = jVMC.util.OutputManager("./HT_Lx=%d_Ly=%d_h=%.3f_depth=%d_numHidden=%d.hdf5" % (Lx,Ly,theta_h,circuitDepth,numHidden), append=False)

dt = 1e-3
maxDt = 1e-2

links = [] #LinkList( jnp.array([[i,i+1] for i in range(L-1)]) )
for x in range(Lx):
    for y in range(Ly):
        links.append(xy_to_id(x,y,Lx), xy_to_id((x+1)%Lx,y,Lx))
        links.append(xy_to_id(x,y,Lx), xy_to_id(x,(y+1)%Ly,Lx))

links = LinkList( jnp.array(links))

net0 = CpxRBM(numHidden=numHidden, bias=True)
net = BlochWrapperCpx(net=net0, theta=theta_h+theta_init)
wrapnet = ZZWrapper(net=net, links=links, angle=0)
psi = jVMC.vqs.NQS(wrapnet, seed=1234)
psi(jnp.ones((1,1,Lx*Ly), dtype=int))

# Set up hamiltonian for initial state (after one application of X-gates)
hamiltonianGS = jVMC.operator.BranchFreeOperator()
# for l in range(L):
#     hamiltonianGS.add(op.scal_opstr(np.cos(2*theta_h), (op.Sz(l), )))
#     hamiltonianGS.add(op.scal_opstr(-np.sin(2*theta_h), (op.Sy(l), )))

for x in range(Lx):
    for y in range(Ly):
        hamiltonianGS.add(op.scal_opstr(np.cos(2*(theta_h+theta_init)),  (op.Sz(xy_to_id(x,y,Lx)), )))
        hamiltonianGS.add(op.scal_opstr(-np.sin(2*(theta_h+theta_init)), (op.Sy(xy_to_id(x,y,Lx)), )))

# Set up hamiltonian for one-qubit gates
hamiltonian = jVMC.operator.BranchFreeOperator()

# for l in range(L):
#     hamiltonian.add(op.scal_opstr(-1., (op.Sx(l), )))

for x in range(Lx):
    for y in range(Ly):
        hamiltonian.add(op.scal_opstr(-1.,  (op.Sx(xy_to_id(x,y,Lx)), )))

# Set up observables
observables = {
    "energy": hamiltonian,
    "Z": jVMC.operator.BranchFreeOperator(),
    "X": jVMC.operator.BranchFreeOperator(),
    "ZZ": jVMC.operator.BranchFreeOperator(),
}

for x in range(Lx):
    for y in range(Ly):
        observables["X"][l].add(op.scal_opstr(1. / (Lx*Ly), (op.Sx(xy_to_id(x,y,L)), )))
        observables["Z"][l].add(op.scal_opstr(1. / (Lx*Ly), (op.Sz(xy_to_id(x,y,L)), )))
        observables["ZZ"].add(op.scal_opstr(1. / (Lx*Ly), (op.Sz(xy_to_id(x,y,Lx)), op.Sz(xy_to_id((x+1)%Lx,y,Lx)))))
        observables["ZZ"].add(op.scal_opstr(1. / (Lx*Ly), (op.Sz(xy_to_id(x,y,Lx)), op.Sz(xy_to_id(x,(y+1)%Ly,Lx)))))

# Set up exact sampler
# sampler = jVMC.sampler.ExactSampler(psi, (Lx,Ly))
sampler = jVMC.sampler.MCSampler(psi, (Lx*Ly,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=25, sweepSteps=Lx*Ly,
                                 numSamples=numSamples, thermalizationSweeps=25)

# Initial state search
outp.print("** Ground state search")
outp.set_group("ground_state_search")

tdvpEquation = jVMC.util.TDVP(sampler, snrTol=1,
                            pinvTol=1e-6,
                            rhsPrefactor=1., diagonalShift=10,
                            makeReal="real", diagonalizeOnDevice=True)

ground_state_search(psi, hamiltonianGS, tdvpEquation, sampler,
                    numSteps=500, varianceTol=1e-6 * Lx**2 * Ly**2,
                    stepSize=1e-2, observables=None, outp=outp)

# Time evolution
outp.print("** Time evolution")
outp.set_group("time_evolution")


tdvpParams = {"svdTol": 1e-5, "rhsPrefactor": -1.j, "makeReal": "imag", "diagonalShift": 0}
# Set up TDVP
tdvpEquation = jVMC.util.TDVP(sampler, rhsPrefactor=1.j)

# Set up stepper
stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=1e-3, tol=1e-5, maxStep=maxDt)

# Get initial parameters
params = psi.get_parameters().copy()

obs = measure(observables, psi, sampler)
outp.print("GS Z-polarization  %d\n" % (obs["Z"]))
outp.print("GS ZZ-correlation  %d\n" % (obs["ZZ"]))

for circuitStep in range(circuitDepth):
    outp.print("### Circuit step  %d\n" % (circuitStep))
    # Diagonal part
    wrapnet = ZZWrapper(net=net, links=links, angle=(circuitStep+1) * theta_J)
    psi = jVMC.vqs.NQS(wrapnet, seed=1234)
    # sampler = jVMC.sampler.ExactSampler(psi, (Lx,Ly))
    sampler = jVMC.sampler.MCSampler(psi, (Lx*Ly,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                     numChains=25, sweepSteps=Lx*Ly,
                                     numSamples=numSamples, thermalizationSweeps=25)
    tdvpEquation = my_tdvp.TDVP(sampler, **tdvpParams)
    psi.set_parameters(params)

    t = 0.
    while t < theta_h:
        stepper.maxStep = min(maxDt, theta_h-t+1e-8)
        stepper.dt = min(stepper.dt, theta_h-t)

        T = circuitStep * theta_h + t
        tic = time.perf_counter()
        outp.print(">  t = %f\n" % (t))

        # Measure observables
        outp.start_timing("measure observables")
        obs = measure(observables, psi, sampler)
        outp.stop_timing("measure observables")

        # TDVP step
        dp, dt = stepper.step(t, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, 
                                numSamples=1000, outp=outp, 
                                normFunction=partial(norm_fun, df=tdvpEquation.S_dot))

        psi.set_parameters(dp)
        t += dt
        energy = tdvpEquation.get_energy_mean()
        energyVar = tdvpEquation.get_energy_variance()
        outp.print("   Time step size: dt = %f" % (dt))
        tdvpErr, tdvpRes = tdvpEquation.get_residuals()
        outp.print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))

        ## Write observables
        obs["energy"] = {}
        obs["energy"]["mean"] = jnp.array([energy])
        obs["energy"]["variance"] = jnp.array([energyVar])
        obs["energy"]["MC_error"] = jnp.array([0.0])
        acc_rate = jnp.array([0.0])
        # if inp["sampler"]["type"] == "MC":
        #     obs["energy"]["MC_error"] = jnp.reshape(np.sqrt(energyVar) / jnp.sqrt(sampler.get_last_number_of_samples()), (1,))
        #     acc_rate = sampler.acceptance_ratio()
        outp.write_observables((T-dt) / theta_h, **obs)
        
        # Write metadata
        outp.write_metadata((T-dt) / theta_h, **tdvpEquation.get_metadata(), acc_rate=acc_rate)
        
        # Write network parameters
        outp.write_network_checkpoint(T / theta_h, psi.get_parameters())

        outp.print("    Energy = %f +/- %f" % (obs["energy"]["mean"], obs["energy"]["MC_error"]))
        outp.print("    Energy variance = %f" % (obs["energy"]["variance"]))

        outp.print_timings(indent="   ")

        toc = time.perf_counter()
        outp.print("   == Total time for this step: %fs\n" % (toc - tic))

    outp.print("")
    outp.print("** Final t / theta_h = %.3f" % (t/theta_h))
    # Store parameters
    params = psi.get_parameters().copy()

wrapnet = ZZWrapper(net=net, links=links, angle=(circuitDepth) * theta_J)
psi = jVMC.vqs.NQS(wrapnet, seed=1234)
psi(jnp.ones((1,1,Lx*Ly), dtype=int))
psi.set_parameters(params)

observables["energy"] = hamiltonian
obs = measure(observables, psi, sampler)
outp.write_observables(circuitDepth, **obs)

