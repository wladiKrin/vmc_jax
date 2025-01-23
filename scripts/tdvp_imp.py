import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jaxlib
import matplotlib.pyplot as plt
import numpy as np

import jVMC
import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
from jVMC.stats import SampledObs


@dataclass
class Sample:
    configs: jaxlib.xla_extension.ArrayImpl
    coeffs: jaxlib.xla_extension.ArrayImpl
    weights: jaxlib.xla_extension.ArrayImpl


def realFun(x):
    return jnp.real(x)


def imagFun(x):
    return 0.5 * (x - jnp.conj(x))
                        

def transform_helper(x, rhsPrefactor, makeReal):
    return makeReal((-rhsPrefactor) * x)


class TDVP:
    """ This class provides functionality to solve a time-dependent variational principle (TDVP).

    With the force vector

        :math:`F_k=\langle \mathcal O_{\\theta_k}^* E_{loc}^{\\theta}\\rangle_c`

    and the quantum Fisher matrix

        :math:`S_{k,k'} = \langle (\mathcal O_{\\theta_k})^* \mathcal O_{\\theta_{k'}}\\rangle_c`

    and for real parameters :math:`\\theta\in\mathbb R`, the TDVP equation reads

        :math:`q\\big[S_{k,k'}\\big]\\dot\\theta_{k'} = -q\\big[xF_k\\big]`

    Here, either :math:`q=\\text{Re}` or :math:`q=\\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.

    For ground state search a regularization controlled by a parameter :math:`\\rho` can be included
    by increasing the diagonal entries and solving

        :math:`q\\big[(1+\\rho\delta_{k,k'})S_{k,k'}\\big]\\theta_{k'} = -q\\big[F_k\\big]`

    The `TDVP` class solves the TDVP equation by computing a pseudo-inverse of :math:`S` via
    eigendecomposition yielding

        :math:`S = V\Sigma V^\dagger`

    with a diagonal matrix :math:`\Sigma_{kk}=\sigma_k`
    Assuming that :math:`\sigma_1` is the smallest eigenvalue, the pseudo-inverse is constructed 
    from the regularized inverted eigenvalues

        :math:`\\tilde\sigma_k^{-1}=\\frac{1}{\\Big(1+\\big(\\frac{\epsilon_{SVD}}{\sigma_j/\sigma_1}\\big)^6\\Big)\\Big(1+\\big(\\frac{\epsilon_{SNR}}{\\text{SNR}(\\rho_k)}\\big)^6\\Big)}`

    with :math:`\\text{SNR}(\\rho_k)` the signal-to-noise ratio of :math:`\\rho_k=V_{k,k'}^{\dagger}F_{k'}` (see `[arXiv:1912.08828] <https://arxiv.org/pdf/1912.08828.pdf>`_ for details).

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``snrTol``: Regularization parameter :math:`\epsilon_{SNR}`, see above.
        * ``pinvTol``: Regularization parameter :math:`\epsilon_{SVD}` (see above) is chosen such that :math:`||S\\dot\\theta-F|| / ||F||<pinvTol`.
        * ``pinvCutoff``: Lower bound for the regularization parameter :math:`\epsilon_{SVD}`, see above.
        * ``makeReal``: Specifies the function :math:`q`, either `'real'` for :math:`q=\\text{Re}` or `'imag'` for :math:`q=\\text{Im}`.
        * ``rhsPrefactor``: Prefactor :math:`x` of the right hand side, see above.
        * ``diagonalShift``: Regularization parameter :math:`\\rho` for ground state search, see above.
        * ``crossValidation``: Perform cross-validation check as introduced in `[arXiv:2105.01054] <https://arxiv.org/pdf/2105.01054.pdf>`_.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
    """

    def __init__(self, sampler, snrTol=2, pinvTol=1e-14, pinvCutoff=1e-8, makeReal='imag', rhsPrefactor=1.j, diagonalShift=0., crossValidation=False, diagonalizeOnDevice=True):
        
        self.sampler = sampler
        self.snrTol = snrTol
        self.pinvTol = pinvTol
        self.pinvCutoff = pinvCutoff
        self.diagonalShift = diagonalShift
        self.rhsPrefactor = rhsPrefactor
        self.crossValidation = crossValidation
        self.key = jax.random.key(4321)

        self.diagonalizeOnDevice = diagonalizeOnDevice
        self.time = 0.

        self.metaData = None

        self.makeReal = realFun
        if makeReal == 'imag':
            self.makeReal = imagFun
        self.trafo_helper = partial(transform_helper, rhsPrefactor=rhsPrefactor, makeReal=self.makeReal)

        # pmap'd member functions
        self.makeReal_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda x: self.makeReal(x)))

    def set_diagonal_shift(self, delta):
        self.diagonalShift = delta

    def set_time(self, time):
        self.time = time

    def set_cross_validation(self, crossValidation=True):
        self.crossValidation = crossValidation

    def _get_tdvp_error(self, update):

        return jnp.abs(1. + jnp.real(update.dot(self.S0.dot(update)) - 2. * jnp.real(update.dot(self.F0))) / self.ElocVar0)

    def get_residuals(self):

        return self.metaData["tdvp_error"], self.metaData["tdvp_residual"]

    def get_snr(self):

        return self.metaData["SNR"]

    def get_spectrum(self):

        return self.metaData["spectrum"]

    def get_metadata(self):

        return self.metaData

    def get_energy_variance(self):

        return self.ElocVar0

    def get_energy_mean(self):

        return jnp.real(self.ElocMean0)

    def get_S(self):

        return self.S

    def get_tdvp_equation(self, Eloc, gradients):

        # self.ElocMean = Eloc.mean()[0]
        # self.ElocVar = Eloc.var()[0]
        #
        # self.F0 = (-self.rhsPrefactor) * (gradients.covar(Eloc).ravel()) #+ jnp.conj(gradients.mean()) * self.ElocMean)
        F = self.makeReal(self.F0)

        # self.S0 = gradients.covar()# + jnp.outer(jnp.conj(gradients.mean()),gradients.mean())
        S = self.makeReal(self.S0)

        # print("F: ", F)
        if self.diagonalShift > 1e-10:
            S = S + jnp.diag(self.diagonalShift * jnp.diag(S))

        return S, F

    def get_sr_equation(self, Eloc, gradients):
        return self.get_tdvp_equation(Eloc, gradients, rhsPrefactor=1.)

    def _transform_to_eigenbasis(self, S, F):
        
        if self.diagonalizeOnDevice:
            try:
                self.ev, self.V = jnp.linalg.eigh(S)
            except ValueError:
                warnings.warn("jax.numpy.linalg.eigh raised an exception. Falling back to numpy.linalg.eigh for "
                              "diagonalization.", RuntimeWarning)
                tmpS = np.array(S)
                tmpEv, tmpV = np.linalg.eigh(tmpS)
                self.ev = jnp.array(tmpEv)
                self.V = jnp.array(tmpV)
        else:
            tmpS = np.array(S)
            tmpEv, tmpV = np.linalg.eigh(tmpS)
            self.ev = jnp.array(tmpEv)
            self.V = jnp.array(tmpV)

        self.VtF = jnp.dot(jnp.transpose(jnp.conj(self.V)), F)

    def _get_snr(self, Eloc, gradients):

        EO = gradients.covar_data(Eloc).transform(
                        linearFun = jnp.transpose(jnp.conj(self.V)),
                        nonLinearFun=self.trafo_helper
                    )
        self.rhoVar = EO.var().ravel()

        self.snr = jnp.sqrt(jnp.abs(mpi.globNumSamples * (jnp.conj(self.VtF) * self.VtF) / self.rhoVar)).ravel()

    def solve(self, hamiltonian, psi, numSamples, t, outp):

        # Get sample
        sample = dict()
        for (i,k) in enumerate(self.sampler.keys()):
            self.start_timing(outp, "sampling"+str(k))
            sample[k] = Sample(*self.sampler[k].sample(numSamples=numSamples))
            self.stop_timing(outp, "sampling"+str(k), waitFor=sample[k].configs)

        if psi.logarithmic:
            self.start_timing(outp, "compute Eloc")
            Eloc = hamiltonian.get_O_loc(sample["rhs"].configs, psi, sample["rhs"].coeffs, t)
            self.stop_timing(outp, "compute Eloc", waitFor=Eloc)

            Eloc = SampledObs(Eloc, sample["rhs"].weights)
            self.ElocMean = Eloc.mean()[0]
            self.ElocVar = Eloc.var()[0]
            # print("Eloc: ", self.ElocMean)

            # Evaluate gradients
            self.start_timing(outp, "compute gradients")

            # lhs
            gradients = psi.gradients(sample["lhs"].configs)
            # self.S0 = mpi.global_sum(jVMC.stats._covar_helper(gradients, gradients)[:,None,...]) #* sample["lhs"].weights.ravel()[0]
            gradients = SampledObs( gradients, sample["lhs"].weights)
            self.S0 = gradients.covar()

            gradients = psi.gradients(sample["rhs"].configs)
            self.stop_timing(outp, "compute gradients", waitFor=gradients)
            gradients = SampledObs( gradients, sample["rhs"].weights)

            self.F0 = (-self.rhsPrefactor) * (gradients.covar(Eloc).ravel())
        else:
            self.start_timing(outp, "compute Eloc")

            ham_samples, matEls = hamiltonian.get_s_primes_unflattened(sample["rhs"].configs)
            # keys = jax.random.split(self.key,)

            newKey, self.key = jax.random.split(self.key,)
            probs = jnp.abs(matEls)**2
            probs /= jnp.sum(probs, axis=2)[:,:,None]
            probs_cum = jnp.cumsum(probs, axis=2)

            indices = jnp.sum(jax.random.uniform(newKey, shape=(probs.shape[1],))[None,:,None] > probs_cum, axis = 2)
            
            new_configs = jnp.take_along_axis(ham_samples, indices[:,:,None,None], axis=2)[:,:,0,:]
            new_coeffs = psi(new_configs)
            new_weights = jnp.full_like(new_coeffs, 1/new_coeffs.shape[1])

            ElocL = hamiltonian.get_O_loc(new_configs, psi, new_coeffs, t)
            ElocR = hamiltonian.get_O_loc(new_configs, psi, new_coeffs, t)

            self.stop_timing(outp, "compute Eloc", waitFor=ElocL)

            ElocL = SampledObs(ElocL, new_weights)
            ElocR = SampledObs(ElocR, new_weights)

            # self.ElocMean = jnp.sum(new_weights * ElocL.obs * jnp.abs(new_coeffs)**2)
            # self.ElocVar = jnp.real(jnp.sum(new_weights * jnp.abs(ElocL.obs* jnp.abs(new_coeffs)**2)**2)) - jnp.real(self.ElocMean)**2

            self.ElocMean = ElocR.mean()[0]
            self.ElocVar = ElocR.var()[0]

            # Evaluate gradients
            self.start_timing(outp, "compute gradients")
            gradientsR = psi.gradients(new_configs)
            gradsMeanR = jnp.sum(new_weights[:,:,None] * gradientsR / new_coeffs[:,:,None], axis = 1) #Eloc.mean()[0] 

            gradients = psi.gradients(new_configs)
            gradsMean = jnp.sum(new_weights[:,:,None] * gradients * jnp.conj(new_coeffs[:,:,None]), axis = 1) # gradsMeanR # 

            grads = jnp.sqrt(new_weights[:,:,None])*(gradients - new_coeffs[:,:,None]*gradsMean[None,:,:])
            # gradsR = jnp.sqrt(new_weights[:,:,None])*(gradientsR/new_coeffs[:,:,None] - gradsMeanR[None,:,:])

            # gradientsR = psi.gradients(new_configs)
            # gradsMeanR = jnp.sum(new_weights[:,:,None] * gradientsR / new_coeffs[:,:,None], axis = 1) #Eloc.mean()[0] 
            # gradsR = jnp.sqrt(new_weights[:,:,None])*(gradientsR - new_coeffs[:,:,None]*gradsMeanR[None,:,:])

            elocL = jnp.sqrt(new_weights[:,:,None])*(new_coeffs[:,:,None]*ElocL.obs[:,:,None] - new_coeffs[:,:,None]*self.ElocMean)
            # elocR = jnp.sqrt(new_weights[:,:,None])/new_coeffs[:,:,None]*(new_coeffs[:,:,None]*ElocR.obs[:,:,None] - new_coeffs[:,:,None]*self.ElocMean)
            # elocR = jnp.sqrt(new_weights[:,:,None])*(ElocR.obs[:,:,None] - self.ElocMean)

            self.S0 = mpi.global_sum(jVMC.stats._covar_helper(grads, grads)[:,None,...])

            # self.F0 = (-self.rhsPrefactor) * mpi.global_sum(jVMC.stats._covar_helper(gradsR, elocR)[:,None,...]).ravel()
            # FWvsq = (-self.rhsPrefactor) * mpi.global_sum(jVMC.stats._covar_helper(gradsR, elocR)[:,None,...]).ravel()
            FUnif = (-self.rhsPrefactor) * mpi.global_sum(jVMC.stats._covar_helper(grads, elocL)[:,None,...]).ravel()

            # FcovWvsq = (-self.rhsPrefactor) * mpi.global_sum(jVMC.stats._covar_helper(jnp.sqrt(new_weights[:,:,None])*(gradientsR/new_coeffs[:,:,None]), jnp.sqrt(new_weights[:,:,None])*ElocR.obs[:,:,None])[:,None,...]).ravel()
            # FcovUnif = (-self.rhsPrefactor) * mpi.global_sum(jVMC.stats._covar_helper(jnp.sqrt(new_weights[:,:,None])*gradients, jnp.sqrt(new_weights[:,:,None])*(new_coeffs[:,:,None]*ElocL.obs[:,:,None]))[:,None,...]).ravel()

            self.F0 = FUnif
            # self.F0 = FUnif
            # print("full vecs: ", np.linalg.norm(FWvsq - FUnif) / np.linalg.norm(FUnif))
            # print("covar part: ", np.linalg.norm(FcovWvsq - FcovUnif) / np.linalg.norm(FcovUnif))
            # print("means part: ", np.linalg.norm(FcovWvsq - FcovUnif) / np.linalg.norm(FcovUnif))

        # Get TDVP equation from MC data
        self.start_timing(outp, "solve TDVP eqn.")

        gradients = SampledObs( gradients, new_weights)
        self.S, F = self.get_tdvp_equation(ElocL, gradients)
        # print(self.S)
        # print(F)
        F.block_until_ready()

        # Transform TDVP equation to eigenbasis and compute SNR
        self._transform_to_eigenbasis(self.S, F) #, Fdata)
        self._get_snr(ElocL, gradients)

        # Discard eigenvalues below numerical precision
        
        self.invEv = jnp.where(jnp.abs(self.ev / self.ev[-1]) > 1e-14, 1. / self.ev, 0.)

        residual = 1.0
        cutoff = 1e-2
        F_norm = jnp.linalg.norm(F)
        while residual > self.pinvTol and cutoff > self.pinvCutoff:
            cutoff *= 0.8
            # Set regularizer for singular value cutoff
            regularizer = 1. / (1. + (max(cutoff, self.pinvCutoff) / jnp.abs(self.ev / self.ev[-1]))**6)

            # if not isinstance(self.sampler, jVMC.sampler.ExactSampler):
            #     # Construct a soft cutoff based on the SNR
            #     regularizer *= 1. / (1. + (self.snrTol / self.snr)**6)

            pinvEv = self.invEv * regularizer

            residual = jnp.linalg.norm((pinvEv * self.ev - jnp.ones_like(pinvEv)) * self.VtF) / F_norm

        update = jnp.real(jnp.dot(self.V, (pinvEv * self.VtF)))

        self.stop_timing(outp, "solve TDVP eqn.")
        # print("update: ", update)
        return update, residual, max(cutoff, self.pinvCutoff)

    def S_dot(self, v):

        return jnp.dot(self.S0, v)
    

    def start_timing(self, outp, name):
        if outp is not None:
            outp.start_timing(name)


    def stop_timing(self, outp, name, waitFor=None):
        if waitFor is not None:
            waitFor.block_until_ready()
        if outp is not None:
            outp.stop_timing(name)


    def __call__(self, netParameters, t, *, psi, hamiltonian, **rhsArgs):
        """ For given network parameters this function solves the TDVP equation.

        This function returns :math:`\\dot\\theta=S^{-1}F`. Thereby an instance of the ``TDVP`` class is a suited
        callable for the right hand side of an ODE to be used in combination with the integration schemes 
        implemented in ``jVMC.stepper``. Alternatively, the interface matches the scipy ODE solvers as well.

        Arguments:
            * ``netParameters``: Parameters of the NQS.
            * ``t``: Current time.
            * ``psi``: NQS ansatz. Instance of ``jVMC.vqs.NQS``.
            * ``hamiltonian``: Hamiltonian operator, i.e., an instance of a derived class of ``jVMC.operator.Operator``. \
                                *Notice:* Current time ``t`` is by default passed as argument when computing matrix elements. 

        Further optional keyword arguments:
            * ``numSamples``: Number of samples to be used by MC sampler.
            * ``outp``: An instance of ``jVMC.OutputManager``. If ``outp`` is given, timings of the individual steps \
                are recorded using the ``OutputManger``.
            * ``intStep``: Integration step number of multi step method like Runge-Kutta. This information is used to store \
                quantities like energy or residuals at the initial integration step.

        Returns:
            The solution of the TDVP equation, :math:`\\dot\\theta=S^{-1}F`.
        """

        tmpParameters = psi.get_parameters()
        sp = np.zeros((1,1,5))
        # print("tmp: ", tmpParameters)
        # print("net: ", netParameters)
        # print("before: ", psi(sp))
        psi.set_parameters(netParameters)
        # print("after: ", psi(sp))

        outp = None
        if "outp" in rhsArgs:
            outp = rhsArgs["outp"]
        self.outp = outp

        numSamples = None
        if "numSamples" in rhsArgs:
            numSamples = rhsArgs["numSamples"]

        update, solverResidual, pinvCutoff = self.solve(hamiltonian, psi, numSamples, t, outp)#Eloc, sampleGradients)
        

        if outp is not None:
            outp.add_timing("MPI communication", mpi.get_communication_time())

        psi.set_parameters(tmpParameters)
        # print(tmpParameters)

        if "intStep" in rhsArgs:
            if rhsArgs["intStep"] == 0:

                self.ElocMean0 = self.ElocMean
                self.ElocVar0 = self.ElocVar

                self.metaData = {
                    "tdvp_error": self._get_tdvp_error(update),
                    "tdvp_residual": solverResidual,
                    "pinv_cutoff": pinvCutoff,
                    "SNR": self.snr, 
                    "spectrum": self.ev,
                }

        return update
