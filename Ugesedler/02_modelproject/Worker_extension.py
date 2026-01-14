import numpy as np
from types import SimpleNamespace

from Worker import WorkerClass  # samme navn som din fil


class WorkerExtension(WorkerClass):
    """
    Extension of WorkerClass with heterogeneous disutility of labor.

    nu_i(p) = nu * (1 + alpha_nu * (p - 1))

    alpha_nu > 0: more productive workers dislike work more
    alpha_nu < 0: more productive workers dislike work less
    alpha_nu = 0: baseline model.
    """

    def setup_worker(self):
        """
        Start from the original setup_worker and then add alpha_nu.
        """
        super().setup_worker()
        par = self.par

        # extra parameter for heterogeneity in disutility of work
        par.alpha_nu = 0.0

    # ----- new pieces for heterogeneity -----

    def nu_i(self, p):
        """
        Individual-specific disutility parameter as a function of productivity p.
        """
        par = self.par
        nu_i = par.nu * (1.0 + par.alpha_nu * (p - 1.0))
        # make sure it stays positive
        return max(nu_i, 1e-8)

    def FOC(self, p, ell):
        """
        First-order condition with heterogeneous nu_i(p):

            dU/dℓ = w p (1-τ)/c - nu_i(p) * ℓ^ε = 0
        """
        par = self.par
        c = self.post_tax_income(p, ell)
        nu_i = self.nu_i(p)

        FOC_val = par.w * p * (1.0 - par.tau) / c - nu_i * (ell ** par.epsilon)
        return FOC_val

    def optimal_choice_FOC(self, p):
        """
        Use the parent's root-finder to get ℓ*(p), but recompute utility
        using nu_i(p) instead of the common nu.
        """
        # first, call the original algorithm; it now uses our new FOC
        opt_base = super().optimal_choice_FOC(p)

        par = self.par
        nu_i = self.nu_i(p)

        # recompute utility with heterogeneous nu_i(p)
        ell_star = opt_base.ell
        c_star = opt_base.c

        U_star = np.log(c_star) - nu_i * (
            ell_star ** (1.0 + par.epsilon)
        ) / (1.0 + par.epsilon)

        # return a clean namespace with (ell, c, U)
        opt = SimpleNamespace(ell=ell_star, c=c_star, U=U_star)
        return opt
