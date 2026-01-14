from Worker import WorkerClass
import numpy as np
from scipy.optimize import minimize

class GovernmentClass(WorkerClass):

    def __init__(self,par=None):

        # a. defaul setup
        self.setup_worker()
        self.setup_government()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

        # c. random number generator
        self.rng = np.random.default_rng(12345)

    def setup_government(self):

        par = self.par

        # a. workers
        par.N = 100  # number of workers
        par.sigma_p = 0.3  # std dev of productivity

        # b. pulic good
        par.chi = 50.0 # weight on public good in SWF
        par.eta = 0.1 # curvature of public good in SWF

    def draw_productivities(self):

        par = self.par
        sol = self.sol
        mu = -0.5 * par.sigma_p**2
        sigma = par.sigma_p

        sol.p = self.rng.lognormal(mean=mu, sigma=sigma, size=par.N) 

        par.p_min = np.min(sol.p)

    def solve_workers(self):

        par = self.par
        sol = self.sol
        if not hasattr(sol,'p'):
            self.draw_productivities()

        N = par.N
        sol.ell = np.empty(N)
        sol.c = np.empty(N)
        sol.U = np.empty(N)

        for i in range(N):
            p_i = sol.p[i]
            opt = self.optimal_choice_FOC(p_i)

            sol.ell[i] = opt.ell
            sol.c[i] = opt.c
            sol.U[i] = opt.U


    def tax_revenue(self):

        par = self.par
        sol = self.sol

        tax_revenue = 0.0

        tax_revenue += par.N * par.zeta

        tax_revenue += np.sum(par.tau * sol.p * par.w * sol.ell)

        return tax_revenue
    
    def SWF(self):

        par = self.par
        sol = self.sol

        G =  self.tax_revenue()
        if G < 0:
            SWF = np.nan
        else:
            G_value = par.chi * (G ** par.eta)

            SWF = G_value + np.sum(sol.U)

        return SWF
    
    def optimal_taxes(self,tau,zeta):

        par = self.par
        sol = self.sol
        
        # a. objective function
        def obj(x):
            par.tau = x[0]
            par.zeta = x[1]


            self.solve_workers()
            SWF_val = self.SWF()


            if not np.isfinite(SWF_val):
                return 1e10

            return -SWF_val
        
        x0 = np.array([tau,zeta])

        bounds = ((0.01, 0.80), (-0.10, 0.10))

        # b. optimization
        res = minimize(obj, x0, bounds=bounds)

        # c. results
        par.tau = res.x[0]
        par.zeta = res.x[1]
        self.solve_workers()
        sol.SWF = self.SWF()

        return res
    
