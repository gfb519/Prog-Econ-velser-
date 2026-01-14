from types import SimpleNamespace

import numpy as np

from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

class WorkerClass:

    def __init__(self,par=None):

        # a. setup
        self.setup_worker()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup_worker(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preferences
        par.nu = 0.015 # weight on labor disutility
        par.epsilon = 1.0 # curvature of labor disutility
        
        # b. productivity and wages
        par.w = 1.0 # wage rate
        par.ps = np.linspace(0.5,3.0,100) # productivities
        par.ell_max = 16.0 # max labor supply
        
        # c. taxes
        par.tau = 0.50 # proportional tax rate
        par.zeta = 0.10 # lump-sum tax
        par.kappa = np.nan # income threshold for top tax
        par.omega = 0.20 # top rate rate
          
    def utility(self,c,ell):

        par = self.par
        u= np.log(c) - par.nu*(ell**(1+par.epsilon))/(1+par.epsilon)
        return u
    
    def tax(self,pre_tax_income):

        par = self.par
        tax = par.tau*pre_tax_income + par.zeta
        return tax
    
    def income(self,p,ell):

        par = self.par
        income = par.w*p*ell
        return income

    def post_tax_income(self,p,ell):

        pre_tax_income = self.income(p,ell)
        tax = self.tax(pre_tax_income)
        return pre_tax_income - tax
    
    def max_post_tax_income(self,p):

        par = self.par
        return self.post_tax_income(p,par.ell_max)

    def value_of_choice(self,p,ell):

        par = self.par

        c = self.post_tax_income(p,ell)
        U = self.utility(c,ell)

        return U
    
    def get_min_ell(self,p):
    
        par = self.par

        min_ell = par.zeta/(par.w*p*(1-par.tau))

        return np.fmax(min_ell,0.0) + 1e-8
    
    def optimal_choice(self,p):

        par = self.par
        opt = SimpleNamespace()

        # a. objective function
        def obj(ell):
            return -self.value_of_choice(p,ell)

        # b. bounds and minimization
        bounds = (self.get_min_ell(p), par.ell_max)
        res = minimize_scalar(obj, bounds=bounds, method='bounded')

        # c. results
        opt.ell = res.x
        opt.U = -res.fun
        opt.c = self.post_tax_income(p,opt.ell)

        return opt
    
    def FOC(self,p,ell):

        par = self.par

        c = self.post_tax_income(p,ell)
        FOC= par.w*p*(1-par.tau)/c - par.nu*(ell**par.epsilon)

        return FOC
    
    def optimal_choice_FOC(self,p):

        par = self.par
        opt = SimpleNamespace()
        
        ell_min = self.get_min_ell(p)
        ell_max = par.ell_max

        try:
            res = root_scalar(lambda ell: self.FOC(p,ell),
                              bracket=[ell_min,ell_max],
                              method='bisect')

            ell_star = res.root
        
        except ValueError:
            # Searching for corner solution
            ell_star = ell_min

        opt.ell = ell_star
        opt.c = self.post_tax_income(p,ell_star)
        opt.U = self.utility(opt.c, ell_star)
        

        return opt