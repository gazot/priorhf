import os 
import json 
import pyhf 
import numpy as np
# scipy distributions
from scipy.stats import gamma, norm, uniform 


def pyhf_model(ws: str) -> pyhf.Model:
    """Wrapper to acces the pyhf_model from 'ws' path"""
    assert os.path.exists(ws), "No workspace '{}'".format(ws)

    with open(ws) as serialized:
        spec = json.load(serialized)
    return pyhf.Workspace(spec).model()


def create_pdf_txt(modifier: str, pset):
    """
    Create conjuagate prior distributions from auxdata.

    Note: 'shapefactor' modifiers are not implemented yet.
    """
    
    # constrained parameter (shapesys, histosys, normsys, staterror, lumi)
    if pset.constrained: 
        
        # constrained params have auxdata
        auxdata = pset.auxdata
        
        if modifier == 'shapesys':      # n parameter
            # Gamma(a, scale) for all bins as iterator
            gamma_pdf_iter = iter(zip(*gamma_param(auxdata)))
            return ['Gamma({}, {})'.format(a, theta) for(a, theta) in gamma_pdf_iter]

        elif modifier == 'histosys':    # 1 parameter
            # Standard Normal distribution centered at 0
            mu, sig = gauss_param(auxdata, 1)
            return ['Normal({},{})'.format(mu, sig)]
        
        elif modifier == 'normsys':     # 1 parameter
            # Normal distribution centered at 0
            mu, sig = gauss_param(auxdata, 1)
            return ['Normal({},{})'.format(mu, sig)]
    
        elif modifier == 'staterror':   # n parameter
            sigmas = pset.sigmas
            # Normal distribution centered at 1
            gauss_pdf_iter = iter(zip(*gauss_param(auxdata, sigmas)))
            return ['Normal({},{})'.format(mu, sig) for (mu, sig) in gauss_pdf_iter]

        elif modifier == 'lumi':        # 1 parameter
            # Normal distribution centered at 1
            mu, sig = gauss_param(auxdata, pset.sigmas)
            return ['Normal({},{})'.format(mu, sig)]
        else: 
            raise TypeError("Unexpected modifier '{}'".format(modifier))

    # unconstrained parameter (normfactor, shapefactor)
    else:
        if modifier == 'normfactor':
            return ['Uniform(0, 5)']
        elif modifier == 'shapefactor':
            raise NotImplementedError
        else: 
            raise TypeError("Unexpected modifier '{}'".format(modifier))


def create_pdf_scipy(modifier: str, pset):
    """
    Create conjuagate prior distributions from auxdata.

    Note: 'shapefactor' modifiers are not implemented yet.
    """
    
    # constrained parameter (shapesys, histosys, normsys, staterror, lumi)
    if pset.constrained: 
        
        # constrained params have auxdata
        auxdata = pset.auxdata
        
        if modifier == 'shapesys':      # n parameter
            # Gamma(a, scale) for all bins as iterator
            gamma_pdf_iter = iter(zip(*gamma_param(auxdata)))
            return [gamma(a, scale=theta) for(a, theta) in gamma_pdf_iter]

        elif modifier == 'histosys':    # 1 parameter
            # Standard Normal distribution centered at 0
            mu, sig = gauss_param(auxdata, 1)
            return [norm(mu, sig)]
        
        elif modifier == 'normsys':     # 1 parameter
            # Normal distribution centered at 0
            mu, sig = gauss_param(auxdata, 1)
            return [norm(mu, sig)]
    
        elif modifier == 'staterror':   # n parameter
            sigmas = pset.sigmas
            # Normal distribution centered at 1
            gauss_pdf_iter = iter(zip(*gauss_param(auxdata, sigmas)))
            return [norm(mu, sig) for (mu, sig) in gauss_pdf_iter]

        elif modifier == 'lumi':        # 1 parameter
            # Normal distribution centered at 1
            mu, sig = gauss_param(auxdata, pset.sigmas)
            return [norm(mu, sig)]
        else: 
            raise TypeError("Unexpected modifier '{}'".format(modifier))

    # unconstrained parameter (normfactor, shapefactor)
    else:
        if modifier == 'normfactor':
            return [uniform(0,5)]
        elif modifier == 'shapefactor':
            raise NotImplementedError
        else: 
            raise TypeError("Unexpected modifier '{}'".format(modifier))


def list_params(ws: str):
    """List parameter (name, modifier) for Debugging."""
    model = pyhf_model(ws) 

    # parameter map and modifiers
    par_map = model.config.par_map
    modifier_dict = dict(model.config.modifiers)

    nparam = len(model.config.suggested_init())

    for name, par in par_map.items():
        num = par['paramset'].n_parameters
        for i in range(num):
            print('{:20s} {:12s}'.format(name, modifier_dict[name]))


# ---------------- conjugate prior update ---------------------------


# Initial prior parameters

GAMMA_PRIOR_ALPHA = 1
GAUSS_PRIOR_VAR = 100


def gauss_param(mu_s, sigma_s=1):
    """
    Compute conjugate prior for Normal distributed priors.
    @param  mu_s    mean of the sample
    @param  var_s   sample variance 
    @return (mu, std) 
    """
    # init Prior is a Gaussina with variance GAUSS_SIGMA**2 centered at mu_s
    mu_0 = np.asarray(mu_s)
    var_0 = GAUSS_PRIOR_VAR
    var_s = np.array(sigma_s)**2 

    # parameter update
    var_new = 1 / (1 / var_0 + 1 / var_s)
    mu_new = var_new * (mu_0/var_0 + 1*np.asarray(mu_s)/var_s)
    return  mu_new, np.sqrt(var_new) 


def gamma_param(mu_s):
    """
    Compute conjugate prior for Gamma distributed priors.
    @param  mu_s  up-scaled Poisson mean from auxdata
    @return (a, scale)    equal to [alpha, 1/beta]
    """
    # initial Prior with alpha = 1, beta = alpha/mu
    alpha_0 = GAMMA_PRIOR_ALPHA
    beta_0 = alpha_0 / np.asarray(mu_s) 

    # parameter update
    alpha_new = np.asarray(mu_s) + 1 
    beta_new = (beta_0 + 1) 

    # re-scale density to mean '1'
    beta_new *= mu_s

    return alpha_new, 1/beta_new
