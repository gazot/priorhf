import numpy as np
import pyhf 

from collections import namedtuple
from .utils import *

# Julia packages
from juliacall import Main as jl
jl.seval("using Distributions")
jl.seval('using ValueShapes')


# Makro to get keys of a dict as parameter string for namedtuple 
key_str = lambda d: ' '.join(list(d.keys()))


def make_prior(ws: str, typ='jl'):
    """
    Create priors from a pyhf workspace 'ws' as dict {'name' : density}
    @param  ws  'path/to/HF/workspace.json'
    @param  typ ['jl', 'scipy', 'txt'] 
                jl:     returns a Tuple (jl.NamedTupleDist, prior_specs)
                scipy:  dict {param_name: scipy.stats density}
                txt:    dict {param_name: txt }
    """
    
    model = pyhf_model(ws)

    # parameter map and modifiers
    par_map = model.config.par_map.values()
    modifier_dict = dict(model.config.modifiers)

    # initialize param vector 
    param_len = len(model.config.suggested_init())
    param = [None] * param_len 

    # create prior distributions for all parameters 
    for par in par_map:
        # paramset holds all parameter properties
        pset = par['paramset']
        pslice = par['slice']
        
        mod = modifier_dict[pset.name]
        if typ == 'jl':
            param[pslice] = create_pdf(mod, pset) 
        elif typ == 'scipy':
            param[pslice] = create_pdf_scipy(mod, pset) 
        elif typ == 'txt':
            param[pslice] = create_pdf_txt(mod, pset) 
        else:
            raise TypeError("Typ '{}' not supported.".format(typ))

    # construct names (replace ' ' with '_')
    names = list()
    for par in par_map:
        nparam = par['paramset'].n_parameters
        name = par['paramset'].name
        # replace chars for namedtuple representation 
        name = name.replace(' ', '_')
        name = name.replace('-', '_')
        name = name.replace('+', '_')
        names += [name] if nparam == 1 else [name + str(i) for i in range(nparam)]

    prior_specs = dict(zip(names, param)) 
    
    if typ == 'jl':
        # create namedtuple to preserve the order (py->jl issue)
        p = namedtuple('Prior', key_str(prior_specs))(**prior_specs)
        # julia typ for prior as flat vector 
        return jl.unshaped(jl.NamedTupleDist(p)), prior_specs 
    else: 
        return prior_specs


def create_pdf(modifier: str, pset):
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
            return [jl.Gamma(float(a), float(theta)) for (a, theta) in gamma_pdf_iter]

        elif modifier == 'histosys':    # 1 parameter
            # Standard Normal distribution centered at 0
            mu, sig = gauss_param(auxdata, 1)
            return [jl.Normal(float(mu), float(sig))]
        
        elif modifier == 'normsys':     # 1 parameter
            # Normal distribution centered at 0
            mu, sig = gauss_param(auxdata, 1)
            return [jl.Normal(float(mu), float(sig))]

        elif modifier == 'staterror':   # n parameter
            sigmas = pset.sigmas
            # Normal distribution centered at 1
            gauss_pdf_iter = iter(zip(*gauss_param(auxdata, sigmas)))
            return [jl.Normal(float(mu), float(sig)) for (mu, sig) in gauss_pdf_iter]

        elif modifier == 'lumi':        # 1 parameter
            # Normal distribution centered at 1
            mu, sig = gauss_param(auxdata, pset.sigmas)
            return [jl.Normal(float(mu), float(sig))]
        else: 
            raise TypeError("Unexpected modifier '{}'".format(modifier))

    # unconstrained parameter (normfactor, shapefactor)
    else:
        if modifier == 'normfactor':
            return [jl.Uniform(0, 5)]
        elif modifier == 'shapefactor':
            raise NotImplementedError
        else: 
            raise TypeError("Unexpected modifier '{}'".format(modifier))
