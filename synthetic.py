"""Explicit synthetic generators; no synthetic curve represents a real sample."""
import numpy as np


def mean_field(*,noise=0.,seed=0,n_fields=160,n_temperatures=21):
    rng=np.random.default_rng(seed); curves=[]
    for T in np.linspace(95,105,n_temperatures):
        t=(T-100)/100
        m=np.linspace(max(np.sqrt(max(-t,0))+.01,.03),1,n_fields)
        h=t*m+m**3
        observed=m*np.exp(rng.normal(0,noise,len(m)))
        curves.append(dict(file=f'synthetic_{T:g}K',T=float(T),H=h,M=observed))
    return curves


def independent_scaling(beta=.23,gamma=2.2,*,noise=0.,seed=0,n_fields=160):
    rng=np.random.default_rng(seed); curves=[]; delta=1+gamma/beta
    for T in np.r_[np.linspace(95,99.5,10),np.linspace(100.5,105,10)]:
        t=(T-100)/100; h=np.geomspace(1e-6,.1,n_fields); x=h/abs(t)**(beta+gamma)
        q=1.3
        f=x/(1+x**q)**((delta-1)/(delta*q)) if t>0 else (1+x**q)**(1/(delta*q))
        m=abs(t)**beta*f*np.exp(rng.normal(0,noise,len(h)))
        curves.append(dict(file=f'synthetic_scaling_{T:g}K',T=float(T),H=h,M=m))
    return curves
