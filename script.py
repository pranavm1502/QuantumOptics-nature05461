#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: script.py
# Author:   Lyu Ming <CareF.Lm@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
N = 20

# All the units are MHz
wr = 5.7E3 * 2 * np.pi    # resonator frequency
wq = 6.95E3 * 2 * np.pi   # qubit frequency
chi = -17 * np.pi         # parameter in the dispersive hamiltonian

kappa = 0.25 * 2 * np.pi  # damping rate for cavity
gamma = 1.8 * 2 * np.pi   # relaxation rate for qubit (1/T1)
gammap = 1.0 * 2 * np.pi  # dephasing rate for qubit (1/T2)

Delta = np.abs(wr - wq)    # detuning
g = np.sqrt(-Delta * chi)  # coupling strength that is consistent with chi

# cavity operators
a = tensor(destroy(N), qeye(2))
nc = a.dag() * a
xc = a + a.dag()

# atomic operators
sp = tensor(qeye(N), destroy(2)) # simga^+
sm = sp.dag()                    # sigma^-
sz = tensor(qeye(N), sigmaz()) 
sx = tensor(qeye(N), sigmax())
nq = sm.dag() * sm
# xq = sm + sm.dag() = sx

I = tensor(qeye(N), qeye(2))

def c_ops(nth = 0.1): 
    return [np.sqrt(kappa*(1+nth)) * a, 
    np.sqrt(kappa*nth) * a.dag(), # damping for cavity
    np.sqrt(gamma) * sm,     # relaxation for qubit
    np.sqrt(gammap) * sz]    # dephasing for qubit

# dispersive hamiltonian
H0 = wr * (a.dag() * a + I/2.0) + (wq / 2.0) * sz + \
    chi * (a.dag() * a + I/2) * sz
delta = 2.0 * 2 * np.pi 
wrf0 = wr - chi + delta    # RF freq (pumping) .... 
# rotating frame diff
def Hr(ws, wrf = wrf0): 
    return wrf * a.dag() * a + (ws / 2.0) * sz

def cavity_state(erf, wrf=wrf0): 
    """
    Return the steady state of the system
    without any sweaping signals /or qubit at ground state. 
    erf: The strength of RF signal (driving for cavity)
    wrf: The frequency of RF signal
    """
    Hrf = erf * xc
    H = H0 - Hr(0.0, wrf) + Hrf 
    rhoss = steadystate(H, c_ops()) 
    return rhoss

def with_sweaping(erf, ws, esp=0.1):
    """
    Return the steady state of the system with sweaping signals.
    erf: The strength of RF signal (driving for cavity)
    ws: The frequency of the sweaping signal
    esp: The strength of the sweaping signal (in our case, it's small
            so that the qubit is approximately at ground state)
    """
    Hrf = erf * xc
    Hsp = esp * (g/Delta) * sx
    rhoss = steadystate(H0 - Hr(ws) + Hrf + Hsp, c_ops()) 
    return rhoss

if __name__ == '__main__':
    wslist = 2 * np.pi * np.linspace(6.75E3, 7.0E3, 500)
    erf = 0.5
    rho_non_sw = cavity_state(erf)
    Ibar = (expect(a.dag() + a, rho_non_sw))**2
    nbar = (expect(a.dag() * a, rho_non_sw))
    print("Ibar=%f, nbar=%f"%(Ibar,nbar))
    trans = np.array([
        expect(a.dag() + a, with_sweaping(erf, w, 0.1)) 
        for w in wslist]) 
    plt.plot(wslist/(2*np.pi),  Ibar - trans**2 )
    plt.gca().invert_xaxis()
    plt.show()