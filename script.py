#!/usr/bin/env python2
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

def cavity_state(erf, wrf=wrf0, nth=0.1): 
    """
    Return the steady state of the system
    without any sweaping signals /or qubit at ground state. 
    erf: The strength of RF signal (driving for cavity)
    wrf: The frequency of RF signal
    """
    Hrf = erf * xc
    H = H0 - Hr(0.0, wrf) + Hrf 
    rhoss = steadystate(H, c_ops(nth)) 
    return rhoss

def with_sweaping(erf, ws, nth=0.1, esp=0.1):
    """
    Return the steady state of the system with sweaping signals.
    erf: The strength of RF signal (driving for cavity)
    ws: The frequency of the sweaping signal
    esp: The strength of the sweaping signal (in our case, it's small
            so that the qubit is approximately at ground state)
    """
    Hrf = erf * xc
    Hsp = esp * (g/Delta) * sx
    rhoss = steadystate(H0 - Hr(ws) + Hrf + Hsp, c_ops(nth)) 
    return rhoss

def draw_sweaping(axis, wslist, erf, nth=0.1, obs = a.dag()+a):
    rho_non_sw = cavity_state(erf, nth=nth)
    Vbar = (expect(obs, rho_non_sw))
    # nbar = (expect(a.dag() * a, rho_non_sw))
    # print("Ibar=%f, nbar=%f"%(Ibar,nbar))
    trans = np.array([
        expect(obs, with_sweaping(erf, w, nth)) 
        for w in wslist]) 
    axis.plot(wslist/(2*np.pi),  Vbar - trans )
    # axis.get_yaxis().set_visible(False)
    # axis.invert_xaxis()

def coherence_drive():
    wslist = 2 * np.pi * np.linspace(6.75E3, 7.0E3, 500)
    nbars = np.array([0.02, 1.4, 2.3, 3.2, 4.1, 4.6])
    erfs = np.sqrt( nbars*(delta**2 + kappa**2 / 4))
    f, axs = plt.subplots(len(nbars), sharex=True)
    N = len(nbars)
    for n in range(N): 
        draw_sweaping(axs[N-1-n], wslist, erfs[n])
        axs[N-1-n].text(0.7, 0.9, r"$\bar n=%3f$"%nbars[n])
        axs[N-1-n].get_yaxis().set_visible(False)
    axs[0].invert_xaxis()
    plt.show()

def thermal():
    wslist = 2 * np.pi * np.linspace(6.75E3, 7.0E3, 500)
    f, axs = plt.subplots(2, sharex=True)
    draw_sweaping(axs[0], wslist, np.sqrt( 2.9*(delta**2 + kappa**2 / 4)))
    axs[0].get_yaxis().set_visible(False)
    draw_sweaping(axs[1], wslist, 0.01, nth=3)
    axs[1].get_yaxis().set_visible(False)
    axs[0].invert_xaxis()
    plt.show()

def wignar_diff():
    wslist = 2 * np.pi * np.linspace(6.75E3, 7.0E3, 500)
    erf = 0.1
    # erf=20
    wrf = np.array([6942])*(2*np.pi)
    # wrf = np.array([6898.8])*(2*np.pi)
    no_sw = cavity_state(erf)
    rhoss_reso = with_sweaping(erf, wrf[0])
    # draw_sweaping(plt.gca(), wslist, erf)
    # plt.plot(wrf/(2*np.pi), [expect(a + a.dag(), no_sw-rhoss_reso)], 'r.')
    # plt.gca().invert_xaxis()
    # plt.show()
    xmax=4
    # xmax=7
    xvec = np.linspace(-xmax,xmax,200)
    wswp = wigner((rhoss_reso- no_sw).ptrace(0), xvec, xvec)
    zmax = 1.8E-9
    # zmax = 1E-7
    cs = plt.contourf(xvec, xvec, wswp, 100
                      ,cmap='seismic', vmin=-zmax, vmax=zmax)
    # plt.contour(cs, wswp, levels=[0], colors=('k'))
    plt.colorbar(cs, label=r"$\Delta P$")
    plt.xlabel(r"$\langle a + a^\dag\rangle$", fontsize=12)
    plt.ylabel(r"$i \langle a - a^\dag\rangle$", fontsize=12)
    plt.show()


if __name__ == '__main__':
    # erf=30
    # rhoc = cavity_state(erf, wrf0).ptrace(0)
    # plt.bar(np.arange(0, N), rhoc.diag())
    # plt.ylabel(r"$\langle n|\rho |n \rangle$", fontsize=12)
    # plt.xlabel(r"$n$", fontsize=12)
    # plt.show()

    # wrflist = wrf0 + np.linspace(-80, 80, 500)
    # nbarlist = [expect(a.dag()*a, cavity_state(erf, wrf)) for wrf in wrflist]
    # plt.plot(wrflist/(2*np.pi), erf**2/(kappa**2/4 + (delta + wrflist-wrf0)**2)+0.1, '--')
    # plt.plot(wrflist/(2*np.pi), nbarlist)
    # plt.ylim((-0.3, 13))
    # plt.arrow(wrf0/(2*np.pi), 3.8, 0, 0.8, ec='r', fc= 'r', head_width=0.4, head_length=0.5)
    # plt.xlabel(r"$\omega_{\mathrm{rf}}/(2\pi)$ (MHz)")
    # plt.ylabel(r"$\langle a^\dagger a\rangle$")
    # plt.show()

    # wslist = 2 * np.pi * np.linspace(6.75E3, 7.0E3, 500)
    # draw_sweaping(plt.gca(), wslist, 20, obs=a.dag()*a)
    # # plt.gca().get_yaxis().set_visible(False)
    # plt.gca().invert_xaxis()
    # plt.xlabel(r"$\omega_s/(2\pi)$ (MHz)", fontsize=12)
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(-3,4))
    # plt.ylabel(r"$\bar n_s - \bar n$", fontsize=12)
    # plt.show()

    # wignar_diff()