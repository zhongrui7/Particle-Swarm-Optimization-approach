#!/usr/bin/env python3
"""
Parameter extraction of solar cells using Particle Swarm Optimization (PSO)
+ Matplotlib plotting of experimental and fitted I-V curves
2025-11-10
"""

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from typing import List, Tuple

# ====================== Matplotlib Style ======================
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 7),
    'lines.linewidth': 2.0,
})

# ====================== Constants ======================
w = 0.2        # inertia weight
c1 = 0.2       # cognitive parameter
c2 = 0.6       # social parameter
maxgen = 2028  # number of generations
sizepop = 1024 # population size
dim = 5        # dimension
popmin = 0.0   # minimum bound

k = 1.38e-23   # Boltzmann constant [J/K]
q = 1.6e-19    # electron charge [C]
T = 300.0      # default temperature [K]
Vt = 0.025875  # kT/q at 300K
A = 1.0        # active area [cm²]
n1 = 1.0
n2 = 2.0

# ====================== Global Variables ======================
V0 = np.zeros(300)
I0 = np.zeros(300)
DSize = 0
model = '1'
CurrentUnit = '1'

Jph0 = Rs0 = Rp0 = 0.0
Js0 = n0 = 1.0
Js10 = Js20 = 0.0

Jsc = Voc = FF = Pmax = Vm = Jm = eff = 0.0

# PSO arrays
pop = np.zeros((sizepop, dim))
V = np.zeros((sizepop, dim))
fitness = np.zeros(sizepop)
pbest = np.zeros((sizepop, dim))
gbest = np.zeros(dim)
fitnesspbest = np.zeros(sizepop)
fitnessgbest = np.inf
genbest = np.zeros((maxgen, dim))
result = np.zeros(maxgen)


# ====================== Diode Models ======================
def IL1(Vm: float, Im: float, Jph: float, Js: float, Rs: float, Rp: float, n: float) -> float:
    return Jph - Js * (np.exp((Vm + Im * Rs) / (n * Vt)) - 1) - (Vm + Im * Rs) / Rp

def IL2(Vm: float, Im: float, Jph: float, Js1: float, Js2: float, Rs: float, Rp: float) -> float:
    return (Jph 
            - Js1 * (np.exp((Vm + Im * Rs) / (n1 * Vt)) - 1)
            - Js2 * (np.exp((Vm + Im * Rs) / (n2 * Vt)) - 1)
            - (Vm + Im * Rs) / Rp)


# ====================== Fitness Functions ======================
def func1(arr: np.ndarray) -> float:
    Jph, Js, Rs, Rp, n = arr
    err = 0.0
    for i in range(DSize):
        diff = IL1(V0[i], I0[i], Jph, Js, Rs, Rp, n) - I0[i]
        err += diff * diff
    return err / DSize

def func2(arr: np.ndarray) -> float:
    Jph, Js1, Js2, Rs, Rp = arr
    err = 0.0
    for i in range(DSize):
        diff = IL2(V0[i], I0[i], Jph, Js1, Js2, Rs, Rp) - I0[i]
        err += diff * diff
    return err / DSize


# ====================== PSO Initialization ======================
def rng_uniform() -> float:
    return np.random.random() - 0.5

def pop_init():
    global pop, V, fitness
    np.random.seed(int(time.time() * 1e6) % 2**32)
    
    for i in range(sizepop):
        if model == '1':
            pop[i][0] = Jph0 * (1 + rng_uniform())
            pop[i][1] = Js0 * (1 + rng_uniform())
            pop[i][2] = Rs0 * (1 + rng_uniform())
            pop[i][3] = Rp0 * (1 + rng_uniform())
            pop[i][4] = n0 * (1 + rng_uniform())
            fitness[i] = func1(pop[i])
        else:
            pop[i][0] = Jph0 * (1 + rng_uniform())
            pop[i][1] = Js10 * (1 + rng_uniform())
            pop[i][2] = Js20 * (1 + rng_uniform())
            pop[i][3] = Rs0 * (1 + rng_uniform())
            pop[i][4] = Rp0 * (1 + rng_uniform())
            fitness[i] = func2(pop[i])
        
        V[i] = pop[i].copy()


# ====================== Helpers ======================
def find_min(fit: np.ndarray) -> Tuple[int, float]:
    idx = np.argmin(fit)
    return idx, fit[idx]

def sort_V_I(V_arr: np.ndarray, I_arr: np.ndarray):
    sorted_idx = np.argsort(V_arr)
    return V_arr[sorted_idx], I_arr[sorted_idx]


# ====================== PV Performance ======================
def PV_perf():
    global Jsc, Voc, Pmax, Vm, Jm, FF, eff, Jph0, Rs0, Rp0, Js0, Js10, Js20

    V_tmp = abs(V0[0])
    I_tmp = abs(I0[0])
    p_idx, s_idx, m_idx = 0, 0, 0

    V_sorted, I_sorted = sort_V_I(V0[:DSize].copy(), I0[:DSize].copy())
    V0[:DSize], I0[:DSize] = V_sorted, I_sorted

    if (V0[1] * I0[1] > 0) and (I0[0] + I0[1] < 0):
        I0[:DSize] *= -1

    Pmax = 0.0
    for i in range(DSize):
        if abs(V0[i]) < V_tmp:
            V_tmp = abs(V0[i])
            p_idx = i
        if abs(I0[i]) < I_tmp:
            I_tmp = abs(I0[i])
            s_idx = i
        power = V0[i] * I0[i]
        if power > Pmax and V0[i] > 0 and I0[i] > 0:
            Pmax = power
            m_idx = i

    Vm = V0[m_idx]
    Jm = I0[m_idx]

    # Interpolate Jsc and Voc
    def interp_y(x0, x1, x2, y0, y1, y2, x_target):
        if x1 == x_target: return y1
        return y0 + (y2 - y0) * (x_target - x0) / (x2 - x0)

    Jsc = abs(I0[p_idx]) if abs(V0[p_idx]) < 1e-6 else interp_y(V0[p_idx-1], V0[p_idx], V0[p_idx+1],
                                                               I0[p_idx-1], I0[p_idx], I0[p_idx+1], 0.0)
    Voc = abs(V0[s_idx]) if abs(I0[s_idx]) < 1e-6 else interp_y(I0[s_idx-1], I0[s_idx], I0[s_idx+1],
                                                               V0[s_idx-1], V0[s_idx], V0[s_idx+1], 0.0)

    print(f"\t Voc ≈ {Voc:.4f} V, Jsc ≈ {Jsc*1000:.3f} mA/cm²")

    if Jsc > 1e-9 and Voc > 1e-9:
        FF = Pmax / (Voc * Jsc)
        eff = Pmax * 1000
        print(f"\t Voc={Voc:7.3f} V, Jsc={Jsc*1000:7.3f} mA/cm², FF={FF*100:5.1f}%, η={eff:6.2f}%")
        print(f"\t Pmax={Pmax*1000:6.2f} mW/cm² at Vm={Vm:.3f} V, Jm={Jm*1000:.2f} mA/cm²")

        def slope_avg(idx):
            if idx < 2 or idx >= DSize - 2: return 0
            dv1 = V0[idx+1] - V0[idx-2]; di1 = I0[idx+1] - I0[idx-2]
            dv2 = V0[idx+2] - V0[idx-1]; di2 = I0[idx+2] - I0[idx-1]
            return (abs(dv1/di1) + abs(dv2/di2)) / 2 if di1 != 0 and di2 != 0 else 0

        Rs0 = slope_avg(s_idx)
        Rp0 = slope_avg(p_idx)
        Jph0 = Jsc

        if model == '1':
            n0 = 1.0
            e1 = np.exp(Voc / (n0 * Vt))
            e2 = np.exp(Rs0 * Jsc / (n0 * Vt))
            Js0 = (Jsc * (Rp0 - Rs0) - Voc) / (Rp0 * (e1 - e2))
            print(f"Initial: Jph={Jph0*1000:.2f} mA/cm², Js={Js0*1000:.2e}, Rs={Rs0:.2f}, Rp={Rp0:.1f}, n={n0:.1f}")
        else:
            e1 = np.exp(Voc / (n1 * Vt))
            e2 = np.exp(Rs0 * Jsc / (n1 * Vt))
            Js10 = (Jsc * (Rp0 - Rs0) - Voc) / (Rp0 * (e1 - e2))
            Js20 = Jsc / (np.exp(Voc * q / (k * T)) - 1)
            print(f"Initial: Jph={Jph0*1000:.2f} mA/cm², Js1={Js10*1000:.2e}, Js2={Js20*1000:.2e}, Rs={Rs0:.2f}, Rp={Rp0:.1f}")
    else:
        print("\n\t DARK IV CURVE DETECTED!")
        Jph0 = 0.0
        mid = DSize // 2
        Js0 = -0.5 * (I0[mid] / (np.exp(V0[mid]/Vt) - 1) + I0[-1] / (np.exp(V0[-1]/Vt) - 1))
        Rs0 = Rp0 = 1e3
        Js10 = Js20 = Js0


# ====================== PSO Main Loop ======================
def PSO_func():
    global fitnessgbest, gbest, pbest, fitnesspbest

    idx, min_fit = find_min(fitness)
    gbest[:] = pop[idx].copy()
    pbest[:] = pop.copy()
    fitnesspbest[:] = fitness.copy()
    fitnessgbest = min_fit

    for gen in range(maxgen):
        for j in range(sizepop):
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            V[j] = w * V[j] + c1 * r1 * (pbest[j] - pop[j]) + c2 * r2 * (gbest - pop[j])
            pop[j] = np.maximum(pop[j] + V[j], popmin)
            pop[j][0] = 0.0 if Jsc == 0 else pop[j][0]

            if model == '1':
                while pop[j][4] <= 0 or pop[j][4] > 3:
                    pop[j][4] = n0 * (1 + rng_uniform() / 50)

            fitness[j] = func1(pop[j]) if model == '1' else func2(pop[j])

        improved = fitness < fitnesspbest
        pbest[improved] = pop[improved]
        fitnesspbest[improved] = fitness[improved]

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < fitnessgbest:
            gbest[:] = pop[best_idx].copy()
            fitnessgbest = fitness[best_idx]

        genbest[gen] = gbest.copy()
        result[gen] = fitnessgbest

        if gen % 64 == 0:
            print("*", end="", flush=True)
    print()


# ====================== Plotting Function ======================
def plot_iv_curve(V_exp, I_exp, best_params, filename):
    V_fit = np.linspace(min(V_exp), max(V_exp), 500)
    I_fit = np.zeros_like(V_fit)

    if model == '1':
        Jph, Js, Rs, Rp, n = best_params
        for i, v in enumerate(V_fit):
            I_fit[i] = IL1(v, 0, Jph, Js, Rs, Rp, n)  # Solve I from V
    else:
        Jph, Js1, Js2, Rs, Rp = best_params
        for i, v in enumerate(V_fit):
            I_fit[i] = IL2(v, 0, Jph, Js1, Js2, Rs, Rp)

    scale = 1000 if CurrentUnit == '2' else 1
    unit = "mA/cm²" if CurrentUnit == '2' else "A/cm²"

    fig, ax = plt.subplots()
    ax.plot(V_exp, I_exp * scale, 'o', label='Experimental', markersize=5, alpha=0.8)
    ax.plot(V_fit, I_fit * scale, '-', label='Fitted (PSO)', linewidth=2.5)

    # Key points
    ax.plot(Voc, 0, 's', color='red', label=f'Voc = {Voc:.3f} V')
    ax.plot(0, Jsc * scale, 's', color='green', label=f'Jsc = {Jsc*scale:.2f} {unit}')
    ax.plot(Vm, Jm * scale, 's', color='purple', label=f'MPP = {Pmax*1000:.1f} mW/cm²')

    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel(f'Current Density [{unit}]')
    ax.set_title(f'{"Single" if model=="1" else "Double"}-Diode Model Fit (PSO)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_file = filename.rsplit('.', 1)[0] + f"_fit{model}{'S' if model=='1' else 'D'}.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\t Plot saved: {plot_file}")


# ====================== Main ======================
def main():
    global DSize, V0, I0, model, CurrentUnit, A, T, Vt

    if len(sys.argv) != 2:
        print("Usage: python pso_solar_fit_plot.py IV-file")
        return

    print("PSO Solar Cell Parameter Extraction + Plotting")
    print("Input: 2-column (V I) ASCII file")

    while CurrentUnit not in ['1', '2']:
        CurrentUnit = input("Current unit (1: A/cm², 2: mA/cm²): ").strip()
    print(f"→ Using { 'A/cm²' if CurrentUnit=='1' else 'mA/cm²' }")

    inp = input("Area [cm²], Temp [K] (default: 1.0, 300): ") or "1.0, 300"
    A, T = map(float, [x.strip() for x in inp.split(',')])
    Vt = 0.025875 * T / 300

    filename = sys.argv[1]
    data = np.loadtxt(filename)
    DSize = len(data)
    V0[:DSize], I0[:DSize] = data[:, 0], data[:, 1] / A
    if CurrentUnit == '2':
        I0[:DSize] /= 1000

    while model not in ['1', '2']:
        model = input("Model (1: Single-diode, 2: Double-diode): ").strip()

    print(f"→ {'Single' if model=='1' else 'Double'}-diode model selected")

    PV_perf()
    pop_init()
    start_time = time.time()
    k = 0

    while True:
        print(f"\nRunning PSO (Run #{k+1})...", end="")
        PSO_func()
        k += 1

        best_gen = np.argmin(result)
        best_fit = result[best_gen]
        best_params = genbest[best_gen]

        print(f"\nBest fit (Gen {best_gen}): MSE = {best_fit:.3e}")

        if model == '1':
            Jph, Js, Rs, Rp, n = best_params
            print(f"Jph = {Jph*1000:6.2f} mA/cm², Js = {Js*1000:.2e}, Rs = {Rs:6.2f}, Rp = {Rp:7.1f}, n = {n:.2f}")
        else:
            Jph, Js1, Js2, Rs, Rp = best_params
            print(f"Jph = {Jph*1000:6.2f} mA/cm², Js1 = {Js1*1000:.2e}, Js2 = {Js2*1000:.2e}, Rs = {Rs:6.2f}, Rp = {Rp:7.1f}")

        # Save data
        out_name = filename.rsplit('.', 1)[0] + f"_fit{model}{'S' if model=='1' else 'D'}.dat"
        np.savetxt(out_name, np.column_stack((V0[:DSize], I0[:DSize]*scale, 
                    [IL1(V0[i], I0[i], *best_params) if model=='1' else IL2(V0[i], I0[i], *best_params) for i in range(DSize)] * scale)),
                   header=f"Bias[V] Jexp[{unit}] Jfit[{unit}]", fmt="%.6f")
        print(f"\t Data saved: {out_name}")

        # Plot
        plot_iv_curve(V0[:DSize], I0[:DSize], best_params, filename)

        cont = input("\nRun PSO again? [y/n]: ").strip().lower()
        if cont not in ['y', 'yes']:
            break

    print(f"\nTotal time: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()
