import numpy as np
import sys
import time
from uuid import uuid4

# Constants
c1 = 1.495  # Acceleration factor
c2 = 1.495
maxgen = 4096  # Number of iterations
sizepop = 1024  # Population size
dim = 5  # Dimension of the particle
popmin = 0  # Individual minimum value

# Physical constants
k = 1.38e-23
q = 1.6e-19
T = 300  # Temperature in Kelvin
Vt = 0.025875  # Vt = kT/q, for example, Vt(300K)=0.0259eV
A = 1.0  # Solar cell area in cm^2

# Initial parameters
Jph0 = 1.2e-3
Rs0 = 0.1
Rp0 = 2345
Js0 = 1.2e-5
n0 = 1
Js10 = 1.2e-7
Js20 = 1.2e-7
n1 = 1
n2 = 2

# Global arrays
pop = np.zeros((sizepop, dim))  # Population array
V = np.zeros((sizepop, dim))  # Population velocity array
fitness = np.zeros(sizepop)  # Fitness array
result = np.zeros(maxgen)  # Optimal value per iteration
pbest = np.zeros((sizepop, dim))  # Individual best positions
gbest = np.zeros(dim)  # Group best position
fitnesspbest = np.zeros(sizepop)  # Individual best fitness
fitnessgbest = 0.0  # Group best fitness
genbest = np.zeros((maxgen, dim))  # Best particles per generation
V0 = np.zeros(300)  # Experimental voltage data
I0 = np.zeros(300)  # Experimental current data
Jsc = 0
Voc = 0
FF = 0
Pmax = 0
Vm = 0
Jm = 0
eff = 0
DSize = 0
model = '1'
CurrentUnit = '1'

# Random number generator for -0.5 to 0.5
def rng_uniform():
    return np.random.uniform(-0.5, 0.5)

# Solar cell I-V function - single diode model
def IL1(V0, I0, Jph, Js, Rs, Rp, n):
    return Jph - Js * (np.exp((V0 + I0 * Rs) / (n * Vt)) - 1) - (V0 + I0 * Rs) / Rp

# Solar cell I-V function - double diode model
def IL2(V0, I0, Jph, Js1, Js2, Rs, Rp):
    return Jph - Js1 * (np.exp((V0 + I0 * Rs) / (n1 * Vt)) - 1) - Js2 * (np.exp((V0 + I0 * Rs) / (n2 * Vt)) - 1) - (V0 + I0 * Rs) / Rp

# Fitness function for single diode model
def func1(arr):
    Jph, Js, Rs, Rp, n = arr
    tot = 0.0
    for i in range(DSize):
        erf = IL1(V0[i], I0[i], Jph, Js, Rs, Rp, n) - I0[i]
        tot += erf * erf
    return tot / DSize

# Fitness function for double diode model
def func2(arr):
    Jph, Js1, Js2, Rs, Rp = arr
    tot = 0.0
    for i in range(DSize):
        erf = IL2(V0[i], I0[i], Jph, Js1, Js2, Rs, Rp) - I0[i]
        tot += erf * erf
    return tot / DSize

# Population initialization
def pop_init():
    global pop, V, fitness, Jph0, Js0, Rs0, Rp0, n0, Js10, Js20
    for i in range(sizepop):
        if model == '1':
            pop[i] = [Jph0 * (1 + rng_uniform()),
                      Js0 * (1 + rng_uniform()),
                      Rs0 * (1 + rng_uniform()),
                      Rp0 * (1 + rng_uniform()),
                      n0 * (1 + rng_uniform())]
            V[i] = pop[i] / 10
            fitness[i] = func1(pop[i])
        elif model == '2':
            pop[i] = [Jph0 * (1 + rng_uniform()),
                      Js10 * (1 + rng_uniform()),
                      Js20 * (1 + rng_uniform()),
                      Rs0 * (1 + rng_uniform()),
                      Rp0 * (1 + rng_uniform())]
            V[i] = pop[i] / 10
            fitness[i] = func2(pop[i])

# Find minimum fitness and index
def min_fit(fit):
    index = np.argmin(fit)
    return index, fit[index]

# PSO optimization
def PSO_func():
    global pbest, gbest, fitnesspbest, fitnessgbest, genbest, result
    pop_init()
    index, best_fitness = min_fit(fitness)
    gbest = pop[index].copy()
    pbest = pop.copy()
    fitnesspbest = fitness.copy()
    fitnessgbest = best_fitness

    for i in range(maxgen):
        for j in range(sizepop):
            for k in range(dim):
                rand1 = np.random.rand()
                rand2 = np.random.rand()
                V[j][k] = 0.9 * V[j][k] + c1 * rand1 * (pbest[j][k] - pop[j][k]) + c2 * rand2 * (gbest[k] - pop[j][k])
                pop[j][k] += V[j][k]
                if pop[j][k] < popmin:
                    pop[j][k] = popmin
            pop[j][0] = 0 if Jsc == 0 else pop[j][0]
            if model == '1':
                while pop[j][4] <= 0 or pop[j][4] > 3:
                    pop[j][4] = n0 * (1 + rng_uniform() / 50)
            fitness[j] = func1(pop[j]) if model == '1' else func2(pop[j])

        for j in range(sizepop):
            if fitness[j] < fitnesspbest[j]:
                pbest[j] = pop[j].copy()
                fitnesspbest[j] = fitness[j]
            if fitness[j] < fitnessgbest:
                gbest = pop[j].copy()
                fitnessgbest = fitness[j]

        genbest[i] = gbest.copy()
        result[i] = fitnessgbest
        if i % 40 == 0:
            print('*', end='', flush=True)

# Swap function for sorting
def swap(arrV, arrI, i, j):
    arrV[i], arrV[j] = arrV[j], arrV[i]
    arrI[i], arrI[j] = arrI[j], arrI[i]

# Selection sort for voltage and current arrays
def sort_v(arrV, arrI, n):
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if arrV[j] < arrV[min_idx]:
                min_idx = j
        swap(arrV, arrI, min_idx, i)

# Calculate solar cell performance parameters
def cell_perf():
    global Jsc, Voc, FF, Pmax, Vm, Jm, eff, Jph0, Js0, Rs0, Rp0, n0, Js10, Js20
    sort_v(V0, I0, DSize)
    Vtmp = abs(V0[0])
    Itmp = abs(I0[0])
    p = s = m = 0

    if V0[1] * I0[1] > 0 and I0[0] + I0[1] < 0:
        I0[:] = -I0[:]

    for i in range(DSize):
        if abs(V0[i]) < Vtmp:
            Vtmp = abs(V0[i])
            p = i
        if abs(I0[i]) < Itmp:
            Itmp = abs(I0[i])
            s = i
        if V0[i] * I0[i] > Pmax and V0[i] > 0 and I0[i] > 0:
            Pmax = V0[i] * I0[i]
            m = i

    Vm = V0[m]
    Jm = I0[m]
    Jsc = abs(I0[p]) if V0[p] == 0 else I0[p-1] - V0[p-1] * (I0[p+1] - I0[p-1]) / (V0[p+1] - V0[p-1])
    Voc = abs(V0[s]) if I0[s] == 0 else V0[s-1] - I0[s-1] * (V0[s+1] - V0[s-1]) / (I0[s+1] - I0[s-1])

    print(f"\t Voc=V0[{s}]={Voc:.3f}, Jsc=I0[{p}]={Jsc:.3f}")
    if Jsc != 0 and Voc != 0:
        FF = Pmax / (Voc * Jsc)
        eff = Pmax * 1000
        print(f"\t Voc={Voc:.3f}[V], Jsc={Jsc*1000:.4f}[mA/cm^2], FF={FF:.2f}, effi={eff:.3f}%")
        print(f"\t Pmax={Pmax*1000:.3f}[mW/cm^2], Vm={Vm:.3f}[V], Jm={Jm*1000:.3f}[mA/cm^2]")
        Rs0 = (abs((V0[s+1] - V0[s-2]) / (I0[s+1] - I0[s-2])) + abs((V0[s+2] - V0[s-1]) / (I0[s+2] - I0[s-1]))) / 2
        Rp0 = (abs((V0[p+1] - V0[p-2]) / (I0[p+1] - I0[p-2])) + abs((V0[p+2] - V0[p-1]) / (I0[p+2] - I0[p-1]))) / 2
        Jph0 = Jsc
        if model == '1':
            Js0 = (Jsc * (Rp0 - Rs0) - Voc) / (Rp0 * (np.exp(Voc / (n0 * Vt)) - np.exp(Rs0 * Jsc / (n0 * Vt))))
            print(f"Initial guess:\n\t Jph0={Jph0*1000:.3e}[mA/cm^2], Js0={Js0*1000:.3e}[mA/cm^2],\n\t Rs0={Rs0:.3f}[Ohm.cm^2], Rp0={Rp0:.3f}[Ohm.cm^2], n0={n0:.2f}")
        elif model == '2':
            Js10 = (Jsc * (Rp0 - Rs0) - Voc) / (Rp0 * (np.exp(Voc / (n1 * Vt)) - np.exp(Rs0 * Jsc / (n1 * Vt))))
            Js20 = Jsc / (np.exp(Voc * q / (k * T)) - 1)
            print(f"Initial guess:\n\t Jph0={Jph0*1000:.3e}[mA/cm^2], Js1={Js10*1000:.3e}[mA/cm^2], Js2={Js20*1000:.3e}[mA/cm^2],\n\t Rs0={Rs0:.3f}[Ohm.cm^2], Rp0={Rp0:.3f}[Ohm.cm^2]")
    else:
        print("\n\t $$$ The input IV curve is a DARK current! $$$")
        Jph0 = 0
        Js0 = -((I0[(DSize + p) // 2] / (np.exp(V0[(DSize + p) // 2] / Vt) - 1) + I0[DSize-1] / (np.exp(V0[DSize-1] / Vt) - 1)) / 2)
        n0 = 1
        Rs0 = (abs((V0[s+1] - V0[s-2]) / (I0[s+1] - I0[s-2])) + abs((V0[s+2] - V0[s-1]) / (I0[s+2] - I0[s-1]))) / 2
        Rp0 = (abs((V0[p+1] - V0[p-2]) / (I0[p+1] - I0[p-2])) + abs((V0[p+2] - V0[p-1]) / (I0[p+2] - I0[p-1]))) / 2
        Js10 = Js0
        Js20 = Js0

# Main function
def main():
    global A, T, Vt, DSize, V0, I0, CurrentUnit, model, Jph0, Js0, Rs0, Rp0, n0, Js10, Js20
    print("Parameter extraction of solar cells based on a single/double diode model using Particle Swarm Optimization")
    print("The input IV curve data must be a 2-column ASCII file (V I)")

    if len(sys.argv) != 2:
        print("Usage: python pso_pviv_fit.py IV-file")
        return

    print("Current Density unit in the IV file (select 1 or 2):")
    print("\t 1: A/cm^2")
    print("\t 2: mA/cm^2")
    while CurrentUnit not in ['1', '2']:
        CurrentUnit = input().strip()

    print("Input Solar Cell Size and Temperature: [default: 1.0cm^2, 300K]")
    A, T = map(float, input().split(','))
    Vt = 0.025875 * T / 300

    try:
        with open(sys.argv[1], 'r') as myFile:
            lines = myFile.readlines()
            DSize = len(lines)
            V0 = np.zeros(DSize)
            I0 = np.zeros(DSize)
            for i, line in enumerate(lines):
                V0[i], I0[i] = map(float, line.split())
                I0[i] /= A
                if CurrentUnit == '2':
                    I0[i] /= 1000
    except FileNotFoundError:
        print("Error opening input file!")
        sys.exit(1)

    print("Which model should be used for the IV curve fitting (enter 1 or 2)?:")
    print("\t 1: a single-diode model")
    print("\t 2: a double-diode model")
    model = input().strip()
    if model not in ['1', '2']:
        print("Error! This model is not available, please enter either 1 or 2, Bye!")
        return
    if model == '1':
        print("Single-diode model is selected:\n\t IL1=Jph - Js*(exp((V0+I0*Rs)/(n*Vt)) -1) - (V0+I0*Rs)/Rp")
    else:
        print("Double-diode model is selected:\n\t IL2=Jph - Js1*(exp((V0+I0*Rs)/(n1*Vt)) -1) - Js2*(exp((V0+I0*Rs)/(n2*Vt)) -1) - (V0+I0*Rs)/Rp")
        print("diode ideality factors n1, and n2,\n\t are fixed to 1 and 2 to represent the diffusion and recombination current terms, respectively!")

    cell_perf()
    start = time.time()
    np.random.seed(int(time.time()))

    k = 0
    while True:
        print(f"PSO is working on Model {model}, please wait:")
        PSO_func()
        k += 1
        best_gen_number, best = min_fit(result)
        print(f"\nAfter iterating {k*maxgen} times, the optimal value is: {best:.3e}.")

        if model == '1':
            Jph0, Js0, Rs0, Rp0, n0 = genbest[best_gen_number]
            print("\nSingle-diode model PSO fitting results:")
            print(f"\t Jph={1000*genbest[best_gen_number][0]:.3f}[mA/cm^2], Js={1000*genbest[best_gen_number][1]:.3e}[mA/cm^2],\n\t Rs={genbest[best_gen_number][2]:.3e}[Ohm/cm^2], Rp={genbest[best_gen_number][3]:.3e}[Ohm/cm^2], n={genbest[best_gen_number][4]:.3f}")
        else:
            Jph0, Js10, Js20, Rs0, Rp0 = genbest[best_gen_number]
            print("\nDouble-diode model PSO fitting results:")
            print(f"\t Jph={1000*genbest[best_gen_number][0]:.3f}[mA/cm^2], Js1={1000*genbest[best_gen_number][1]:.3e}[mA/cm^2], Js2={1000*genbest[best_gen_number][2]:.3e}[mA/cm^2],\n\t Rs={genbest[best_gen_number][3]:.3f}[Ohm/cm^2], Rp={genbest[best_gen_number][4]:.3f}[Ohm/cm^2]")

        print("Continue the PSO fitting? [y/n]")
        chr = input().strip()
        if chr.lower() != 'y':
            break

    duration = time.time() - start
    print(f"\n#- Program running time: {duration:.3f} seconds -#")

    out = sys.argv[1][:6] + ("_fit1.dat" if model == '1' else "_fit2.dat")
    try:
        with open(out, 'w') as fp:
            fp.write("Bias[V] Jexp[{}] Jfit[{}]\n".format("mA/cm2" if CurrentUnit == '2' else "A/cm2", "mA/cm2" if CurrentUnit == '2' else "A/cm2"))
            for i in range(DSize):
                if model == '1':
                    fit = IL1(V0[i], I0[i], *genbest[best_gen_number])
                else:
                    fit = IL2(V0[i], I0[i], *genbest[best_gen_number])
                if CurrentUnit == '2':
                    fp.write(f"{V0[i]:.6f} {1000*I0[i]:.6f} {1000*fit:.6f}\n")
                else:
                    fp.write(f"{V0[i]:.6f} {I0[i]:.6f} {fit:.6f}\n")
        print(f"\n\t Simulated curve is saved in '{out}'!")
    except IOError:
        print(f"Error opening OUTPUT file {out}!")
        sys.exit(1)

if __name__ == "__main__":
    main()
