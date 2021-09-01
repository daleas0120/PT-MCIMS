
import sys
print(sys.path)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import trange
from utils.MCIM import spinVis
from utils.MCIM import spinLattice
from utils.MCIM import metropolisAlgorithm

def routine(p):
    beta_array = np.linspace(p.T_MIN, p.T_MAX, p.numTemp, endpoint=True)
    # Output parameters
    d = {'beta': beta_array}
    dataset = pd.DataFrame(data=d)

    # Initialize main variables
    spins = spinLattice(p.M, p.N, p.J, p.mu, 'oop')
    params = metropolisAlgorithm(p.BURN_IN, p.STEPS, beta_array)

    J_matrix = spins.initJmatrix()
    spinVis(J_matrix, 'J_matrix.png')

    mu_matrix = spins.initFieldmatrix()
    spinVis(mu_matrix, 'mu_matrix.png')

    nHS_A_tmp = np.zeros((p.STEPS, len(beta_array)))
    nHS_B_tmp = np.zeros((p.STEPS, len(beta_array)))

    t = time.time()

    L = spins.initPlaneCoupledLattice()
    spinVis3D(L, 'OG_spins_oop.png')

    for i in trange(p.numTrials):
        df, L, nHS_A_tmp, nHS_B_tmp = \
            params.anneal(spins, L, J_matrix, mu_matrix, nHS_A_tmp, nHS_B_tmp, i)

        dataset = pd.concat([dataset, df], axis=1)
