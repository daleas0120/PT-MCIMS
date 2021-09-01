import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import random
from tqdm import trange

class spinLattice:
    def __init__(self, rows, cols, J, mu, type, k):
        self.rows = rows
        self.cols = cols
        self.J = J
        self.mu = mu
        self.mu_matrixA = np.empty(1)
        self.mu_matrixB = np.empty(1)
        self.couplingType = type
        self.latticeA = np.empty(1)
        self.latticeB = np.empty(1)
        self.lattice = np.empty(1)
        self.k = k

    def initLattice(self):
        M = self.rows
        N = self.cols

        # Create randomized lattice
        L = np.random.rand(M, N)

        # Assign points to high spin (HS) or low spin (LS) state
        L[L < 0.5] = -1
        L[L >= 0.5] = 1
        self.lattice = L

        return L

    def initLineCoupledLattice(self):

        """

        :param M:
        :param N:
        :return:
        """
        M = self.rows
        N = self.cols

        # Create randomized lattice
        self.latticeA = np.random.rand(M, N)
        self.latticeB = np.random.rand(M, N)

        # Assign points to high spin (HS) or low spin (LS) state
        self.latticeA[self.latticeA < 0.5] = -1
        self.latticeA[self.latticeA >= 0.5] = 1
        self.latticeB[self.latticeB < 0.5] = -1
        self.latticeB[self.latticeB >= 0.5] = 1

        self.lattice = np.concatenate((self.latticeA, self.latticeB), axis=1)

        return self.lattice

    def initPlaneCoupledLattice(self):
        """

        :param N:
        :return:
        """
        M = self.rows
        N = self.cols

        # Create randomized lattice
        self.latticeA = np.random.rand(M, N)
        self.latticeB = np.random.rand(M, N)

        # Assign points to high spin (HS) or low spin (LS) state
        self.latticeA[self.latticeA < 0.5] = -1
        self.latticeA[self.latticeA >= 0.5] = 1
        self.latticeB[self.latticeB < 0.5] = -1
        self.latticeB[self.latticeB >= 0.5] = 1

        self.lattice = np.concatenate((self.latticeA, self.latticeB), axis=1)

        return self.lattice

    def initPointCoupledLattice(self):
        """

        :param M:
        :param N:
        :return:
        """
        M = self.rows
        N = self.cols

        # Create randomized lattice
        self.latticeA = np.random.rand(M, N)
        self.latticeB = np.random.rand(M, N)

        # Assign points to high spin (HS) or low spin (LS) state
        self.latticeA[self.latticeA < 0.5] = -1
        self.latticeA[self.latticeA >= 0.5] = 1
        self.latticeB[self.latticeB < 0.5] = -1
        self.latticeB[self.latticeB >= 0.5] = 1

        self.lattice = np.concatenate((self.latticeA, self.latticeB), axis=1)

        return self.lattice

    def initJmatrix(self):
        M = self.rows
        N = self.cols
        J = self.J

        J_matrix = J * np.ones((M, N))

        return J_matrix


    def initFieldmatrix(self):
        mu = self.mu
        M = self.rows
        N = self.cols
        type = self.couplingType

        if type == 'oop':
            self.mu_matrixA = mu * np.ones((M, N))
            self.mu_matrixB = -1*mu*np.ones((M, N))

        else:
            self.mu_matrixA = mu * np.ones((M, N))
            self.mu_matrixB = -1*mu*np.ones((M, N))


class metropolisAlgorithm():
    def __init__(self, BURN_IN, STEPS, TEMP):
        self.burn_in = BURN_IN
        self.steps = STEPS
        self.temp = TEMP

    def MC_step_plane(self, spins, J, beta, step):
        """
        Completes a single Monte Carlo step for the entire matrix L given parameters

        :param L: spin matrix
        :param J: coupling matrix
        :param mu: mean field matrix
        :param beta: temperature parameter
        :return: annealed lattice
        """
        A = spins.latticeA
        B = spins.latticeB
        M = spins.rows
        N = spins.cols

        def loop_over_spins(L, L2, M, N, J_matrix, mu_matrix):
            """
            Function that performs the MC step
            :param L: The primary lattice under consideration
            :param L2: The secondary lattice coupled to the primary lattice
            :param M: Number of rows
            :param N: Number of columns
            :param J_matrix: The matrix of coupling values J at each point
            :param mu_matrix: The matrix of mean field values mu at each point
            :return: the annealed primary lattice
            """
            for row in range(0, M):
                for col in range(0, N):

                    i = random.randint(0, M-1)
                    j = random.randint(0, N-1)

                    sumNN = J_matrix[((i - 1) + M) % M, j] * L[((i - 1) + M) % M, j] + \
                            J_matrix[(i + 1) % M, j] * L[(i + 1) % M, j] + \
                            J_matrix[i, ((j - 1) + N) % N] * L[i, ((j - 1) + N) % N] + \
                            J_matrix[i, ((j + 1) % N)] * L[i, ((j + 1) % N)] + \
                            spins.k*L2[(i, j)]

                    E_a = -0.5 * L[i, j] * sumNN + mu_matrix[i, j] * L[i, j]

                    E_b = -1 * E_a

                    if E_b < E_a:
                        L[i, j] = -1 * L[i, j]
                    elif np.exp(-1 * (E_b - E_a) * beta) > np.random.rand():
                        L[i, j] = -1 * L[i, j]

        if step % 2 == 0:
            loop_over_spins(B, A, M, N, J, spins.mu_matrixB)
            loop_over_spins(A, B, M, N, J, spins.mu_matrixA)
        else:
            loop_over_spins(A, B, M, N, J, spins.mu_matrixA)
            loop_over_spins(B, A, M, N, J, spins.mu_matrixB)

    def MC_step_line(self, spins, J, beta, step):
        """
        Completes a single Monte Carlo step for the entire matrix L given parameters

        :param L: spin matrix
        :param J: coupling matrix
        :param mu: mean field matrix
        :param beta: temperature parameter
        :return: annealed lattice
        """
        A = spins.latticeA
        B = spins.latticeB
        M = spins.rows
        N = spins.cols

        def loop_over_spins(L, L2, M, N, J_matrix, mu_matrix):

            for row in range(0, M):
                for col in range(0, N):

                    i = random.randint(0, M-1)
                    j = random.randint(0, N-1)

                    if i==math.floor(M/2):
                        sumNN = J_matrix[((i - 1) + M) % M, j] * L[((i - 1) + M) % M, j] + \
                                J_matrix[(i + 1) % M, j] * L[(i + 1) % M, j] + \
                                J_matrix[i, ((j - 1) + N) % N] * L[i, ((j - 1) + N) % N] + \
                                J_matrix[i, ((j + 1) % N)] * L[i, ((j + 1) % N)] + \
                                spins.k*L2[(i, j)]
                    else:
                        sumNN = J_matrix[((i - 1) + M) % M, j] * L[((i - 1) + M) % M, j] + \
                                J_matrix[(i + 1) % M, j] * L[(i + 1) % M, j] + \
                                J_matrix[i, ((j - 1) + N) % N] * L[i, ((j - 1) + N) % N] + \
                                J_matrix[i, ((j + 1) % N)] * L[i, ((j + 1) % N)]

                    E_a = -0.5 * L[i, j] * sumNN + mu_matrix[i, j] * L[i, j]

                    E_b = -1 * E_a

                    if E_b < E_a:
                        L[i, j] = -1 * L[i, j]
                    elif np.exp(-1 * (E_b - E_a) * beta) > np.random.rand():
                        L[i, j] = -1 * L[i, j]

        if step % 2 == 0:
            loop_over_spins(B, A, M, N, J, spins.mu_matrixB)
            loop_over_spins(A, B, M, N, J, spins.mu_matrixA)
        else:
            loop_over_spins(A, B, M, N, J, spins.mu_matrixA)
            loop_over_spins(B, A, M, N, J, spins.mu_matrixB)

    def MC_step(self, spins, J, beta, step):
        """
        Completes a single Monte Carlo step for the entire matrix L given parameters

        :param L: spin matrix
        :param J: coupling matrix
        :param mu: mean field matrix
        :param beta: temperature parameter
        :return: annealed lattice
        """
        A = spins.latticeA
        B = spins.latticeB
        M = spins.rows
        N = spins.cols

        def loop_over_spins(L, L2, M, N, J_matrix, mu_matrix):

            for row in range(0, M):
                for col in range(0, N):

                    i = random.randint(0, M-1)
                    j = random.randint(0, N-1)

                    if (i==math.floor(M/2) and j == math.floor(N/2)):
                        sumNN = J_matrix[((i - 1) + M) % M, j] * L[((i - 1) + M) % M, j] + \
                                J_matrix[(i + 1) % M, j] * L[(i + 1) % M, j] + \
                                J_matrix[i, ((j - 1) + N) % N] * L[i, ((j - 1) + N) % N] + \
                                J_matrix[i, ((j + 1) % N)] * L[i, ((j + 1) % N)] + \
                                spins.k*L2[(i, j)]
                    else:
                        sumNN = J_matrix[((i - 1) + M) % M, j] * L[((i - 1) + M) % M, j] + \
                                J_matrix[(i + 1) % M, j] * L[(i + 1) % M, j] + \
                                J_matrix[i, ((j - 1) + N) % N] * L[i, ((j - 1) + N) % N] + \
                                J_matrix[i, ((j + 1) % N)] * L[i, ((j + 1) % N)]

                    deltaSig = -1 * L[i, j] - L[i, j]

                    L[i, j] *= -1

                    dE = deltaSig * (-1 * sumNN + mu_matrix[i, j])

                    p = math.exp(-1 * dE * beta)

                    if dE < 0 or p >= np.random.rand():
                        continue
                    else:
                        L[i, j] *= -1

                    # E_a = -0.5 * L[i, j] * sumNN + mu_matrix[i, j] * L[i, j]
                    #
                    # E_b = -1 * E_a
                    #
                    # if E_b < E_a:
                    #     L[i, j] = -1 * L[i, j]
                    # elif np.exp(-1 * (E_b - E_a) * beta) > np.random.rand():
                    #     L[i, j] = -1 * L[i, j]

        if step % 2 == 0:
            loop_over_spins(B, A, M, N, J, spins.mu_matrixB)
            loop_over_spins(A, B, M, N, J, spins.mu_matrixA)
        else:
            loop_over_spins(A, B, M, N, J, spins.mu_matrixA)
            loop_over_spins(B, A, M, N, J, spins.mu_matrixB)

    def MC_3Dstep(self, L, J_matrix, mu_matrix, beta):
        """
        Completes a single Monte Carlo step for the entire matrix L given parameters

        :param L: spin matrix
        :param J_matrix: coupling matrix
        :param mu_matrix: mean field matrix
        :param beta: temperature parameter
        :return: annealed lattice
        """

        M = L.shape[0]
        N = L.shape[1]

        for i in range(0, M):
            for j in range(0, N):
                for k in range(1, 3):

                    sumNN = J_matrix[((i - 1) + M) % M, j, k] * L[((i - 1) + M) % M, j, k] + \
                            J_matrix[(i + 1) % M, j, k] * L[(i + 1) % M, j, k] + \
                            J_matrix[i, ((j - 1) + N) % N, k] * L[i, ((j - 1) + N) % N, k] + \
                            J_matrix[i, ((j + 1) % N), k] * L[i, ((j + 1) % N), k] + \
                            J_matrix[i, j, (k + 1) % N] * L[i, j, (k + 1) % N] + \
                            J_matrix[i, j, (k - 1) % N] * L[i, j, (k - 1) % N]

                    deltaSig = -1 * L[i, j, k] - L[i, j, k]

                    L[i, j, k] *= -1

                    dE = deltaSig * (-1 * sumNN + mu_matrix[i, j, k])

                    p = math.exp(-1 * dE * beta)

                    if dE < 0 or p >= np.random.rand():
                        continue
                    else:
                        L[i, j, k] *= -1

                    # E_a = -0.5 * L[i, j, k] * sumNN + mu_matrix[i, j, k] * L[i, j, k]
                    #
                    # E_b = -1 * E_a
                    #
                    # if E_b < E_a:
                    #     L[i, j, k] = -1 * L[i, j, k]
                    # elif np.exp(-1 * (E_b - E_a) * beta) > np.random.rand():
                    #     L[i, j, k] = -1 * L[i, j, k]

        return L

    def anneal(self, spins, J_matrix, mu_matrix, nHS_A_tmp, nHS_B_tmp, i):

        beta_array = self.temp
        steps = self.steps
        burn_in = self.burn_in

        time.sleep(0.01)
        nHS_A = []
        nHS_B = []

        # spinVis(L, 'spinsOG_'+str(i)+'.png')

        for idx in range(len(beta_array)):
            beta = beta_array[idx]

            for b in range(burn_in):
                if spins.couplingType == 'oop':
                    self.MC_step_plane(spins, J_matrix, mu_matrix, beta)
                if spins.couplingType == 'line':
                    self.MC_step_line(spins, J_matrix, beta, s)
                else:
                    self.MC_step(spins, J_matrix, beta, s)

            for s in trange(steps):
                if spins.couplingType == 'oop':
                    self.MC_step_plane(spins, J_matrix, beta, s)
                if spins.couplingType == 'line':
                    self.MC_step_line(spins, J_matrix, beta, s)
                else:
                    self.MC_step(spins, J_matrix, beta, s)

                # Get averaged values
                nHS_A_tmp[s] = 0.5 * (1 + np.mean(spins.latticeA))
                nHS_B_tmp[s] = 0.5 * (1 + np.mean(spins.latticeB))

            nHS_A.append(nHS_A_tmp)
            nHS_B.append(nHS_B_tmp)

        colA = 'nHS_A_' + str(i)
        colB = 'nHS_B_' + str(i)

        d = {colA: nHS_A, colB: nHS_B}
        df = pd.DataFrame(data=d)

        return df, d, nHS_A_tmp, nHS_B_tmp


def spinVis(spins, imgName):
    fig, ax = plt.subplots(1, 1)

    ax.imshow(spins)

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affectedg
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        left=False,
        labelleft=False) # labels along the bottom edge are off
    plt.savefig(imgName)
    #plt.show()
    plt.clf()

def spinVis3D(spins, imgName):
    x = np.linspace(0, spins.shape[0], num=spins.shape[0], endpoint=True)
    y = np.linspace(0, spins.shape[1], num=spins.shape[1], endpoint=True)
    z = np.linspace(0, spins.shape[2], num=spins.shape[2], endpoint=True)

    xpts, ypts, zpts = np.meshgrid(x, y, z)

    xpts = xpts.flatten()
    ypts = ypts.flatten()
    zpts = zpts.flatten()
    spins = spins.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xpts, ypts, zpts, c=(spins), alpha=0.5)

    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        left=False,
        labelleft=False)  # labels along the bottom edge are off

    plt.savefig(imgName)
    #plt.show()

    plt.clf()


    bp = 0

def torusPlotPoint(spins, figName, save=False):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    colorMapA = cm.viridis(spins.latticeA+1)
    colorMapB = cm.viridis(spins.latticeB+1)

    n = spins.rows

    theta = np.linspace(0, 2. * np.pi, n)
    phi = np.linspace(0, 2. * np.pi, n)
    theta, phi = np.meshgrid(theta, phi)

    #radius c and tube radius a

    c, a = 4, 1
    x = (c + a * np.cos(theta)) * np.cos(phi) - (c+a+0.05)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

    x1 = (c + a * np.cos(theta)) * np.cos(phi) + (c+a+0.05)
    y1 = (c + a * np.cos(theta)) * np.sin(phi)
    z1 = a * np.sin(theta)

    fig, ax2 = plt.subplots(figsize=(6,4), dpi=90, subplot_kw={"projection": "3d"})

    ax2.set_zlim(-5, 5)
    ax2.set_xlim(-9, 9)
    ax2.set_ylim(-9, 9)
    ax2.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colorMapA, alpha=0.05)
    ax2.plot_surface(x1, y1, z1, rstride=1, cstride=1, facecolors=colorMapB, alpha=0.05)
    ax2.view_init(36, 90)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.set_axis_off()
    plt.tight_layout()
    if save:
        plt.savefig(figName, transparent=True)

    #plt.show()

def torusPlotLine(spins, figName, save=False):
    from matplotlib import cm

    colorMapA = cm.viridis(spins.latticeA+1)
    colorMapB = cm.viridis(spins.latticeB+1)

    n = spins.rows

    theta = np.linspace(0, 2. * np.pi, n)
    phi = np.linspace(0, 2. * np.pi, n)
    theta, phi = np.meshgrid(theta, phi)

    #radius c and tube radius a

    c, a = 4, 1
    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta) - (a+0.1)

    x1 = (c + a * np.cos(theta)) * np.cos(phi)
    y1 = (c + a * np.cos(theta)) * np.sin(phi)
    z1 = a * np.sin(theta) + (a+0.1)

    fig, ax2 = plt.subplots(figsize=(6,4), dpi=90, subplot_kw={"projection": "3d"})

    ax2.set_zlim(-5, 5)
    ax2.set_xlim(-9, 9)
    ax2.set_ylim(-9, 9)
    ax2.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colorMapA, alpha=0.05)
    ax2.plot_surface(x1, y1, z1, rstride=1, cstride=1, facecolors=colorMapB, alpha=0.05)
    ax2.view_init(36, 90)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.set_axis_off()
    plt.tight_layout()
    if save:
        plt.savefig(figName, transparent=True)

    #plt.show()

def torusPlotSurface(spins, figName, save=False):
    from matplotlib import cm

    colorMapA = cm.viridis(spins.latticeA+1)
    colorMapB = cm.viridis(spins.latticeB+1)

    n = spins.rows

    theta = np.linspace(0, 2. * np.pi, n)
    phi = np.linspace(0, 2. * np.pi, n)
    theta, phi = np.meshgrid(theta, phi)

    #radius c and tube radius a
    c = 4
    a1 = 1
    a2 = 0.5

    x = (c + a1 * np.cos(theta)) * np.cos(phi)
    y = (c + a1 * np.cos(theta)) * np.sin(phi)
    z = a1 * np.sin(theta)

    x1 = (c + a2 * np.cos(theta)) * np.cos(phi)
    y1 = (c + a2 * np.cos(theta)) * np.sin(phi)
    z1 = a2 * np.sin(theta)

    fig, ax2 = plt.subplots(figsize=(6,4), dpi=90, subplot_kw={"projection": "3d"})

    ax2.set_zlim(-5, 5)
    ax2.set_xlim(-9, 9)
    ax2.set_ylim(-9, 9)
    ax2.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colorMapA, alpha=0.05)
    ax2.plot_surface(x1, y1, z1, rstride=1, cstride=1, facecolors=colorMapB, alpha=0.05)
    ax2.view_init(36, 90)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.set_axis_off()
    plt.tight_layout()
    if save:
        plt.savefig(figName, transparent=True)

    #plt.show()


