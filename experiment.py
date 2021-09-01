import os
import json
import sys
import PTMCIM_A
import PTMCIM_B
import PTMCIM_C
print(sys.path)


class parameters:

    def __init__(self):

        # Spin lattice parameters
        self.M = 300  # Number of rows
        self.N = 300  # Number of cols
        self.J = 1  # Coupling between locations
        self.mu = 0.003  # field
        self.k = 20  # coupling between lattices

        # Metropolis Algorithm parameters
        self.numTrials = 1
        self.BURN_IN = 0
        self.STEPS = 5000  # How many time steps
        self.BOLTZ_CONST = 8.617333e-5
        self.T_MIN = 0
        self.T_MAX = 10
        self.numTemp = 150
        self.T_C = 0.44


def main():

    params = parameters()

    k_values = [1, 2, 3, 5, 10, 20, 30, 50]
    numTrials = 3
    # type = {1: 'point', 2: 'line', 3:'plane'}
    case_type = {1: 'point', 2: 'line'}

    for case in case_type:
        case_name = case_type[case]
        os.mkdir(case_name)
        os.chdir(case_name)

        for k in k_values:

            params.k = k

            trial_folder = case_name + ', ' + 'k = ' + str(k)
            os.mkdir(trial_folder)
            os.chdir(trial_folder)

            for trial in range(numTrials):

                print(case_name+', k = '+str(k)+', trialNum = ' + str(trial))

                os.mkdir(str(trial))
                os.chdir(str(trial))

                if case_name == 'point':
                    PTMCIM_A.routine(params)
                elif case_name == 'line':
                    PTMCIM_B.routine(params)
                elif case_name == 'plane':
                    PTMCIM_C.routine(params)

                # save parameters to folder with data
                with open("params.json", "w") as outfile:
                    json.dump(params.__dict__, outfile)

                path_parent = os.path.dirname(os.getcwd())
                # go up one level to
                os.chdir(path_parent)

            # new k value follows, so leave this k value level
            os.chdir(os.path.dirname(os.getcwd()))

        # new case type follows, so leave this case level and go up one
        os.chdir(os.path.dirname(os.getcwd()))


if __name__ == "__main__":
    main()
