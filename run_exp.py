
import numpy as np

from afpo import AgeFitnessPareto
from simulation import visualize

experiment_parameters = {
    'num_trials': 5,
    'target_population_size': 500,
    'max_generations': 2000,
}

def main():
    # Try running with 0 extra layers (control, traditional CA) or with 2 extra
    # layers (hierarchical CA)
    # TODO: Do we also want to try with and without growth enabled?
    for layers in (0, 2):
        label = 'Control' if layers == 0 else 'Experiment'
        fitness_scores = []
        # Run a few instances of each
        for trial in range(experiment_parameters['num_trials']):
            experiment_parameters['layers'] = layers
            # TODO: It would be more efficient to run all trials for both
            # experiment and control in one batch, but that would also take
            # some non-trivial refactoring. We should add that optimization
            # only if necessary.
            single_run = AgeFitnessPareto(experiment_parameters)
            sol = single_run.evolve()
            fitness_scores.append(sol.fitness)
            visualize(
                sol.phenotype[0], # Save just the first layer
                f'{label}_t{trial}_f{sol.fitness}.gif')

        print(f'Mean fitness for {label}: {np.mean(fitness_scores)}')

if __name__ == '__main__':
    main()
