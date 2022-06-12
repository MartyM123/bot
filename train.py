from data_import import data, make_frame, make_list
import pygad.nn
import pygad.gann
import numpy
from wallet import *

# Preparing the NumPy array of the inputs.
data_inputs = make_list(make_frame(data()))

GANN_instance = pygad.gann.GANN(num_solutions=3, #numbers of solutions in one generation
                                num_neurons_input=40,
                                num_neurons_output=3,
                                num_neurons_hidden_layers=[60,60],
                                hidden_activations="relu",
                                output_activation="softmax")

GANN_instance.num_neurons_output=3
#0 buy
#1 sell
#2 wait

def fitness_func(solution, sol_idx):
    global GANN_instance, data_inputs

    wall = wallet(start_budget=1000)

    for i in range(len(data_inputs[0])):

        prediction = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],data_inputs=data_inputs[0].reshape((1, 40)))

        if prediction[0] == 0:
            #buy
            wall.buy(crypto_am=1, price=data_inputs[0][-1])

        elif prediction[0] == 1:
            #sell
            wall.sell(crypto_am=2, price=data_inputs[0][-1])

        else:
            #wait
            pass

    wall.sell_all(price=data_inputs[0][-1])
    solution_fitness = wall.usd-wall.start_budget

    return solution_fitness



population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

initial_population = population_vectors.copy()

ga_instance = pygad.GA(num_generations=20,

                       mutation_num_genes=2,

                       num_parents_mating = 2,

                       initial_population=initial_population,

                       fitness_func=fitness_func,

                       sol_per_pop = 50,

                       mutation_percent_genes=5,

                       init_range_low=-2,

                       init_range_high=5,

                       parent_selection_type="sss",

                       crossover_type="single_point",

                       mutation_type='random',

                       keep_parents=1
                       )
ga_instance.sol_per_pop=50
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
