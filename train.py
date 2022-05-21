from data_import import data, make_frame, make_list
import pygad.nn
import pygad.gann
import numpy

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
    global GANN_instance, data_inputs, data_outputs

    prediction = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],data_inputs=data_inputs[0].reshape((1, 40)))
    prediction = int(prediction[0])
    solution_fitness = 1
    print(data_inputs[0])
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
