from data_import import data, make_frame, make_list
import pygad.nn
import pygad.gann
import numpy

# Preparing the NumPy array of the inputs.
data_inputs = make_list(make_frame(data()))

num_inputs = data_inputs.shape[1]
num_classes = 2
num_classes = 3
num_solutions = 6

GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=num_inputs,
                                num_neurons_hidden_layers=[60,60],
                                num_neurons_output=num_classes,
                                hidden_activations=["relu"],
                                output_activation="softmax")

#0 buy
#1 sell
#2 wait

def fitness_func(solution, sol_idx):
    global GANN_instance, data_inputs, data_outputs

    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                   data_inputs=data_inputs)
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions/data_outputs.size)*100

    return solution_fitness



population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

initial_population = population_vectors.copy()

num_parents_mating = 4

num_generations = 500

mutation_percent_genes = 5

parent_selection_type = "sss"

crossover_type = "single_point"

keep_parents = 1

init_range_low = -2
init_range_high = 5


ga_instance = pygad.GA(num_generations=num_generations,
                       
                       mutation_num_genes=2,
                       
                       num_parents_mating=num_parents_mating,
                       
                       initial_population=initial_population,
                       
                       fitness_func=fitness_func,
                       
                       mutation_percent_genes=mutation_percent_genes,
                       
                       init_range_low=init_range_low,
                       
                       init_range_high=init_range_high,
                       
                       parent_selection_type=parent_selection_type,
                       
                       crossover_type=crossover_type,
                       
                       mutation_type='random',
                       
                       keep_parents=keep_parents,
                       
                       callback_generation=None)

ga_instance.run()
