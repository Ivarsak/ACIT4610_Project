
import multiprocessing
import os
import pickle
import matplotlib.pyplot as plt
import neat
import numpy as np
import gym
import visualize

#Note:

#The following code was developed based on the examples in the following links:
#https://github.com/neat-python/neat-python/tree/master/examples/pole_balancing/single_pole
#https://github.com/Sentdex/NEAT-samples


fitness_mean = []
runs_per_net = 2
#simulation_seconds = 60.0
#time_const = 0.01
fitness_log = []
# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    #net = neat.ctrnn.CTRNN.create(genome, config,time_const)
    fitnesses = []


    for runs in range(runs_per_net):
        env = gym.make("CartPole-v1")

        observation = env.reset()
        fitness = 0.0
        done = False
        while not done:
            #env.render()
            action = net.activate(observation)
            observation, reward, done, info = env.step(int(action[0]))
            fitness += reward

        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    logger = []
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward_network')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)
    print(winner)

    # Save the winner.

    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    #fitness_log.remove('[None]')
    #print(winner)
    #np.savetxt('neat_test.csv', fitness_log, delimiter=',')

    #visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'Velocity', -3: 'Theta', -4: 'Omega', 0: 'Output'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)




if __name__ == '__main__':
    run()




