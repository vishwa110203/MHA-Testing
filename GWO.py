import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt


# get cities info
def getCity():
    cities = []
    f = open("TSP51.txt")
    for i in f.readlines():
        node_city_val = i.split()
        cities.append(
            [node_city_val[0], float(node_city_val[1]), float(node_city_val[2])]
        )

    return cities


# calculating distance of the cities
def calcDistance(cities, wolf):
    total_sum = 0
    for i in range(len(cities) - 1):
        cityA = cities[wolf[i]]
        cityB = cities[wolf[i + 1]]

        d = math.sqrt(
            math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2)
        )

        total_sum += d

    cityA = cities[wolf[0]]
    cityB = cities[wolf[-1]]
    d = math.sqrt(math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))

    total_sum += d

    return total_sum


# calculate distance of a path between any 2 cities
def distance(cityA, cityB):
    d = math.sqrt(
            math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))
    return d



def pmx_crossover(parent1, parent2):
    """Partially Mapped Crossover (PMX)"""
    # Choose two random points for crossover
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))

    # Initialize offspring as a copy of the first parent
    child = parent1[:]

    # Map the portion between the two points from the second parent to the first parent
    for i in range(point1, point2):
        # Get the value from parent2
        gene = parent2[i]
        # Find the position of the value in parent1
        j = np.where(parent1 ==  gene)
        # Swap the values
        child[i], child[j] = child[j], child[i]

    # Repair the duplicated values
    for i in range(size):
        if i < point1 or i >= point2:
            while child[i] in child[point1:point2]:
                index = child.index(child[i], point1, point2)
                child[i], child[index] = child[index], child[i]

    return child



# GWO algorithm
def grey_wolf_optimization(cities, n_generations, n_wolves, target):
    n_cities = len(cities)
    gen_number = 0
    path_lengths = []
    fitness_scores = []
    avg_fitness_scores = []
    start_time = time.time()
    
    population = [np.random.permutation(n_cities) for _ in range(n_wolves)]
    
    for i in range(n_wolves):
        path_lengths.append(calcDistance(cities, population[i]))
    
    
    for generation in range (n_generations):
            
        alpha_index = path_lengths.index(min(path_lengths))
        path_lengths[alpha_index] = float('inf')  
        beta_index = path_lengths.index(min(path_lengths))
        path_lengths[beta_index] = float('inf')   
        delta_index = path_lengths.index(min(path_lengths))
            
            
        alpha_wolf = population[alpha_index]
        beta_wolf = population[beta_index]
        delta_wolf = population[delta_index]
        
        for i in range(n_wolves):
            s = []
            s.append(population[i])
            s1 = pmx_crossover(alpha_wolf, population[i])
            s.append(s1)
            s2 = pmx_crossover(beta_wolf, population[i])
            s.append(s2)
            s3 = pmx_crossover(delta_wolf, population[i])
            s.append(s3)
    
            for array in s:
                distance = calcDistance(cities, array)
                if distance < path_lengths[i]:
                    path_lengths[i] = distance
                    population[i] = array
                    
                    

            j, k = random.sample(range(len(population[i])), 2)
            # Swap elements at indices j and k
            population[i][j], population[i][k] = population[i][k], population[i][j]
            # Calculate the distance for the new array
            new_distance = calcDistance(cities, population[i])
            
            # If the new array is better, update the current distance and array
            if new_distance < path_lengths[i]:
                path_lengths[i] = new_distance
            else:
            # Revert the swap if the new array is not better
                population[i][j], population[i][k] = population[i][k], population[i][j]
                
        
        best_path_length = min(path_lengths)
        # Calculate average fitness
        total_fitness = sum(1 / i for i in path_lengths)
        avg_fitness = total_fitness / n_wolves
        avg_fitness_scores.append(avg_fitness)
        
        # Calculate best fitness
        best_fitness = 1 / best_path_length
        fitness_scores.append(best_fitness) 
        
        gen_number += 1  
        
        if gen_number % 10 == 0:
            print(gen_number, best_path_length, best_fitness)

        if best_path_length < target:
            break
        
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time: {:.4f} seconds".format(execution_time))     
    
    return gen_number, best_path_length, fitness_scores, avg_fitness_scores    
        
        
def main():
    # initial values
    n_generations = 200
    n_wolves = 50
    target = 450.0

    cities = getCity()
    genNumber, min_distance, fitness_scores, avg_fitness_scores = grey_wolf_optimization(
        cities,
        n_generations,
        n_wolves,
        target
    )

    print("\n----------------------------------------------------------------")
    print("Generation: " + str(genNumber))
    print("Minimum distance after training: " + str(min_distance))
    print("Target distance: " + str(target))
    print("----------------------------------------------------------------\n")        

main()    