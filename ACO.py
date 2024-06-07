import numpy as np
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


# calculate distance of a path between any 2 cities
def distance(cityA, cityB):
    d = math.sqrt(
            math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))
    return d



# ACO algorithm
def ant_colony_optimization(cities, n_ants, n_generations, alpha, beta, evaporation_rate, Q, target):
    n_cities = len(cities)
    pheromone = np.ones((n_cities, n_cities))
    best_path = None
    best_path_length = np.inf
    gen_number = 0
    fitness_scores = []
    avg_fitness_scores = []
    start_time = time.time()
    
    for generation in range(n_generations):
        paths = []
        path_lengths = []
        
        
        for ant in range(n_ants):
            visited = [False]*(n_cities)
            current_city = np.random.randint(n_cities)
            visited[current_city] = True
            path = [current_city]
            path_length = 0
            
            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))
                
                for i, unvisited_city in enumerate(unvisited):
                    probabilities[i] = pheromone[current_city, unvisited_city]**alpha / distance(cities[current_city], cities[unvisited_city])**beta
                
                probabilities /= np.sum(probabilities)
                
                next_city = np.random.choice(unvisited, p=probabilities)
                path.append(next_city)
                path_length += distance(cities[current_city], cities[next_city])
                visited[next_city] = True
                current_city = next_city
            
            paths.append(path)
            path_lengths.append(path_length)
            
            
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length
        
        # Update pheromones
        
        pheromone *= evaporation_rate
        
        for path, path_length in zip(paths, path_lengths):
            for i in range(n_cities-1):
                pheromone[path[i], path[i+1]] += Q/path_length
            pheromone[path[-1], path[0]] += Q/path_length
        
        
        #calculate average fitness
        total_fitness = sum(1 / individual for individual in path_lengths)
        avg_fitness = total_fitness / len(path_lengths)
        avg_fitness_scores.append(avg_fitness)
        
        
        #calculate best fitness
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
            
    
    return best_path, best_path_length, gen_number, fitness_scores, avg_fitness_scores        

            
# draw cities and answer map
def drawMap(city, best_path):
    for i in range(len(best_path) - 1):
        first_index = best_path[i]
        second_index = best_path[i + 1]
        first_city = city[first_index]
        second_city = city[second_index]

        plt.plot([first_city[1]], [first_city[2]], "ro") 
        plt.annotate(first_city[0], (first_city[1], first_city[2]))
        plt.plot([second_city[1]], [second_city[2]], "ro")  
        plt.annotate(second_city[0], (second_city[1], second_city[2]))
        
        plt.plot([first_city[1], second_city[1]], [first_city[2], second_city[2]], 'gray')  

    first_index = best_path[0]
    last_index = best_path[-1]
    first_city = city[first_index]
    last_city = city[last_index]
    plt.plot([first_city[1]], [first_city[2]], "ro")  
    plt.plot([last_city[1]], [last_city[2]], "ro")  
    plt.plot([first_city[1], last_city[1]], [first_city[2], last_city[2]], 'gray')  

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Final Path for TSP Solution')
    plt.grid(True)
    plt.savefig("Optimal path")
    plt.clf()


# Plot fitness graph
def plot_fitness_graph(fitness_scores, avg_fitness_scores):
    generations = range(1, len(fitness_scores) + 1)
    plt.plot(generations, fitness_scores, label='Best Fitness', linestyle='-')
    plt.plot(generations, avg_fitness_scores, label='Average Fitness', linestyle='--')
    plt.title('Fitness Graph')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig("Fitness over Generation")
    plt.clf()

    
def main():
    # initial values
    n_generations = 200
    n_ants = 25
    alpha = 0.9
    beta = 1.5
    evaporation_rate = 0.9
    Q = 10
    target = 450.0

    cities = getCity()

    best_path, best_path_length, genNumber, fitness_scores, avg_fitness_scores = ant_colony_optimization(
        cities, n_ants, n_generations, alpha, beta, evaporation_rate, Q, target,
    )

    print("\n----------------------------------------------------------------")
    print("Generation: " + str(genNumber))
    print("Minimum distance after training: " + str(best_path_length))
    print("Target distance: " + str(target))
    print("----------------------------------------------------------------\n")
    
    drawMap(cities, best_path)
    plot_fitness_graph(fitness_scores, avg_fitness_scores)


main()

