import numpy as np

def total_dist(route, cost):
  d = 0.0  # total distance between cities
  n = len(route)
  for i in range(n-1):
    d+= cost[route[i], route[i+1]]
  return d

def initial_state(n, rnd, pop_size):
  pop = np.zeros((pop_size, n), dtype=np.int64)
  for i in range(pop_size):
    pop[i] = np.arange(n, dtype=np.int64)
    rnd.shuffle(pop[i])
  return pop

def fitness(pop, cost, rnd):
  n = len(pop)
  new_pop = np.zeros((n,len(pop[0])), dtype=np.int64)
  fit = np.zeros(n)
  total = 0.0
  for i in range(n):
    fit[i] = 1/total_dist(pop[i], cost)
    total += fit[i]
  fit[0] = fit[0] / total
  for i in range(n - 1):
    fit[i + 1] = fit[i + 1] / total + fit[i]

  for i in range(n):
    p = rnd.random()
    for j in range(n):
      if p < fit[j]:
        new_pop[i] = pop[j]
        break

  return new_pop

def select_parents(pop, rnd):
  # select pop in pairs, using all the population
  parents = np.arange(len(pop), dtype=np.int64)
  rnd.shuffle(parents)
  return parents

def crossover(parents, pop, rnd):
  n = len(parents)
  new_pop = np.zeros((n, len(pop[0])), dtype=np.int64)
  for i in range(0, n, 2):
    p1 = pop[parents[i]]
    p2 = pop[parents[i+1]]
    j = rnd.randint(len(pop[0]))
    aux = p1[j]
    p1[j] = p2[j]
    for k in range(len(p1)):
      if p1[k] == p1[j]:
        if k == j:
          continue
        p1[k] = aux
    aux2 = p2[j]
    p2[j] = aux
    for k in range(len(p2)):
      if p2[k] == p2[j]:
        if k == j:
          continue
        p2[k] = aux2
    new_pop[i] = p1
    new_pop[i+1] = p2

  return new_pop

def mutate(pop, rnd):
  n = len(pop)
  for i in range(n):
    p = rnd.random()
    if p < 0.01:
      j = rnd.randint(len(pop[0]))
      k = rnd.randint(len(pop[0]))
      tmp = pop[i][j]
      pop[i][j] = pop[i][k]
      pop[i][k] = tmp

  return pop

def best_route(pop, cost):
  n = len(pop)
  best = 0
  min_dist = total_dist(pop[0], cost)
  for i in range(1, n):
    d = total_dist(pop[i], cost)
    if d < min_dist:
      min_dist = d
      best = i
  return pop[best]



def solve(n, rnd, pop_size, max_iter, cost):
  pop = initial_state(n, rnd, pop_size)
  #print("Initial population: ")
  #print(pop)
  print("Initial distance: ")
  print(total_dist(best_route(pop,cost), cost))
  iteration = 0
  while iteration < max_iter:
    # fitness
    pop = fitness(pop, cost, rnd)
    # select parents
    parents = select_parents(pop, rnd)
    # crossover
    pop = crossover(parents, pop, rnd)
    # mutate
    pop = mutate(pop, rnd)

    iteration += 1
  
  return best_route(pop, cost)

def main():
  n = 20
  pop_size = 10
  max_iter = 2000
  rnd = np.random.RandomState(4)
  cost = rnd.randint(1, 11, size=(n, n))
  #print(cost)
  solution = solve(n, rnd, pop_size, max_iter, cost)
  print("Best route: ")
  print(solution)
  print("Best distance: ")
  print(total_dist(solution, cost))

if __name__ == "__main__":
  main()