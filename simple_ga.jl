module SimpleGa

# The fitness function is not in this module because it is specific to the
# problem we want to solve.
# Selection, crossover and mutation can all be done in a varieties of ways.

mutable struct Individual
    genes::Array{Float64,1}  # Also called chromosome

    # Fitness is computed by the fitness function
    # It is located in the SmartSnake module
    fitness::Float64
end

function Individual(genes::Array{Float64,1})::Individual
    return Individual(genes,0.0)
end

mutable struct Population
    individuals::Array{Individual,1}

    # Best and second best individual are obtained during the selection
    best::Union{Individual,Nothing}
    second_best::Union{Individual,Nothing}
end

function Population(individuals::Array{Individual,1})::Population
    return Population(individuals,Nothing(),Nothing())
end

mutable struct World
    generations::Array{Population,1}
    mutation_rate::Float64
    mutation_amount::Float64
    crossover_rate::Float64
end

function World(
    mutation_rate::Float64 = 0.0,
    mutation_amount::Float64 = 0.0,
    crossover_rate::Float64 = 0.0
    )::World
    return World([],mutation_rate,mutation_amount,crossover_rate)
end

# Select the best 2 individuals of a population
function selection!(population::Population)::Tuple{Int,Int}
    best_1::Int = 1  # index of best individual
    best_2::Int = 2  # index of second best individual
    for (idx,individual) in enumerate(population.individuals)
        if individual.fitness >= population.individuals[best_1].fitness
            best_2 = best_1
            best_1 = idx
        elseif individual.fitness >= population.individuals[best_2].fitness
            best_2 = idx
        end
    end
    population.best = population.individuals[best_1]
    population.second_best = population.individuals[best_2]
    #return population.best, population.second_best
    return best_1,best_2
end

# Do a crossover between the best 2 individuals of a population, produce a new
# individual (offspring)
function crossover(population::Population,crossover_rate::Float64)::Individual
    # New individual have 'crossover_rate'% chance of having a gene from the
    # best, 'crossover_rate-1'% chance of having it from the second best
    genes::Array{Float64,1} = []

    for i = 1:size(population.best.genes)[1]
        gene::Float64 = population.best.genes[i]
        random = rand(Float64)
        if random > crossover_rate
            gene = population.second_best.genes[i]
        end
        push!(genes,gene)
    end

    return Individual(genes)
end

# Create a mutated version of the individual
function mutation(individual::Individual, mutation_rate::Float64, mutation_amount::Float64)::Individual
    mutated_genes::Array{Float64,1} = []
    for gene in individual.genes
        mutated_gene::Float64 = gene
        ismuted = rand(Float64)
        if ismuted < mutation_rate
            mutated_gene += (rand(Float64)-0.5) * mutation_amount
        end
        push!(mutated_genes,mutated_gene)
    end
    return Individual(mutated_genes)
end

# Get the next generation, individual from the previous generation
# must have computed fitness
function next_generation!(world::World)::Population
    population = last(world.generations)
    selection!(population)
    individuals::Array{Individual,1} = []
    push!(individuals,population.best)  # First individual is best in previous generation
    offspring::Individual = crossover(population,world.crossover_rate)
    for i = 2:size(population.individuals)[1]
        push!(individuals,mutation(offspring,world.mutation_rate,world.mutation_amount))
    end

    push!(world.generations,Population(individuals))
    last(world.generations).individuals[1].fitness = 0.0  # Remove first individual score

    return last(world.generations)
end

end
