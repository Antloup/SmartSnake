include("./snake_game.jl")
include("./simple_mlp.jl")
include("./simple_ga.jl")

module SmartSnake

using ..SimpleGa
using ..SimpleMlp
using ..SnakeGame

# In this module, we need to cross the game, the MLP and the GA modules.

# For the GA, we need to determined what will be our population,
# its genes, and the fitness function.
# - Our population will be neural networks,
# - Individual genes will be neural networks weights and biases.
# - The fitness function will be the score of the game
# We hence need to cross the MLP with the GA. It will be done by creating
# conversion function from neural network to individual and vise-versa.
# Note : In a real case scenario, genes are manipulated directly from
# neural networks matrices.

# Set the fitness of an individual
function fitness!(individual::SimpleGa.Individual,game::SnakeGame.Game)::Float64
    individual.fitness += SnakeGame.game_score(game)
    return individual.fitness
end

# Converting a neural network to an individual
function Individual(nn::SimpleMlp.NeuralNetwork)::SimpleGa.Individual
    genes::Array{Float64,1} = []
    for layer in nn.layers[2:end]
        for neuron in layer.neurons
            push!(genes, neuron.bias)
            for synapse in neuron.synapses
                push!(genes, synapse[2])
            end
        end
    end
    return SimpleGa.Individual(genes)
end

# Converting a neural network array to a population
function get_population(nns::Array{SimpleMlp.NeuralNetwork,1})::SimpleGa.Population
    individuals::Array{SimpleGa.Individual,1} = []
    for nn in nns
        push!(individuals, Individual(nn))
    end
    return SimpleGa.Population(individuals)
end

# Converting a individual to a neural network
function NeuralNetwork(individual::SimpleGa.Individual,network_dim::Array{Int,1})::SimpleMlp.NeuralNetwork
    nn = SimpleMlp.NeuralNetwork(network_dim)
    idx::Int = 1
    for layer in nn.layers[2:end]
        for neuron in layer.neurons
            neuron.bias = individual.genes[idx]
            idx += 1
            for (s_idx,synapse) in enumerate(neuron.synapses)
                neuron.synapses[s_idx] = (synapse[1],individual.genes[idx])
                idx += 1
            end
        end
    end
    return nn
end

# Converting a population to a neural network array
function get_neural_networks(population::SimpleGa.Population,network_dim::Array{Int,1})::Array{SimpleMlp.NeuralNetwork,1}
    neural_networks::Array{SimpleMlp.NeuralNetwork,1} = []
    for individual in population.individuals
        push!(neural_networks,NeuralNetwork(individual,network_dim))
    end
    return neural_networks
end

# For the MLP, we need to determined our inputs and outputs
# Our inputs will be the following features:
# - All square in the game. Snake square will be 0, others will be 1
# - Snake head position (x and y)
# - Food position (x and y)
# - The distance between the food and the snake head (x and y)
# Our outputs will be all possible directions
# We hence need to extract features from the game so we can input them in the
# neural network.

function extract_features(game::SnakeGame.Game)::Array{Float64,1}
    features::Array{Float64,1} = []
    features = vcat(features, near_square_feature(game))
    #features = vcat(features, square_feature(game))
    #features = vcat(features, head_feature(game))
    #features = vcat(features, food_feature(game))
    features = vcat(features, distance_feature(game))
    return features
end

function near_square_feature(game::SnakeGame.Game)::Array{Float64,1}
    snake_head::SnakeGame.Position = last(game.snake.positions)
    near_square_position::Array{SnakeGame.Position,1} = []
    push!(near_square_position,snake_head + SnakeGame.Position(0,-1)) # Up
    push!(near_square_position,snake_head + SnakeGame.Position(0,1)) # Down
    push!(near_square_position,snake_head + SnakeGame.Position(-1,0)) # Left
    push!(near_square_position,snake_head + SnakeGame.Position(1,0)) # Right
    near_square::Array{Float64,1} = []
    for square in near_square_position
        if square.x < 1 || square.y < 1 || square.x > game.size || square.y > game.size || square in game.snake.positions
            push!(near_square,0.0)
        elseif square == game.food
            push!(near_square,1.0)
        else
            push!(near_square,0.6)
        end
    end
    return near_square
end

function square_feature(game::SnakeGame.Game)::Array{Float64,1}
    feature::Array{Float64,1} = []
    for y = 1: game.size
        for x = 1: game.size
            if SnakeGame.Position(x,y) in game.snake.positions
                push!(feature,1.0)
            else
                push!(feature,0.0)
            end
        end
    end
    return feature
end

function head_feature(game::SnakeGame.Game)::Array{Float64,1}
    feature::Array{Float64,1} = []
    push!(feature,last(game.snake.positions).x/game.size)
    push!(feature,last(game.snake.positions).y/game.size)
    return feature
end

function food_feature(game::SnakeGame.Game)::Array{Float64,1}
    feature::Array{Float64,1} = []
    push!(feature,game.food.x/game.size)
    push!(feature,game.food.y/game.size)
    return feature
end

function distance_feature(game::SnakeGame.Game)::Array{Float64,1}
    feature::Array{Float64,1} = []
    push!(feature,(game.food.x - last(game.snake.positions).x)/game.size)
    push!(feature,(game.food.y - last(game.snake.positions).y)/game.size)
    return feature
end

# Convert neural network output into game input
function play!(game::SnakeGame.Game,inputs::Array{Float64,1})::Int
    best::Int = 1
    for (idx,input) in enumerate(inputs)
        if input > inputs[best]
            best = idx
        end
    end
    dir::SnakeGame.Direction = SnakeGame.up
    if best == 2
        dir = SnakeGame.down
    elseif best == 3
        dir = SnakeGame.left
    elseif best == 4
        dir = SnakeGame.right
    end
    SnakeGame.next_frame!(game,dir)
    return best
end

end

using .SmartSnake

function main()

    # The program will articulate as follows:
    # - Init the neural networks and world first generation
    # - For each neural network, let it play until game over
    #   - Extract features, input of neural network is the size of the features
    #   - Evaluate the neural network
    #   - Get the output
    # - When game over, get its fitness
    # - Get the next generations
    # - Update neural networks
    # - Repeat until number of generations reached

    population_number = 100
    generation_number = 1000
    game_number = 5
    max_move_number = 50
    game_size = 5
    world = SimpleGa.World(0.1,0.1,0.7)
    input_size = size(SmartSnake.extract_features(SnakeGame.Game(game_size)))[1]
    output_size = 4
    nn_dim = [input_size,output_size]
    nns::Array{SimpleMlp.NeuralNetwork,1} = []

    # Init neural networks
    for i = 1:population_number
        push!(nns,SimpleMlp.NeuralNetwork(nn_dim))
    end

    # Init world and first generation
    push!(world.generations,SmartSnake.get_population(nns))

    for i = 1:generation_number
        world.mutation_amount *= 1
        # For each neural network
        for (idx,nn) in enumerate(nns)
            # Get its individual
            individual = last(world.generations).individuals[idx]

            for j = 1:game_number
                game = SnakeGame.Game(game_size)
                # Make it play
                while !game.over && game.num_move < max_move_number
                    input = SmartSnake.extract_features(game)
                    output = SimpleMlp.evaluate_neural_network!(nn,input)
                    #if i == generation_number && idx == 1 # Only print the game of the best on in the previous generation
                        #println("Snake ",idx," of generation:",i)
                        # SnakeGame.print(game)
                        # println("Input:",input)
                        # println("Output:",output)
                    #end
                    SmartSnake.play!(game,output)
                end
                # Update individual fitness
                SmartSnake.fitness!(individual,game)
            end

            if i == generation_number || idx == 1
                println("Gen:",i,", nn:",idx,", avg. score:",individual.fitness/game_number)
            end
        end

        # Creating new generation, update neural networks with next generation
        nns = SmartSnake.get_neural_networks(SimpleGa.next_generation!(world),nn_dim)
    end
end

main()
