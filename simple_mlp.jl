module SimpleMlp

# All of this can be done very quickly using matrices.
# This module use structures to (hopefully) make neural network easy to understand.
# The formal notations (of matrices and functions) have been noted on each
# variables and functions.
# Backpropagation won't be implemented since we will be using genetic algorithm (GA)
# to update weights and biases.
# Same for cost function, it will be replaced by fitness function in the GA.

mutable struct Neuron
    # The value of the neuron will be evaluated by the evaluation function
    # Values are stored in a matrix denoted 'A'
    value::Float64

    # Biases are stored in a matrix denoted 'b'
    bias::Float64

    # Neurons are connected to each neuron of the previous layer
    # with a weight linked them.
    # Weights are stored in a matrix denoted 'W'
    # In this example, the tuple (Neuron,Weight) will be called a synapse
    synapses::Array{Tuple{Neuron,Float64},1}
end

# A layer is a list of neurons
mutable struct NeuronLayer
    neurons::Array{Neuron,1}
end

# A multi-layer perceptron (MLP) is composed of several layers of different size.
mutable struct NeuralNetwork
    layers::Array{NeuronLayer,1}
end

# In the initialization process of a neuron,
# Neurons weights are randomly generated within a certain range (here [-1:1])
# Bias is set to 0
# Value is computed in the forward function
function Neuron(prev_layer::Union{NeuronLayer,Nothing})::Neuron
    bias::Float64 = 0.0
    value::Float64 = 0.0
    synapses::Array{Tuple{Neuron,Float64},1} = []
    if !(prev_layer isa Nothing)
        # We are linking the created neuron with neurons from the previous layout
        for neuron in prev_layer.neurons
            push!(synapses,(neuron,rand(Float64)*2-1))
            #push!(synapses,(neuron,0.0))
        end
    end
    return Neuron(value,bias,synapses)
end

# This is the activation function, denoted 'g'
# It is used to squash the result of Z within a certain range (here [0:1])
# We will use sigmoid function
function activation_function(x::Float64)::Float64
    return 1.0 / (1.0 + exp(-x))  # Sigmoid
    #return max(0,x)
end

# This is this evaluation function denoted g(Z)
# With Z = W*X+b
# For each neuron, its new value (A) is the activation function (g) of
# the weighted (W) sum of previous layer values (X) + the bias (b)
# Hence the formula : A = g(Z)
function evaluate_neuron!(neuron::Neuron)::Float64
    # Computing Z
    value::Float64 = neuron.bias
    for synapse in neuron.synapses
        value += synapse[2] * synapse[1].value
    end

    # Computing g(Z)
    neuron.value = activation_function(value)
    return neuron.value
end

function NeuronLayer(
    prev_layer::Union{NeuronLayer,Nothing},
    current_layer_size::Int
    )::NeuronLayer
    neurons::Array{Neuron,1} = []
    for i = 1:current_layer_size
        push!(neurons,Neuron(prev_layer))
    end
    return NeuronLayer(neurons)
end

function NeuralNetwork(network_dim::Array{Int,1})::NeuralNetwork
    layers::Array{NeuronLayer,1} = []
    push!(layers,NeuronLayer(Nothing(),network_dim[1]))
    for i = 2:size(network_dim)[1]
        push!(layers,NeuronLayer(last(layers),network_dim[i]))
    end
    return NeuralNetwork(layers)
end

# The first layer of the MLP is our input data
function set_input!(neural_network::NeuralNetwork,input::Array{Float64,1})
    for (idx,neuron) in enumerate(neural_network.layers[1].neurons)
        neuron.value = input[idx]
    end
end

# This is where all neurons values (A) of the network will be evaluated
# We are evaluating layers from the second one (the first on being the input)
# to the last one (hence the name 'forward' propagation) since each neurons
# need the value of the previous layer neurons in order to be evaluated
function forward_prop!(neural_network::NeuralNetwork)
    for layer in neural_network.layers[2:end]
        for neuron in layer.neurons
            evaluate_neuron!(neuron)
        end
    end
end

# The last layer of the MLP is our output data
function get_output(neural_network::NeuralNetwork)::Array{Float64,1}
    output::Array{Float64,1} = []
    for neuron in last(neural_network.layers).neurons
        push!(output,neuron.value)
    end
    return output
end

function evaluate_neural_network!(
    neural_network::NeuralNetwork,
    input::Union{Array{Float64,1},Nothing} = Nothing()
    )::Array{Float64,1}

    if !(input isa Nothing)
        set_input!(neural_network,input)
    end
    forward_prop!(neural_network)
    return get_output(neural_network)
end

function print(neural_network::NeuralNetwork, print_synapses::Bool = false)
    for (i,layer) in enumerate(neural_network.layers)
        println("Layer ",i, ":")
        print(layer, print_synapses)
    end
end

function print(layer::NeuronLayer, print_synapses::Bool = false)
    for (j,neuron) in enumerate(layer.neurons)
        println("Neuron ",j, ":")
        print(neuron, print_synapses)
    end
end

function print(neuron::Neuron, print_synapses::Bool = false)
    println("Value ", neuron.value,", bias ", neuron.bias)
    if print_synapses
        for (k,synapse) in enumerate(neuron.synapses)
            println("Synapse ",k,", weight ", synapse[2])
        end
    end
end

end
