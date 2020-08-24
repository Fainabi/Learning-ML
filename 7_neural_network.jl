module NeuralNetwork

import Random, CUDA
using ProgressBars, LinearAlgebra
export backpropagation, ActivationFunction
CUDA.allowscalar(false)


"""
    ActivationFunction

ActivationFunction contains several functions that used in neural network. 
"""
@enum ActivationFunction begin
    Linear
    Logistic
    ReLU
    Tanh
end

logistic(x) = 1 / (1 + exp(-x))
relu(x) = x>0 ? x : 0

deri_logistic(x) = begin
    log_val = logistic(x)
    log_val * (1 - log_val)
end
deri_relu(x) = x>0 ? 1 : 0
deri_tanh(x) = 1 - tanh(x)^2

acfun = Dict(
    Linear::ActivationFunction => identity,
    Logistic::ActivationFunction => logistic,
    ReLU::ActivationFunction => relu,
    Tanh::ActivationFunction => tanh,
)

deri_acfun = Dict(
    Linear::ActivationFunction => (x) -> 1,
    Logistic::ActivationFunction => deri_logistic,
    ReLU::ActivationFunction => deri_relu,
    Tanh::ActivationFunction => deri_tanh,
)



"""
    Net

The neural network.
"""
mutable struct Net
    # training set and parameters
    X::Array
    Y::Array

    # hyper parameters
    Nodes::Array
    Activations::Array
    α::AbstractFloat
    max_iter::Number
    loss::Function
    deri_loss::Function

    # model
    model::Function
    mode::String
end

function initialize_net(X=[], Y=[]; kwargs...)
    net = Net(X, Y, [], [], .01, 1e3, identity, identity, identity, "CPU")
    validate_set = (:Nodes, :Activations, :α, :max_iter, :loss, :deri_loss, :mode)

    for kw in kwargs
        if kw.first in validate_set
            :($(net).$(kw.first) = $(kw.second)) |> eval
        end
    end
    return net
end

function train(net::Net)
    w, b = backpropagation(
        net.X, net.Y,
        loss=net.loss,
        deri_loss=net.deri_loss,
        activations=net.Activations,
        nodes=net.Nodes,
        α=net.α,
        max_iter=net.max_iter,
        mode=net.mode,
    )
    model(x) = begin
        for layer = 1:length(net.Nodes)
            x = w[layer]' * x .+ b[layer] .|> acfun[net.Activations[layer]]
        end
        return x
    end
    net.model = model;
end

"""
    backpropagation(X, Y; <keyword arguments>, max_iter=1e3, α=0.1)

Backpropagation NN. The accessible activation functions are `logistic`, `ReLU` and `tanh` which could

be inspected by NeuralNetwork.ActivationFunction.

...
# Arguments
- `X::Array`: samples
- `Y::Array`: labels
- `loss::Function`: loss function, this function just in normal vector-by-vector form
- `deri_loss::Function`: derivative of loss function, this function ___must___ be in matrix form
- `activations::Array{ActivationFunction}`: arrays that contains activation function types for every layer. 
- `nodes::Array`: number of nodes in every layer
- `max_iter::Number`: maximum time to iterate, default is 1e3
- `α::AbstractFloat`: learning rate, default is 0.001
...

"""
function backpropagation(X, Y; loss, deri_loss, activations, nodes, α=0.001, max_iter=1e3, ε=1e-3, mode="CPU")
    validate_inputs(X, Y)
    if length(activations) != length(nodes)
        throw(ArgumentError("layer number not match"))
    end
    
    Batch_size = 1000
    M = size(X, 2)
    dim_feature = size(X, 1)
    dim_label = size(Y, 1)

    δ = []
    z = []
    w = []
    b = []
    a = []
    layer_num = length(nodes)
    batch_samples = []
    batch_labels = []
    batch_group_number = Int64(ceil(M/Batch_size))
    batch_size_number = []
    deri_activations = map(act -> deri_acfun[act], activations)
    activations = map(act -> acfun[act], activations)

    println("number of layers is $layer_num")
    println("size of training data is $M")
    println("scale of NN is $dim_feature × $nodes")
    """ 
        Initialize hyper-parameters in memories.
    With initialization, we could use '.=' and avoid
    massive assignment and gc.
    """
    for layer = 1:layer_num
        if layer == 1
            push!(w, Random.randn(dim_feature, nodes[1]))
        else
            push!(w, Random.randn(nodes[layer-1], nodes[layer]))
        end
        push!(b, zeros(nodes[layer]))
        push!(z, zeros(nodes[layer], Batch_size))
        push!(δ, zeros(nodes[layer], Batch_size))
        push!(a, zeros(nodes[layer], Batch_size))
    end
    for batch = 1:batch_group_number
        if batch < batch_group_number
            push!(batch_samples, X[:, (batch-1)*Batch_size+1:batch*Batch_size])
            push!(batch_labels, Y[:, (batch-1)*Batch_size+1:batch*Batch_size])
            push!(batch_size_number, Batch_size)
        else
            push!(batch_samples, X[:, (batch-1)*Batch_size+1:end])
            push!(batch_labels, Y[:, (batch-1)*Batch_size+1:end])
            push!(batch_size_number, M - Batch_size*(batch-1))
        end
    end
    
    # train
    for num_iter = tqdm(1:max_iter)
        for batch = 1:batch_group_number
            batch_m = batch_size_number[batch]
            """
                Forward propagation
            Some of '=' are reference assigning, so we donnot need
            to use the '.='. Some of '.=' need submatrix because of
            different size in every batch. a[1] is input samples.
            a[end] is the last second layer output. Because it is
            z^{[n]} but not a^{[n]} sent to the backpropagation,
            we do this trick on index.
            """
            z[1][:, 1:batch_m] .= w[1]' * batch_samples[batch] .+ b[1]
            for layer = 2:layer_num
                a[layer-1] .= z[layer-1] .|> activations[layer]  # map activation function
                z[layer] .= w[layer]' * a[layer-1] .+ b[layer]  # linear combination
            end
            a[end] .= z[end] .|> activations[end]

            """
                Backward propagation

            """
            δ[end][:, 1:batch_m] .= deri_loss(a[end][:, 1:batch_m], batch_labels[batch])
            for layer = layer_num-1:-1:1
                δ[layer] .= w[layer+1] * δ[layer+1] .* broadcast(deri_activations[layer], z[layer])
            end
            for layer = 2:layer_num
                δ_layer = @view δ[layer][:, 1:batch_m]  # just a reference
                w[layer] .-= a[layer-1][:, 1:batch_m] *  # update w and b
                                transpose(δ_layer) * α / batch_m
                b[layer] -= sum(δ_layer, dims=2) * α / batch_m
            end
            w[1] .-= batch_samples[batch][:, 1:batch_m] * 
                        transpose(δ[1][:, 1:batch_m]) * α / batch_m
            b[1] -= sum(δ[1][:, 1:batch_m], dims=2) * α / batch_m
        end
    end

    return w, b
end

# inner functions
function validate_inputs(X, Y)
    if X === [] || Y === [] 
        throw(ArgumentError("input matrix should not be empty"))
    elseif size(X, 2) != size(Y, 2)
        throw(DimensionMismatch("input samples and labels should have same number of rows")) 
    end
end

end