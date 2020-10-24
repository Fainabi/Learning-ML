module Net

using LinearAlgebra, ProgressBars, CUDA, FFTW
import Random, YAML

@enum ActivationFunction begin
    Linear
    Logistic
    ReLU
    Tanh
end

logistic(x) = 1 / (1 + exp(-x))
relu(x) = max(x, 0)

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

@enum LayerType begin
    Normal
    Convolutional
    MaxPooling
end

"""
    Structure Definition
"""
# Node struct
mutable struct LayerNodes  # all hyper parameters
    Activation::ActivationFunction
    Connection::Array  # nodes that connected at the former layer
    Weight
    Bias
end


# Layer struct
abstract type AbstractLayer end
abstract type AbstractSingleChannelLayer <: AbstractLayer end
abstract type AbstractMultipleChannelLayer <: AbstractLayer end
mutable struct InputLayer <: AbstractSingleChannelLayer
    NodeNum::Union{Integer,Tuple}
    # Nodes::LayerNodes
    a
end
mutable struct FullyConnectedLayer <: AbstractSingleChannelLayer
    NodeNum::Integer
    Nodes::LayerNodes
    δ

    z
    a
end
mutable struct ConvolutionalLayer <: AbstractMultipleChannelLayer
    NodeNum::Tuple
    Nodes::LayerNodes
    δ

    z
    a
    Kernel::Array{Any}
    KernelSize
    Stride::Tuple
end
mutable struct MaxPoolingLayer <: AbstractMultipleChannelLayer
    NodeNum::Tuple
    # Nodes::LayerNodes
    δ

    a
    Coordinate
    KernelSize
    Stride::Tuple
end

function propagate!(layer1::AbstractSingleChannelLayer, layer2::FullyConnectedLayer)
    # println("z: $(size(layer2.z)), w: $(size(layer2.Nodes.Weight)), a: $(size(layer1.a))")
    layer2.z .= layer2.Nodes.Weight' * layer1.a .+ layer2.Nodes.Bias
    layer2.a .= layer2.z .|> acfun[layer2.Nodes.Activation];
end
function propagate!(layer1::AbstractMultipleChannelLayer, layer2::InputLayer)
    """
        Get dense into fully connected layer.
    """
    # for sampleIdx = 1:size(layer1.a, 4)
    #     layer2.a[:, sampleIdx] .= reshape(layer1.a[:,:,:,sampleIdx], layer2.NodeNum, 1)
    # end
    layer2.a .= reshape(layer1.a, layer2.NodeNum, length(layer1.a)÷layer2.NodeNum);
end
function propagate!(layer1::AbstractMultipleChannelLayer, layer2::ConvolutionalLayer)
    startX = layer2.KernelSize[1]
    startY = layer2.KernelSize[2]
    kernel_num = length(layer2.Kernel)
    println(size(layer1.a, 4), size(layer1.a, 3), kernel_num)
    for sampleIdx = 1:size(layer1.a, 4)
        for channel = 1:size(layer1.a, 3), (idx,kernel) in enumerate(layer2.Kernel)
            layer2.z[:,:,(channel-1)*kernel_num+idx,sampleIdx] .= 
                convolve(layer1.a[:,:,channel,sampleIdx], kernel)[
                    startX:layer2.Stride[1]:end, startY:layer2.Stride[2]:end]
        end
    end
    layer2.a .= layer2.z .|> acfun[layer2.Nodes.Activation];
end
function propagate!(layer1::AbstractSingleChannelLayer, layer2::ConvolutionalLayer)
    startX = layer2.KernelSize[1]
    startY = layer2.KernelSize[2]
    for sampleIdx = 1:length(layer1.a)
        for (idx, kernel) in enumerate(layer2.Kernel)
            layer2.z[:, :, idx, sampleIdx] .= convolve(layer1.a[sampleIdx], kernel)[
                startX:layer2.Stride[1]:end, startY:layer2.Stride[2]:end]
        end
    end
    layer2.a .= layer2.z .|> acfun[layer2.Nodes.Activation];
end
# We don't need single channel layer to pooling layer
function propagate!(layer1::AbstractMultipleChannelLayer, layer2::MaxPoolingLayer)
    for sampleIdx = 1:size(layer2.a, 4)
        xrange, yrange = size(layer2.a)
        for aIdx = 1:size(layer2.a, 3)
            for xIdx = 1:xrange, yIdx = 1:yrange
                layer2.a[xIdx, yIdx, aIdx, sampleIdx], layer2.Coordinate[xIdx, yIdx, aIdx, sampleIdx] = 
                    findmax(layer1.a[
                        1+layer2.Stride[1]*(xIdx-1):min(layer2.KernelSize[1]+layer2.Stride[1]*(xIdx-1), end),
                        1+layer2.Stride[2]*(yIdx-1):min(layer2.KernelSize[2]+layer2.Stride[2]*(yIdx-1), end),
                        aIdx,
                        sampleIdx
                    ])
            end
        end
    end
end

function backpropagate!(layer1::FullyConnectedLayer, layer2::FullyConnectedLayer)
    layer1.δ .= layer2.Nodes.Weight * layer2.δ .* broadcast(deri_acfun[layer1.Nodes.Activation], layer1.z);
end

function update!(layer1::AbstractSingleChannelLayer, layer2::FullyConnectedLayer, α, batch_m)
    layer2.Nodes.Weight .-= layer1.a * transpose(layer2.δ) * α / batch_m
    layer2.Nodes.Bias .-= sum(layer2.δ, dims=2) * α / batch_m;
end

function convolve(A, K)
    # Size must match that size(A) == size(K)
    # this work should be done at initialization for saving time of GC
    ifft(fft(A).*fft(K)) .|> real
end

# Net struct
abstract type AbstractNet end
mutable struct FullyConnectedNet <: AbstractNet
    dataset
    label

    Layers::Array  # [X, L1, L2, ..., Ln]
    α::AbstractFloat  # Learning Rate
    Max_iter::Integer
    Batch_size::Integer

    Loss::Function
    Deri_loss::Function
end
mutable struct ConvolutionalNet <: AbstractNet
    dataset
    label

    Layers::Array  # [X, C1, P1, C2, P2, ..., Y, L1, L2, ..., Ln]
    α::AbstractFloat
    Max_iter::Integer
    Batch_size::Integer

    Loss::Function
    Deri_loss::Function
end

abstract type AbstractNetConfig end
mutable struct GenericNetConfig <: AbstractNetConfig
    nodes
    layertype
    activations
    α
    max_iter
    batch_size

    loss
    deri_loss

    # CNN
    kernel
    kernelsize
    stride
    mode
end
mutable struct FullyConnedtedNetConfig <: AbstractNetConfig
    nodes
    activations
    α
    max_iter
    batch_size

    loss
    deri_loss
    mode
end

mutable struct ConvolutionalNetConfig <: AbstractNetConfig
    nodes
    layertype
    activations
    α
    max_iter
    batch_size

    loss
    deri_loss
    mode
end

function initialize(X=[], Y=[], configFile=missing)
    config = YAML.load_file(configFile)

    layerConfig = config["Layer"]
    batch_size = get(config, "BatchSize", 1000)
    mode = get(config, "Mode", "CPU")
    """
        Set config on layers' z, a, activation
    """
    inputLayer = if get(config, "InputDimension", 1) == 1
        map(1:Int(ceil(size(X, 2)/batch_size))) do batch_idx
            if batch_idx < Int(ceil(size(X, 2)/batch_size))
                batch = batch_size*(batch_idx-1)+1:batch_size*batch_idx
            else
                batch = batch_size*(batch_idx-1)+1:size(X, 2)
            end
            a = X[:, batch]

            if get(config, "Mode", "CPU") == "GPU"
                a = cu(a)
            end

            # FullyConnectedLayer(nodes[idx], layernode, δ, z, a)
            InputLayer(size(X, 1), a)
        end
    else
        map(1:Int(ceil(length(X)/batch_size))) do batch_idx
            if batch_idx < Int(ceil(length(X)/batch_size))
                batch = batch_size*(batch_idx-1)+1:batch_size*batch_idx
            else
                batch = batch_size*(batch_idx-1)+1:length(X)
            end
            a = X[batch]

            if get(config, "Mode", "CPU") == "GPU"
                a = cu(a)
            end

            InputLayer(config["InputSize"] |> eval ∘ Meta.parse, a)
        end
    end
    layers = []
    for (idx, layer) in enumerate(layerConfig)
        type = get(layer, "layerType", "Normal")
        pLayerNodes = (idx==1) ? inputLayer[1].NodeNum : layerConfig[idx-1]["nodeNum"]
        
        if type == "Convolutional"
            channelNum = (idx==1) ? 1 : size(layers[end].a, 3)  # this convolutional layer only set at first
            kernel_num = layer["kernelNum"]
            kernel_size = layer["kernelSize"] |> eval ∘ Meta.parse
            stride = layer["stride"] |> eval ∘ Meta.parse
            kernels = map(1:kernel_num) do _
                if typeof(pLayerNodes) <: String
                    pLayerNodes = pLayerNodes |> eval ∘ Meta.parse
                end
                k = zeros(pLayerNodes)
                # println(pLayerNodes)
                k[1:kernel_size[1],1:kernel_size[2]] .= Random.randn(kernel_size...)
                k
            end
            nodeNum = (pLayerNodes .- kernel_size .+ 1) ./ stride .|> Int ∘ ceil
            z = zeros(nodeNum..., kernel_num*channelNum, batch_size)
            a = copy(z)
            δ = copy(z)
            bias = zeros(nodeNum..., kernel_num*channelNum)
            layerConfig[idx]["nodeNum"] = string(nodeNum)

            if get(config, "Mode", "CPU") == "GPU"
                z = cu(z)
                a = cu(a)
                δ = cu(δ)
                bias = cu(bias)
            end

            rlayer = ConvolutionalLayer(
                nodeNum,
                LayerNodes(
                    layer["activation"] |> eval ∘ Meta.parse,
                    [],
                    [],
                    bias
                ),
                δ,
                z,
                a,
                kernels,
                kernel_size,
                stride
            )
            
        elseif type == "MaxPooling"
            if typeof(pLayerNodes) <: String
                pLayerNodes = pLayerNodes |> eval ∘ Meta.parse
            end
            channelNum = (idx==1) ? 1 : size(layers[end].a, 3)
            kernel_size = layer["kernelSize"] |> eval ∘ Meta.parse
            stride = layer["stride"] |> eval ∘ Meta.parse
            nodeNum = (pLayerNodes .- kernel_size .+ 1) ./ stride .|> Int ∘ ceil
            a = zeros(nodeNum..., channelNum, batch_size)
            δ = copy(a)
            coord = Array{Any}(undef, size(a)...)
            layerConfig[idx]["nodeNum"] = string(nodeNum)

            if mode == "GPU"
                a = cu(a)
                δ = cu(δ)
            end

            rlayer = MaxPoolingLayer(
                nodeNum,
                δ,
                a,
                coord,
                kernel_size,
                stride,
            )
        else
            if typeof(pLayerNodes) <: String
                layerNodeNum = *((Meta.parse(pLayerNodes) |> eval)..., size(layers[end].a, 3))  # channelNum
                a = zeros(layerNodeNum, batch_size)
                if get(config, "Mode", "CPU") == "GPU"
                    a = cu(a)
                end
                rlayer = InputLayer(
                    layerNodeNum,
                    a
                )
                push!(layers, rlayer)
                pLayerNodes = layerNodeNum
            end
            layerNodeNum = layer["nodeNum"]
            z = zeros(layerNodeNum, batch_size)
            a = copy(z)
            δ = copy(z)
            bias = zeros(layerNodeNum, 1)
            weight = Random.randn(pLayerNodes, layerNodeNum)

            if get(config, "Mode", "CPU") == "GPU"
                z = cu(z)
                a = cu(a)
                δ = cu(δ)
                bias = cu(bias)
                weight = cu(weight)
            end

            rlayer = FullyConnectedLayer(
                layer["nodeNum"],
                LayerNodes(
                    layer["activation"] |> eval ∘ Meta.parse, 
                    [], 
                    weight, 
                    bias
                ),
                δ,
                z,
                a
            )
        end

        push!(layers, rlayer)
    end
    
    layers = [inputLayer, layers...]
    net_type = config["Type"]
    net = if net_type == "Convolutional"
        ConvolutionalNet(
            X,
            Y,
            layers,
            get(config, "LearningRate", .01),
            get(config, "Epoches", 100),
            get(config, "BatchSize", 1000),
            get(config, "Loss", "(y_hat, y) -> (y_hat - y |> v -> dot(v, v)/2)") |> eval ∘ Meta.parse,
            get(config, "DerivativeLoss", "(y_hat, y) -> y_hat - y") |> eval ∘ Meta.parse
        )
    elseif net_type == "Recursive"

    else  # FullyConnected
        FullyConnectedNet(
            X, 
            Y, 
            layers,
            get(config, "LearningRate", .01),
            get(config, "Epoches", 100),
            get(config, "BatchSize", 1000),
            get(config, "Loss", "(y_hat, y) -> (y_hat - y |> v -> dot(v, v)/2)") |> eval ∘ Meta.parse,
            get(config, "DerivativeLoss", "(y_hat, y) -> y_hat - y") |> eval ∘ Meta.parse
        )
    end

    if mode == "GPU"
        net.label = cu(net.label)
        net.dataset = cu(net.dataset)
    end

    return net
end



function train_once!(net::FullyConnectedNet)
    layer_len = length(net.Layers)
    batch_size = net.Batch_size
    if layer_len > 1
        @views for (batch_idx, layer) in enumerate(net.Layers[1])
            propagate!(layer, net.Layers[2])
            for idx = 2:layer_len-1
                propagate!(net.Layers[idx], net.Layers[idx+1])
            end

            if batch_idx < Int(ceil(size(net.label, 2)/batch_size))
                batch = batch_size*(batch_idx-1)+1:batch_size*batch_idx
            else
                batch = batch_size*(batch_idx-1)+1:size(net.label, 2)
            end

            net.Layers[end].δ = net.Deri_loss(net.Layers[end].a, net.label[:, batch])
            for idx = layer_len-1:-1:2
                backpropagate!(net.Layers[idx], net.Layers[idx+1])
            end
            for idx = layer_len:-1:3
                update!(net.Layers[idx-1], net.Layers[idx], net.α, net.Batch_size)
            end
            update!(layer, net.Layers[2], net.α, net.Batch_size)

        end
    end

    # println("Loss: $(get_loss(net))")
end

function train!(net::FullyConnectedNet)
    for epoch = tqdm(1:net.Max_iter)
        train_once!(net)
    end
end

function get_loss(net::FullyConnectedNet)
    val = net.dataset
    for layer in net.Layers[2:end]
        val = layer.Nodes.Weight' * val .+ layer.Nodes.Bias .|> acfun[layer.Nodes.Activation]
    end
    net.Loss(val, net.label) / size(net.label, 2)
end

function report(net::FullyConnectedNet)
    """
    This is a fully connected net.
    In this net, layer's nodes are
        $([layer.NodeNum for layer in net.Layers])
    with activation functions
        $([layer.Nodes.Activation for layer in net.Layers[2:end]])
    Learing rate is $(net.α)
    Batch size is $(net.Batch_size)
    Max epoch is $(net.Max_iter)
    Loss function is $(net.Loss)
    Derivative of loss function is $(net.Deri_loss)
    """ |> println
end
end