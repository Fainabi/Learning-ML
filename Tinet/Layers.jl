include("Parameters.jl")
include("Utils.jl")
abstract type AbstractLayer end
abstract type AbstractSingleChannelLayer <: AbstractLayer end
abstract type AbstractMultiChannelLayer <: AbstractLayer end

mutable struct InputLayer <: AbstractLayer
    a::Array  # (batch_size, sizeof(X)...)
    δ::Array  # same shape with X if needed
end

mutable struct FullyConnectedLayer <: AbstractSingleChannelLayer
    a::Array  # value after activation
    z::Array  # value before activation
    δ::Array  # backprop buffer

    para::Parameters  # parameters to learn
    hyper::FullyConnectedLayerHyperparameters
end

mutable struct ConvolutionalLayer <: AbstractMultiChannelLayer
    a::Array
    z::Array
    δ::Array

    para::Parameters
    hyper::ConvolutionalLayerHyperparameters
end

mutable struct MaxPoolingLayer <: AbstractMultiChannelLayer
    a::Array
    δ::Array

    para::Parameters
    hyper::MaxPoolingLayerHyperparameters
end
function init_a(shape::Tuple)
    batch = shape[1]
    [zeros(shape[2:end]) for _ in 1:batch]
end
function newInputLayer(shape::Tuple, shapeδ=())
    InputLayer(zeros(shape), zeros(shapeδ))
end
function newFullyConnectedLayer(shape::Tuple, type)
    a = zeros(shape[3], shape[1])  # nodeNum, batch
    z = copy(a)
    δ = copy(a)
    para = newParameters(shape[2:end], shape[end])
    activation, deri_ac = fromType(type)
    hyper = FullyConnectedLayerHyperparameters(activation, deri_ac)
    FullyConnectedLayer(a, z, δ, para, hyper)
end
newConvolutionalLayer(shape::Tuple, para::Parameters, hyper::ConvolutionalLayerHyperparameters) =
    ConvolutionalLayer(zeros(shape), zeros(shape), zeros(shape), para, hyper)
newMaxPoolingLayer(shape::Tuple, para::Parameters, hyper::MaxPoolingLayerHyperparameters) =
    MaxPoolingLayer(zeros(shape), zeros(shape), para, hyper)

# =============== Propagate ==============
function propagate!(layer1::Union{AbstractSingleChannelLayer, InputLayer}, layer2::FullyConnectedLayer)
    layer2.z .= layer2.para.weight' * layer1.a .+ layer2.para.bias
    layer2.a .= layer2.hyper.activation.(layer2.z)
end

# =============== Loses =================



# =============== Backpropagate ==============
function backprapagate!(layer1::FullyConnectedLayer, layer2::FullyConnectedLayer)
    layer1.δ .= layer2.para.weight * layer2.δ .* layer2.hyper.deri_activation.(layer1.z)
end