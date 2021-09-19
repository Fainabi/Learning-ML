module Tinet

using LinearAlgebra, ProgressBars, CUDA, FFTW
import Random, YAML
include("Layers.jl") 

mutable struct NeuralNetwork
    layers::Array{AbstractLayer}
    hyper::ModelHyperparameters
end

function new_net(configFile)
    config = YAML.load_file(configFile)

    layers = []
    # input layer
    batchSize = get(config, "BatchSize", 1)
    input_size = config["InputSize"] |> parse_str
    input = newInputLayer((batchSize, input_size...))
    push!(layers, input)

    for layerConfig in config["Layer"]
        layer_type= get(layerConfig, "layerType", "Fully")
        activation = layerConfig["activation"]
        if layer_type == "Fully"  # fully connected layer
            if layers[end] isa AbstractMultiChannelLayer  # dense into single channel

            end
            nodeNum = size(layers[end].a)[end]  # number of nodes in lastest layer
            shape = (batchSize, nodeNum, layerConfig["nodeNum"])  # for weight w
            layer = newFullyConnectedLayer(shape, layerConfig["activation"])
            push!(layers, layer)
        end
    end
#     Type: "Convolutional"
# LayerNum: 2
# LearningRate: 1
# # Mode: "GPU"
# Epoches: 2
# BatchSize: 1
# InputSize: "(30,10)"
# InputDimension: 2
# Layer:
#   - layerType: Convolutional
#     activation: ReLU
#     kernelNum: 3
#     kernelSize: (3, 3)
#     stride: (1, 1) 
#   - layerType: MaxPooling
#     # activation: ReLU
#     # kernelNum: 3
#     kernelSize: (3, 3)
#     stride: (3, 3)
#   - layerType: Convolutional
#     kernelSize: (2, 2)
#     stride: (1, 1)
#     kernelNum: 2
#     activation: ReLU
#   # - activation: Linear
#   #   layerType: Normal  # normal or empty
#   #   nodeNum: 1
    learning_rate = config["LearningRate"]
    return NeuralNetwork(
        layers,
        ModelHyperparameters(learning_rate)
    )
end

function toDevice!(nn::NeuralNetwork, device)

end

end