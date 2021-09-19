include("Net.jl")

net = Net.initialize([rand(30, 10)], rand(1, 1), "conv.yaml")
Net.propagate!(net.Layers[1][1], net.Layers[2])
Net.propagate!(net.Layers[2], net.Layers[3])
Net.propagate!(net.Layers[3], net.Layers[4])
net.Layers[4].δ .= net.Layers[4].a
Net.backpropagate!(net.Layers[3], net.Layers[4])
Net.backpropagate!(net.Layers[2], net.Layers[3])