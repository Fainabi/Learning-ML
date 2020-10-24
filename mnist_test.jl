include("Net.jl")

images = open("data/mnist/train-images-idx3-ubyte", "r") do f
    if hton(read(f, UInt32)) != 2051  # magic number
        println("image file format is not legal")
        return
    end
    training_num = read(f, UInt32) |> hton  # big endian
    row_size = read(f, UInt32) |> hton
    column_size = read(f, UInt32) |> hton
    sample_size = row_size * column_size  # N features
    println("Start reading images. Read $(row_size)×$(column_size) data $(training_num) times.")
    # fill data
    images = fill(Float64(0.0), (sample_size, training_num))  # [x1, x2, ..., xM]
    for idx = 1:training_num
        images[:, idx] = read(f, sample_size) |> Array{Float64}
    end
    return images
end

labels = open("data/mnist/train-labels-idx1-ubyte", "r") do f
    if hton(read(f, UInt32)) != 2049  # magic number
        println("label file format is not legal")
        return
    end
    labels_num = read(f, UInt32) |> hton  # big endian
    println("Start reading labels. Read $labels_num data.")
    # fill labels
    labels = fill(Float64(0.0), (10, labels_num))
    idx = 1
    for label in read(f, labels_num)
        labels[label+1, idx] = 1
        idx += 1
    end
    return labels
end

images = images / 255


nodes = [100 50 10]
activations = fill(Net.ReLU, (1, 3))

# net = Net.initialize_net(images, labels; nodes=nodes, activations=activations, max_iter=100, α=1, mode="GPU")
net = Net.initialize(images, labels, "mnist_test.yaml")
Net.train!(net)

Net.get_loss(net)
