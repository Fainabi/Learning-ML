module CNN

# Net struct
mutable struct CNNNet
    # dataset and labels
    X::Array
    Y::Array

    # hyper parameters
    FullConnectedNodes::Array

end


end