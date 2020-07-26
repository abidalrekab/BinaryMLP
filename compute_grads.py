def compute_grads(loss, network):
    layers = lasagne.layers.get_all_layers(network)
        grads = []

        for layer in layers:

                params = layer.get_params(binary=True)
                if params:
                        # print(params[0].name)
                        grads.append(theano.grad(loss, wrt=layer.Wb))

        return grads