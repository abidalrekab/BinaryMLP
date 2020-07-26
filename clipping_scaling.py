def clipping_scaling(updates, network):
    layers = lasagne.layers.get_all_layers(network)
        updates = OrderedDict(updates)

        for layer in layers:

                params = layer.get_params(binary=True)
                for param in params:
                        print("W_LR_scale = " + str(layer.W_LR_scale))
                        print("H = " + str(layer.H))
                        updates[param] = param + layer.W_LR_scale * (updates[param] - param)
                        updates[param] = T.clip(updates[param], -layer.H, layer.H)

        return updates