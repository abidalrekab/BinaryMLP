from binarization import binarization
import numpy as np

class DenseLayer(lasagne.layers.DenseLayer):

        def __init__(self, incoming, num_units,
                     binary=True, stochastic=True, H=1., W_LR_scale="Glorot", **kwargs):

                self.binary = binary
                self.stochastic = stochastic

                self.H = H
                if H == "Glorot":
                        num_inputs = int(np.prod(incoming.output_shape[1:]))
                        self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
                        # print("H = "+str(self.H))

                self.W_LR_scale = W_LR_scale
                if W_LR_scale == "Glorot":
                        num_inputs = int(np.prod(incoming.output_shape[1:]))
                        self.W_LR_scale = np.float32(1. / np.sqrt(1.5 / (num_inputs + num_units)))

                self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

                if self.binary:
                        super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H, self.H)),
                                                         **kwargs)
                        # add the binary tag to weights
                        self.params[self.W] = set(['binary'])

                else:
                        super(DenseLayer, self).__init__(incoming, num_units, **kwargs)

        def get_output_for(self, input, deterministic=False, **kwargs):

                self.Wb = binarization(self.W, self.H, self.binary, deterministic, self.stochastic, self._srng)
                Wr = self.W
                self.W = self.Wb

                rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)

                self.W = Wr

                return rvalue


