import hard_sigmoid
import round3

def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.