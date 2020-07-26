from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

class Round3(UnaryScalarOp):
    def c_code(self, node, name, x, z, sub):
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,

round3 = Elemwise(Round3(same_out_nocomplex, name='round3'))
