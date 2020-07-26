def binarization(W, H, binary=True, deterministic=False, stochastic=False, srng=None):
    # (deterministic == True) <-> test-time <-> inference-time
        if not binary or (deterministic and stochastic):
                # print("not binary")
                Wb = W

        else:

                # [-1,1] -> [0,1]
                Wb = hard_sigmoid(W / H)
                # Wb = T.clip(W/H,-1,1)

                # Stochastic BinaryConnect
                if stochastic:

                        # print("stoch")
                        Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

                # Deterministic BinaryConnect (round to nearest)
                else:
                        # print("det")
                        Wb = T.round(Wb)

                # 0 or 1 -> -1 or 1
                Wb = T.cast(T.switch(Wb, H, -H), theano.config.floatX)

        return Wb