import theano.tensor as T

def logsumexp(x, y):
    max = T.switch(x > y, x, y)
    min = T.switch(x > y, y, x)
    return T.log1p(T.exp(min - max)) + max

def dropout(dropout_rate, rng, parameter):
    mask = rng.binomial(parameter.shape, p=1.-dropout_rate, dtype=parameter.dtype)
    return parameter * mask / (1. - dropout_rate)

def simple_gradient_ascend(parameter, parameter_gradient, learning_rate=.1):
    return (parameter, parameter + learning_rate * parameter_gradient)