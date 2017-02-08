import theano.tensor as T
import theano
import numpy as np

floatX = theano.config.floatX

def logsumexp(x, y):
    max = T.switch(x > y, x, y)
    min = T.switch(x > y, y, x)
    return T.log1p(T.exp(min - max)) + max

def dropout(dropout_rate, rng, parameter):
    mask = rng.binomial(parameter.shape, p=1.-dropout_rate, dtype=parameter.dtype)
    return parameter * mask / (1. - dropout_rate)

def simple_gradient_ascend(parameter, parameter_gradient, learning_rate=.1):
    return (parameter, parameter + learning_rate * parameter_gradient)

def clip(gradient, bound):
	"""
	will work for bound > 0
	"""
    assert bound > 0
    return T.clip(gradient, -bound, bound)

def rmsprop(parameter, parameter_gradient, learning_rate=.05, fudge_factor=1e-10, rho=.9, clip_threshold=1):
    clipped_gradient = T.clip(parameter_gradient, -clip_threshold, clip_threshold)
    rmsprob_moving_avg = theano.shared(np.ones(parameter.get_value().shape, dtype=floatX) * 0, "rmsprop_historical")
    next_rmsprop_avg = rho * rmsprob_moving_avg + (1. - rho) * T.pow(clipped_gradient, 2)
    update = rmsprob_moving_avg, next_rmsprop_avg
    grad_step = learning_rate / T.sqrt(fudge_factor + next_rmsprop_avg) * clipped_gradient
    parameter_update = parameter, parameter + grad_step
    ratio = grad_step.norm(2) / parameter.norm(2)
    return (update, parameter_update), ratio

def nesterov_rmsprop(parameter, parameter_gradient, learning_rate, momentum, fudge_factor=1e-10, rho=.9):
    memory = theano.shared(np.zeros_like(parameter.get_value(), dtype=floatX), name="nesterov_momentum")
    rmsprop_moving_avg = theano.shared(np.zeros(parameter.get_value().shape, dtype=floatX), "rmsprop_historical")
    next_rmsprop_avg = rho * rmsprop_moving_avg + (1. - rho) * T.pow(parameter_gradient, 2)
    memory_update = memory, momentum * memory + learning_rate / T.sqrt(fudge_factor + next_rmsprop_avg) * parameter_gradient
    """
    check if types of first and second memory_update object are same
    """
    assert str(memory_update[0].type).split('(')[-1] == str(memory_update[1].type).split('(')[-1]
    grad_step = - momentum * memory + (1. + momentum) * memory_update[1]
    parameter_update = parameter, parameter + grad_step
    ratio = grad_step.norm(2) / parameter.norm(2)
    return (memory_update, parameter_update, (rmsprop_moving_avg,  next_rmsprop_avg)), ratio
