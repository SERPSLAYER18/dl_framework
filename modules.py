import numpy as np


class Module(object):
    """
    The module should be able to perform a backward pass: to differentiate the `forward` function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.
        gradInput = module.backward(input, gradOutput)
    """

    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.
        """

        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.
        """

        pass

    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"


# # Sequential container


class Sequential(Module):
    """
    This class implements a container, which processes `input` data sequentially.

    `input` is processed by each module (layer) in self.modules consecutively.
    The resulting array is called `output`.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})
        """

        self.output = input.copy()
        for module in self.modules:
            self.output = module.forward(self.output)

        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)

        """

        for i in reversed(range(1, len(self.modules))):
            module_input = self.modules[i - 1].output
            gradOutput = self.modules[i].backward(module_input, gradOutput)

        self.gradInput = self.modules[0].backward(input, gradOutput)

        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]

    def setParameters(self, parameters):
        pass

    def getGradParameters(self):
        """
        Gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + "\n" for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


# # Layers

# ## 1. Linear transform layer
# Also known as dense layer, fully-connected layer, FC-layer, InnerProductLayer (in caffe), affine transform
# - input:   **`batch_size x n_feats1`**
# - output: **`batch_size x n_feats2`**


class Linear(Module):
    """
    A module which applies a linear transformation
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        stdv = 1.0 / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        dot = np.dot(input, self.W.T)
        self.output = dot + self.b
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradW = np.matmul(gradOutput.T, input)
        self.gradb = gradOutput.sum(axis=0)
        pass

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = "Linear %d -> %d" % (s[1], s[0])
        return q


# ## 2. SoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
#
# $\text{softmax}(x)_i = \frac{\exp x_i} {\sum_j \exp x_j}$
#
# Recall that $\text{softmax}(x) == \text{softmax}(x - \text{const})$. It makes possible to avoid computing exp() from large argument.


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        np.exp(self.output, out=self.output)
        np.divide(self.output, self.output.sum(axis=1)[:, None], out=self.output)
        return self.output

    @staticmethod
    def jacobian(x):
        """
        Jacobian of 1-D input vector x
        """
        x = np.exp(x)
        x = x / (x.sum())
        return np.diag(x.flatten()) - np.outer(x, x)

    def updateGradInput(self, input, gradOutput):
        normalized = np.subtract(input, input.max(axis=1, keepdims=True))
        jacobians = np.apply_along_axis(SoftMax.jacobian, axis=1, arr=normalized)
        self.gradInput = np.asarray(
            [gradOutput[i] @ jacobians[i] for i in range(gradOutput.shape[0])]
        )

        return self.gradInput

    def __repr__(self):
        return "SoftMax"


# ## 3. LogSoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
#
# $\text{logsoftmax}(x)_i = \log\text{softmax}(x)_i = x_i - \log {\sum_j \exp x_j}$
#
# The main goal of this layer is to be used in computation of log-likelihood loss.


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        exps_sum = np.exp(self.output).sum(axis=1)
        self.output = self.output - np.log(exps_sum)[:, None]

        return self.output

    @staticmethod
    def jacobian(x):
        x = np.exp(x)
        x = x / (x.sum())
        return np.eye(x.shape[0]) - x[None, :]

    def updateGradInput(self, input, gradOutput):

        normalized = np.subtract(input, input.max(axis=1, keepdims=True))
        jacobians = np.apply_along_axis(LogSoftMax.jacobian, axis=1, arr=normalized)
        self.gradInput = np.asarray(
            [gradOutput[i] @ jacobians[i] for i in range(gradOutput.shape[0])]
        )
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"


# ## 4. Batch normalization
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.0):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        input = input.astype(np.float32)
        if self.training:
            batch_mean = input.mean(axis=0, dtype=np.float32)
            batch_variance = input.var(axis=0, dtype=np.float32)
        else:
            batch_mean = self.moving_mean
            batch_variance = self.moving_variance

        self.output = np.zeros_like(input)
        buf_1d = np.zeros_like(batch_variance)

        np.subtract(input, batch_mean, out=self.output)
        np.add(BatchNormalization.EPS, batch_variance, out=buf_1d)
        np.sqrt(buf_1d, out=buf_1d)
        np.divide(self.output, buf_1d, out=self.output)

        if self.training:
            if self.moving_mean is None:
                self.moving_mean = batch_mean
            if self.moving_variance is None:
                self.moving_variance = batch_variance

            self.moving_mean = self.moving_mean * self.alpha + batch_mean * (
                1 - self.alpha
            )
            self.moving_variance = (
                self.moving_variance * self.alpha + batch_variance * (1 - self.alpha)
            )

        return self.output

    def updateGradInput(self, input, gradOutput):

        batch = input.astype(np.float32)
        grads = gradOutput.astype(np.float32)
        n = np.float32(gradOutput.shape[0])
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        sqrt_var = np.sqrt(np.add(batch_var, BatchNormalization.EPS, dtype=np.float32))
        self.gradInput = np.divide(
            np.subtract(
                np.subtract(np.multiply(n, grads), grads.sum(axis=0)),
                np.multiply(
                    np.divide(
                        np.subtract(batch, batch_mean),
                        np.add(batch_var, BatchNormalization.EPS),
                    ),
                    np.multiply(grads, np.subtract(batch, batch_mean)).sum(axis=0),
                ),
            ),
            np.multiply(n, sqrt_var),
        )
        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"


class ChannelwiseScaling(Module):
    """
    Implements linear transform of input y = \gamma * x + \beta
    where \gamma, \beta - learnable vectors of length x.shape[-1]
    """

    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1.0 / np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput * input, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"


# ## 5. Dropout
# This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons.
#
# While training (`self.training == True`) it should sample a mask on each iteration (for every batch), zero out elements and multiply elements by $1 / (1 - p)$. The latter is needed for keeping mean values of features close to mean values which will be in test mode. When testing this module should implement identity transform i.e. `self.output = input`.
#
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, input):
        self.output = input
        if self.training:
            self.mask = np.random.random(size=input.shape) > self.p
            self.output = self.output * self.mask * (1 / (1 - self.p))
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.mask * (1 / (1 - self.p))
        return self.gradInput

    def __repr__(self):
        return "Dropout"


# # Activation functions

# ## ReLU


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"


# ## 6. Leaky ReLU


class LeakyReLU(Module):
    def __init__(self, slope=0.05):
        super(LeakyReLU, self).__init__()
        self.slope = slope
        self.mask = None

    def updateOutput(self, input):
        self.mask = (input > 0) + (input < 0) * self.slope
        self.output = self.mask * input
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.mask
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


# ## 7. ELU


class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def updateOutput(self, input):
        self.mask = input < 0
        self.output = input.copy()
        self.output[self.mask] = self.alpha * (np.exp(self.output[self.mask]) - 1)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.copy()
        self.gradInput[self.mask] = (
            self.gradInput[self.mask] * self.alpha * np.exp(input[self.mask])
        )
        return self.gradInput

    def __repr__(self):
        return "ELU"


# ## SoftPlus


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, input):
        self.output = np.log(1 + np.exp(input))
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * (1 - 1 / (1 + np.exp(input)))
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"


# # Criterions

# Criterions are used to score the models answers.


class Criterion(object):
    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input, target):
        """
        Given an input and a target, compute the loss function
        associated to the criterion and return the result.

        For consistency this function should not be overrided,
        all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
        Given an input and a target, compute the gradients of the loss function
        associated to the criterion and return the result.

        For consistency this function should not be overrided,
        all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Criterion"


# The **MSECriterion**, which is basic L2 norm usually used for regression, is implemented here for you.
# - input:   **`batch_size x n_feats`**
# - target: **`batch_size x n_feats`**
# - output: **scalar**


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


# ## 9. Negative LogLikelihood criterion (numerically unstable)
# - input:   **`batch_size x n_feats`** - probabilities
# - target: **`batch_size x n_feats`** - one-hot representation of ground truth
# - output: **scalar**


class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15

    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()

    def updateOutput(self, input, target):
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        self.output = -(target * np.log(input)).sum(axis=1).mean()
        return self.output

    def updateGradInput(self, input, target):
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        self.gradInput = -(target / input) / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterionUnstable"


# ## 10. Negative LogLikelihood criterion (numerically stable)
# - input:   **`batch_size x n_feats`** - log probabilities
# - target: **`batch_size x n_feats`** - one-hot representation of ground truth
# - output: **scalar**
class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = -(target * input).sum(axis=1).mean()
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = -target / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterion"


# # Optimizers

# ### SGD optimizer with momentum
# - `variables` - list of lists of variables (one list per layer)
# - `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
# - `config` - dict with optimization parameters (`learning_rate` and `momentum`)
# - `state` - dict with optimizator state (used to save accumulated gradients)


def sgd_momentum(variables, gradients, config, state):

    state.setdefault("accumulated_grads", {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):

            old_grad = state["accumulated_grads"].setdefault(
                var_index, np.zeros_like(current_grad)
            )

            np.add(
                config["momentum"] * old_grad,
                config["learning_rate"] * current_grad,
                out=old_grad,
            )

            current_var -= old_grad
            var_index += 1


def adam_optimizer(variables, gradients, config, state):

    state.setdefault("m", {})  # first moment vars
    state.setdefault("v", {})  # second moment vars
    state.setdefault("t", 0)  # timestamp
    state["t"] += 1
    for k in ["learning_rate", "beta1", "beta2", "epsilon"]:
        assert k in config, config.keys()

    var_index = 0
    lr_t = (
        config["learning_rate"]
        * np.sqrt(1 - config["beta2"] ** state["t"])
        / (1 - config["beta1"] ** state["t"])
    )

    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state["m"].setdefault(
                var_index, np.zeros_like(current_grad)
            )
            var_second_moment = state["v"].setdefault(
                var_index, np.zeros_like(current_grad)
            )

            np.add(
                config["beta1"] * var_first_moment,
                (1 - config["beta1"]) * current_grad,
                out=var_first_moment,
            )
            np.add(
                config["beta2"] * var_second_moment,
                (1 - config["beta2"]) * current_grad * current_grad,
                out=var_second_moment,
            )
            current_var -= np.divide(
                lr_t * var_first_moment, np.sqrt(var_second_moment) + config["epsilon"]
            )

            assert var_first_moment is state["m"].get(var_index)
            assert var_second_moment is state["v"].get(var_index)
            var_index += 1
