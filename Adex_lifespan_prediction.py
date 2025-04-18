import nengo
import numpy as np
import scipy.special as sp
import scipy.stats as st
import pickle as pickle
import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace
import sys
from nengo.neurons import NeuronType
from nengo.builder import Builder, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import *
from nengo.builder.learning_rules import *
from nengo.params import (NumberParam)
import numpy as np
from nengo.utils.neurons import settled_firingrate
import numpy as np
from nengo.builder import Builder, Signal
from nengo.builder.neurons import SimNeurons
from nengo.params import NumberParam
from nengo.neurons import NeuronType

class Adex(NeuronType):
    """Adaptive Exponential Integrate-and-Fire (Adex) neuron model."""

    probeable = ('spikes', 'voltage', 'adaptation')

    # Define model parameters
    tau_w = NumberParam('tau_w', low=0, low_open=True)  # Adaptation time constant
    a = NumberParam('a', low=0)  # Subthreshold adaptation
    b = NumberParam('b', low=0)  # Spike-triggered adaptation
    delta_T = NumberParam('delta_T', low=0)  # Slope factor
    reset_voltage = NumberParam('reset_voltage')  # Reset voltage (E_L)
    V_T = NumberParam('V_T')  # Threshold voltage

    def __init__(self, tau_w=144e-3, a=4e-9, b=0.0805e-9,
                 delta_T=2e-3, reset_voltage=-70e-3, V_T=-50.4e-3):
        super().__init__()
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.delta_T = delta_T
        self.reset_voltage = reset_voltage
        self.V_T = V_T

    @property
    def _argreprs(self):
        args = []
        def add(attr, default):
            if getattr(self, attr) != default:
                args.append(f"{attr}={getattr(self, attr)}")
        add("tau_w", 144e-3)
        add("a", 4e-9)
        add("b", 0.0805e-9)
        add("delta_T", 2e-3)
        add("reset_voltage", -70e-3)
        add("V_T", -50.4e-3)
        return args


    def rates(self, x, gain, bias):
        """Computes steady-state firing rates given input x."""
        
        J = self.current(x, gain, bias)  # Compute input current
        voltage = np.ones_like(J) * self.reset_voltage
        adaptation = np.zeros_like(J)

        # Use a short simulation to determine steady-state firing rate
        return settled_firingrate(self.step_math, J, [voltage, adaptation],
                                settle_time=0.001, sim_time=1.0)

    def step_math(self, dt, J, spiked, voltage, adaptation):
        """Performs one time step of simulation."""
        C = 281e-12  # Membrane capacitance (F)
        g_L = 30e-9  # Leak conductance (S)
        E_L = self.reset_voltage  # Leak potential (V)

        # Compute voltage update (adaptive exponential IF equation)
        dV = (-g_L * (voltage - E_L) 
              + g_L * self.delta_T * np.exp((voltage - self.V_T) / self.delta_T)
              - adaptation + J) / C
        voltage[:] += dV * dt

        # Check for spikes
        spiked[:] = (voltage >= 20e-3) / dt  # Spike occurs at 20mV
        voltage[spiked > 0] = self.reset_voltage  # Reset voltage to E_L
        adaptation[:] += (self.a * (voltage - E_L) - adaptation) * dt / self.tau_w
        adaptation[spiked > 0] += self.b  # Increase adaptation after spike

@Builder.register(Adex)
def build_adex(model, adex, neurons):
    """Build the Adex neuron model for simulation."""
    model.sig[neurons]['voltage'] = Signal(
        np.ones(neurons.size_in) * adex.reset_voltage,
        name=f"{neurons}.voltage"
    )
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in), 
        name=f"{neurons}.adaptation"
    )

    model.add_op(SimNeurons(neurons=adex,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['voltage'],
                                    model.sig[neurons]['adaptation']]))

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print ("Require a file to store the output data")
    exit(0)


max_age = dim = 120

# our domain is thetas (i.e., age from 1 to 120)
thetas = np.linspace(start=1, stop=max_age, num=max_age)


# prior parameters
skew = -6
loc = 99
scale = 27

def likelihood(x):
    x = int(x)
    like = np.asarray([1/p for p in thetas])
    like[0:x-1] = [0]*np.asarray(x-1)
    return like

def skew_gauss(skew, loc, scale):
    return [(st.skewnorm.pdf(p, a=skew, loc=loc, scale=scale)) for p in thetas] 
    
def posterior(x, skew, loc, scale):
    post = likelihood(x=x)*skew_gauss(skew=skew, loc=loc, scale=scale)
    return post

def normalized_posterior(x, skew, loc, scale):
    post = posterior(x, skew, loc, scale)
    post = post/sum(post)
    return post



ages = np.linspace(start=1, stop=100, num=100, dtype=np.int32)
data = {}
n_basis = 20
for x in ages:
    if x<5:
        pad = 5-x+1
    else:
        pad = 0
       
    # define sub-spaces
    space = nengo.FunctionSpace(
            nengo.dists.Function(skew_gauss,
                             skew=nengo.dists.Uniform(skew-1, skew+2), 
                          loc=nengo.dists.Uniform(loc-1,loc+2), 
                          scale=nengo.dists.Uniform(scale-1, scale+2)),
                            n_basis=n_basis)

    from copy import deepcopy
    space_raw = deepcopy(space.space)


    lik_space = nengo.FunctionSpace(
                    nengo.dists.Function(likelihood,
                                x=nengo.dists.Uniform(x-5+pad,x+5+pad)),
                    n_basis=n_basis)
    
    lik_space_raw = deepcopy(lik_space.space)

    post_space = nengo.FunctionSpace(
                    nengo.dists.Function(posterior,
                                 x=nengo.dists.Uniform(x-5,x+5),
                                skew=nengo.dists.Uniform(skew-1, skew+2), 
                              loc=nengo.dists.Uniform(loc-50,loc+2), 
                              scale=nengo.dists.Uniform(scale-1, scale+2)),
                    n_basis=n_basis)
    
    post_space_raw = deepcopy(post_space.space)

    norm_post_space = nengo.FunctionSpace(
                nengo.dists.Function(normalized_posterior,
                             x=nengo.dists.Uniform(x-5+pad,x+5+pad),
                            skew=nengo.dists.Uniform(skew-1, skew+2), 
                          loc=nengo.dists.Uniform(loc-50,loc+2), 
                          scale=nengo.dists.Uniform(scale-1, scale+2)),
                n_basis=n_basis)

    norm_post_space_raw = deepcopy(norm_post_space.space)

   
    k = np.zeros((120, n_basis))        
    j = 0
    for element in space.basis.T:
        a = np.multiply(element, lik_space.basis.T[j])
        k[:, j] = a 
        j = j + 1        

    post_space._basis = k
    model = nengo.Network()
    with model:
        stim = nengo.Node(label="prior input", output=space.project(skew_gauss(skew=skew, loc=loc, scale=scale)))
        ens = nengo.Ensemble(label="Prior", n_neurons=200, dimensions=space.n_basis, neuron_type=Adex(),
                             encoders=space.project(space_raw),
                             eval_points=space.project(space_raw),
                            )
        
        stim2 = nengo.Node(label="likelihood input", output=lik_space.project(likelihood(x=x)))
        ens2 = nengo.Ensemble(label="Likelihood", n_neurons=200, dimensions=lik_space.n_basis, neuron_type=Adex(),
                             encoders=lik_space.project(lik_space_raw),
                             eval_points=lik_space.project(lik_space_raw),
                            )
        
        
        nengo.Connection(stim, ens)
        probe_func = nengo.Probe(ens, synapse=0.03)
        
        nengo.Connection(stim2, ens2)
        probe_func2 = nengo.Probe(ens2, synapse=0.03)
        
        # elementwise multiplication
        post = nengo.Ensemble(label="Posterior", n_neurons=200, dimensions=post_space.n_basis,
                                 encoders=post_space.project(post_space_raw),
                                 eval_points=post_space.project(post_space_raw),
                                 neuron_type = nengo.Direct())
        product = nengo.networks.Product(n_neurons=100*2, dimensions=post_space.n_basis, input_magnitude=1)
        
        nengo.Connection(ens, product.A)
        nengo.Connection(ens2, product.B)
        nengo.Connection(product.output, post)
        probe_func3 = nengo.Probe(post, synapse=0.03)
        
        # normalization
        def normalize(a):
            b = np.dot(a, k.T)
            total = np.sum(b)
            if total == 0:
                return [0]*dim
            return b/total
        
        
        # Note: this population needs to have around 250 neurons for accurate representation
        norm_post = nengo.Ensemble(label="Normalized Posterior", n_neurons=800, dimensions=dim, neuron_type=Adex(),
                                   encoders=norm_post_space_raw,
                                 eval_points=norm_post_space_raw)
        nengo.Connection(post, norm_post, function=normalize)
        probe_func4 = nengo.Probe(norm_post, synapse=0.03)
        
        # prediction
        def median(b):
            med = 0
            for n in np.arange(len(b)):
                cum = sum(b[:n+1])
                if cum == 0.5 or cum > 0.5:
                    med = n + 1
                    break
            return int(med)
    
        
        prediction = nengo.Node(label="Prediction", output=None, size_in=1)
           
        nengo.Connection(norm_post, prediction, function=median, synapse=0.03)
        probe_func5 = nengo.Probe(prediction, synapse=0.03)
            
            
    sim = nengo.Simulator(model)
    sim.run(0.5)

    node_prediction = sim.data[probe_func5][-1][0]
    data[x] = [0, node_prediction]

pickle.dump(data, open(fname, 'wb'))
print("pickle complete")
print(fname)