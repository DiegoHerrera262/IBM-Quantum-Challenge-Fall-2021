# Introduction to Qiskit Finance

In this challenge, the idea is to perform some **portfolio optimizations** using quantum computers. There are several approaches to this problem from a quantum computing perspective. The _fault tolerant_ perspective relies on _Grover's algorithm_ for finding the optimal asset allocation that maximizes revenue for a given market risk. The _near term_ perspective relies on hybrid quantum computing (i.e. VQE, QAOA, etc.) for solving that problem.

Let's state in a clearer fashion the problem of portfolio optimization. A portfolio is a collection of **assets**, modelled by a probability distribution, on which each asset has an associated revenue and risk. Usually, the higher the risk, the higher the revenue, so that there is a tradeoff. The natural question thus is how to choose a subset of assets so that expected revenue is maximized, while risk is minimized or at least controlled.

From a mathematical perspective, assets in a portfolio problem are modelled by a _multivariate probability distribution_

$$\vec{X} = (X_1, X_2, \cdots, X_N)$$

In general, each partial distribution determines the expected revenue obtained by investing on a particular asset. And the variance is associated to a particular distribution is related to the risk of investment. In general:

$$\vec{X} = \mathcal{N}(\vec{\mu}, \Sigma)$$

Where $\Sigma$ is the **covariance matrix**. And $\vec{\mu}$ is the vector of mean expected values of revenue for each asset. Wee can choose a binary vector $\vec{x}$, that indicates which assets are to be included or excluded from our portfolio. WIth this, the cost function we would like to maximize is of the shape

$$
\langle \vec{x} \cdot \vec{X}\rangle - q R(\vec{x}, \vec{X})
$$

The first term is simply the expected revenue, while the second term reflects that we are constrained by the risk in our inversions. We would like to have a fixed risk, low if possible, and also have the optimal revenue. A common measure of the risk is the _variance_. And so, the function to be maximized in this case is

$$
\langle \vec{x} \cdot \vec{X}\rangle - q \text{Var}(\vec{x} \cdot \vec{X})
$$

Here $q$ is a Lagrange multiplier commonly referred to as _risk factor_. Under the assumption that the joint distribution is normal, the expression above reduces to

$$
\vec{x} \cdot \vec{\mu} - q \vec{x} \cdot \Sigma \cdot \vec{x}
$$

> In this set up, the main question is how to find a binary vector (of zeroes and ones) that describes the optimal choice of assets constrained by the risk involved.

The happy insight is that this is a quadratic program, something that can be optimized with quantum algorithms.

## Description of the challenge

Well, the basic workflow is the following:

1. Use historical data for computing the expected revenue and the covariance matrix.
1. Use this data to define a portfolio optimization problem.
1. Use any of Qiskit's classical or hybrid solvers.

### Random data generation with Qiskit

This can be done by using any of data generators from `qiskit_finance.data_providers`:

```python
from qiskit_finance.data_providers import *

random_data = RandomDataProvider(
    tickers = ['ASSET%S' % i for i in range(num_assets)],
    start=datetime.datetime(1955,11,5),
    end=datetime.datetime(1985,10,26),
    seed=seed
)
```

This produces historic data for model validation. The cool part of this data provider is that it can compute the _mean expected revenue_

```python
mu = data.get_period_return_mean_vector()
```

As well as the asset covariance matrix

```python
sigma = data.get_period_return_covariance_matrix()
```

### Quadratic problem definition

A portfolio optimization problem can be instantiated using Qiskit Finance's dedicated class `PortfolioOptimization`. That can be done as follows

```python
from qiskit_finance.applications.optimization import PortfolioOptimization

portfolio_opt_problem = PortfolioOptimization(
    expected_returns = mu,
    covariances = sigma,
    risk_factor = risk_factor,
    budget = budget
)
```

The magic of Qiskit is that it maps a variety of real world problems to a variety of abstract optimization problems that can be solved by standard techniques. This is such a case. Now, we have seen that the portfolio optimization problem can be mapped to a standard quadratic problem. To perform the mapping, the above class has a dedicated method

```python
quadratic_program = portfolio_opt_problem.to_quadratic_program()
```

Now, we can use Qiskit's built-in tools for solving the problem.

### Solving the problem with VQE

Now, solving a problem is pretty straightforward with qiskit. An instance of an standard algorithm is created using some variational form, a classical optimizer, and a backend for quantum computations. A pretty common variational form is called `TwoLocal`. This form is based upon rotation and entangling layers in tandem (rotation - entangling). For finance problems, since the problem maps directly to a _real-valued_ Ising Model, the proper rotation gates are Ry. As entangling gates, thus, the best are CNOT. A variational form can be instantiated as follows

```python
from qiskit.utils import algorithm_globals

var_form = TwoLocal(
    num_qubits = num_assets,
    rotation_blocks = 'ry',
    entanglement_blocks = 'cx',
    entanglement = 'full',
    reps = 3
)
```

What we are doing here is defining a variational form that has de desired entangling and rotation blocks, and is composed of 3 tandem layers. It is acting on a register whose number of qubits is equal to the number of assets in our portfolio estimation problem. As a classical optimizer, we can use one that fits best. It can be COBYLA or SQLP. Instantiating a classical optimizer is easy:

```python
optimizer = SLSQP(maxiter=1000)
```

For reproducibility, a seed for the random generators may be set:

```python
algorithm_globals.random_seed = 1234
```

Now, a quantum instance is used for actually running the quantum part of the computation

```python
seed = 123
quantum_instance = QuantumInstance(
    backend = Aer.get_backend('statevector_simulator'),
    seed_simulator = seed,
    seed_transpiler = seed
)
```

Finally, a VQE can be instantiated for optimizing the cost function

```python
vqe = VQE(
    var_form,
    optimizer = optimizer,
    quantum_instance = quantum_instance
)
```

Since the whole point of VQE is to find eigenvalues, we use a minimum eigenvalue solver instance for solving the quadratic program

```python
from qiskit_optimization.algorithms import MinimumEigenOptimizer

vqe_meo_solver = MinimumEigenOptimizer(vqe)
results = vqe_meo_solver.solve(quadratic_program)
```

I find it awesome that Qiskit takes care of the qubit mapping of the cst function, and translates the quadratic program into a Hamiltonian problem. This can be misleading for newcomers to quantum physics.
