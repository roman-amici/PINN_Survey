# PINN_Survey

A Survey of Physics Informed Neural Network Test Problems, Architectures, and Optimization Techniques

## Introduction

Physics Informed Neural Networks (PINNs) were introduced in [https://arxiv.org/abs/1711.10561](Raissi, 2017) as a potentially elegant way to integrate knowledge of physical laws and deep neural networks. The authors conclude with the following

> Although a series of promising results was presented, the reader may perhaps agree this work creates more questions than it answers.

In the almost three years since the original preprint appeared, it has gained approximately 130 citations with many papers following on the original approach. Has this flurry of activity yielded answers to these questions? At the moment, I think the answer is that its hard to tell. Many seem to be simply retreading the same ground as the initial paper-- performing the same sorts of experiments on different example problems and further declaring the results acceptable as a first step while noting there is more work to be done in this domain.

A second class of paper puts forward something new, be it a new architecture, loss function, or optimization procedure, and then claims that it is an improvement over what came before. Such claimed improvements can be hard to judge for a number of reasons. The first is that PINNs are sensitive to the choice of hyperparameters and of random initialization. It is hard to disentangle whether an improvement came from the novel addition or was a matter of luck. This is compounded by the fact that comparisons to baselines often occur on problems introduced for the first time (in the PINNs context) in the paper itself. Thus, its hard to know if a better base line could be obtained by tweaking some or another parameter. Speaking of tweaking, many papers often suffer from changing many things at once. They may introduce a new architecture, and a new training procedure, and a new loss function all at once. Without a proper ablation study, it is impossible to tell which change was the most impactful. More importantly, it can be hard to tell which improves things for a specific problem and which produces more general improvements.

This is not disparage anyone's work, nor to claim that this problem is specific to PINNs as a research area, it is merely to point out that the current state of things makes it difficult to perform and share reasearch. PINNs currently lack both clear goals and clear ways to evaluate progress toward those goals.

I would like to take steps toward fixing this. I would like to first propose several guiding questions for each of the four identified uses for PINNs. I will then attempt to put forth measurable benchmarks in these areas. Implementations of these benchmarks will gathered in a common repository and implemented using a common programmatic framework to aid in reproducibility and compositionality.

## Guiding Questions

It is helpful to organize these investigations by the following existential question: Is there anything PINNs are good for? As Raissi et al put it,

> the proposed methods should not be viewed as replacements of classical numerical methods for solving partial differential equations (e.g., finite elements, spectral methods, etc.). Such methods have matured over the last 50 years and, in many cases, meet the robustness and computational efficiency standards required in practice.

Unlike in other machine learning domains, where neural networks beat the existing state of the art, it is not clear to me that PINNs currently have any area where they definitively beat classical methods whether in relative or absolute terms. It may be that particular examples do exist already. It would be helpful to catalogue them. This should be supplemented by an inventory of where we think PINNs are most likely to outperform classical methods given further research.

This should also emphasize the need for comparisons to classical methods where possible. This can be in terms of error, computation time, or even ease of use.

I have decided to place the following uses of PINNs into two main categories. For each category I will list proposed sources of advantage for each

- Finding solutions to differential equations
  - Mesh-free formulation defined at all points on the domain
  - Scalability to high dimensional problems
  - Robustness to noise in boundary/initial conditions
  - Ability to learn models for families of differential equations
  - Superlinear convergence due to non-linear approximating function
  - Ability to leverage empirical data in the solution
- Inference of parameters or residual functions from data
  - Training points need not be limited to a regular grid
  - Robustness to noise
  - Minimal errors in taking derivatives due to auto-differentiation
  - Ability to fill with differential points where data points are sparse

### General Questions

1. What problems do PINNs work on currently? Where do they not work and where do they work well?

   - Can this be mapped onto existing categories in the theory of differential equations? Linear, non-linear, linearizable? Parabolic, hyperbolic, elliptic?

2. What are current best practices surrounding the formulation and training of PINNs?

   - Adam or BFGS? What activations? How large should the network be? How many points are necessary

3.
