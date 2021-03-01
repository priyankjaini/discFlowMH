# SurVAE Augmented MCMC for Combinatorial Spaces

> Official code for [Sampling in Combinatorial Spaces with SurVAE Flow Augmented MCMC](https://arxiv.org/abs/2102.02374)  
by Priyank Jaini, Didrik Nielsen, Max Welling.

This paper presents a general purpose sampler for discrete distributions using recent advances in neural transport methods like normalizing flows and the Metropolis-Hastings rule. The algorithm learns a continuous embedding of the discrete space using a surjective map and subsequently learns a bijective transformation from the continuous space to an approximately Gaussian distributed latent variable. Sampling proceeds by simulating MCMC chains in the latent space and mapping these samples to the target discrete space via the learned transformations.

![alt text](https://pbs.twimg.com/media/EtfbV80XIAEXmXU?format=jpg&name=medium)

For more details, see [the paper](https://arxiv.org/abs/2102.02374).

## Contents

* `/ising/`: Code for experiments for the Ising model.
* `/quantized/`: Code for experiments for Quantized regression.
* `/bayesian_regression/`: Code for experiments for Bayesian variable selection

## Reference

    @inproceedings{jaini2021sampling,
        title={Sampling in Combinatorial Spaces with SurVAE Flow Augmented MCMC},
        author={Jaini, Priyank and Nielsen, Didrik and Welling, Max},
        booktitle={The 24th International Conference on Artificial Intelligence and Statistics, AISTATS},
        year={2021}
    }
    
For any questions, please contact Priyank Jaini (p.jaini@uva.nl) or Didrik Nielsen (didni@dtu.dk) 

