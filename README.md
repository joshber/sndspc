# Sndspc

## Pattern spaces for sound

Josh Berson, josh@joshberson.net

This is a series of experiments in constructing **pattern spaces** for acoustic information.

The basic idea is to

1. Formulate robust low-dimensional embeddings of high-dimensional similarity relationships among brief (~10–60s) samples of field-recorded ambient sound. “Similarity” here is defined in terms of pairwise relationships in the spectral envelopes of the sampled sounds. Part of the challenge is to come up with a sparse approximation of spectral envelope. Another part of the challenge is to make the embedding parametric, so that we might reliably interpolate new sounds into the spectral pattern space, without doing too much violence to nonlinearities in the similarity mapping.

2. Building on what we learn from (1), formulate a supervised learning problem for the sensorimotor qualities of different sounds — how sounds in different parts of the pattern space condition (or at least, are implicated in) different qualities of mood and alertness. Then we could extend the pattern space analysis to these higher-order sensorimotor features of sound.

This project builds on my work, in collaboration with [LUSTlab](http://lustlab.net), on the [Cartographies of Rest](https://lust.nl/#projects-7158) project for the Wellcome-sponsored project Hubbub.

It also builds on [Nadav Hochman](http://nadavhochman.net/publications/) and Lev Manovich' work on **style spaces** for images, especially as Hochman has applied the style space program to photographic social media.
