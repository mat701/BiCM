FAQ
=====

Does the package work with monopartite networks?
------

No, it doesn't. For a complete guide of Maximum entropy configuration models, see https://meh.imtlucca.it/ .

The computation of the BiCM takes too long. What can I do?
------

Try working with the functions that only use the degree sequence of your bipartite network, use a different solver, different initial conditions, or a lower tolerance.

The computation of the projection takes too long. What can I do?
------

Try a different Poisson-binomial approximation. However, this computation can be slow for big networks.