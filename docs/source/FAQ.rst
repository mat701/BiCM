FAQ
=====

Does the package work with monopartite networks?
------

No, it doesn't. Maybe you are looking for a different configuration model? For a complete guide of maximum entropy configuration models, see meh.imtlucca.it .

Does the package work with weighted networks?
------

No.

The computation of the BiCM takes too long. What can I do?
------

Try working with the functions that only use the degree sequence of your bipartite network, use a different solver, different initial conditions, or a lower tolerance. Make sure you don't have isolated nodes or fully connected nodes in your network. If you have them, only initializing the object via a biadjacency matrix will discount this information.

The computation of the projection takes too long. What can I do?
------

If you're using the Poisson-Binomial method (method='poibin') try a different Poisson-binomial approximation, like the Poisson one (method='poisson'). However, this computation can be slow for big networks.