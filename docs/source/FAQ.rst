FAQ
=====

Does the package work with monopartite networks?
---------------------------------------------------------------------------------------

No, it doesn't. But you are looking for the `NEMtropy package <https://github.com/nicoloval/NEMtropy>`_, see https://meh.imtlucca.it .

Does the package work with weighted networks?
---------------------------------------------------------------------------------------

Yes. The BipartiteGraph class solves the BiWCM_d (integer weights) and BiWCM_c (continuous weights) models and validates the links of the network, see :doc:`usage`. More models (BiECM, BiPECM, BiCReMA) and Monte Carlo projections of weighted networks are in :doc:`weighted`. The analytical projection is implemented for binary networks only.

The computation of the BiCM takes too long. What can I do?
---------------------------------------------------------------------------------------

Try working with the functions that only use the degree sequence of your bipartite network, use a different solver, different initial conditions, or a lower tolerance. Make sure you don't have isolated nodes or fully connected nodes in your network. If you have them, only initializing the object via a biadjacency matrix will discount this information.

The computation of the projection takes too long. What can I do?
---------------------------------------------------------------------------------------

Since version 4.0 all the methods, the exact one included (approx_method='poibin'), work on the distinct degrees of the network and are much faster. If the network has millions of nodes or does not fit in memory, use :doc:`fast_projection`, which works directly on the list of edges.

Some p-values are extremely small, like 1e-200. Are they reliable?
---------------------------------------------------------------------------------------

Yes. The p-values are computed as right tails with a small relative error, they are not affected by the machine precision of "1 - cdf". Values below about 1e-300 cannot be represented and are returned as zero. See :doc:`usage`.