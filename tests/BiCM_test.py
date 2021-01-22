# import sys
# sys.path.append('../bicm/')
# from BipartiteGraph import *
# import bicm
from bicm import BipartiteGraph
import numpy as np

# This test uses a plant-pollinator network from Petanidou, T. (1991). Pollination ecology in a phryganic ecosystem.
# This script is run on an ordinary laptop.

biad_mat_names = np.loadtxt('test_dataset.csv', delimiter=',', dtype=str)
plants = biad_mat_names[1:,0]
pollinators = biad_mat_names[0, 1:]
biad_mat = biad_mat_names[1:, 1:].astype(np.ubyte)
plants_dict = dict(enumerate(np.unique(plants)))
plants_inv_dict = {v:k for k,v in plants_dict.items()}
pollinators_dict = dict(enumerate(np.unique(pollinators)))
pollinators_inv_dict = {v:k for k,v in pollinators_dict.items()}

# You can manipulate a data structure into another with the functions of the package. It supports biadjacency matrices (numpy arrays or scipy sparse matrices), edgelists, adjacency lists (dictionaries) or just the degree sequences.

# Let's create an edgelist with names.

# These data type converters return additional objects, keeping track of the node labeling and the node degrees.
# The desired data structure is always at index 0. Check the documentation for more info.
edgelist = edgelist_from_biadjacency(biad_mat)[0]
edgelist_names = [(plants_dict[edge[0]], pollinators_dict[edge[1]]) for edge in edgelist]
print(edgelist_names[:5])

# Now we can use the edgelist to create the bipartite graph. In two ways:

myGraph = BipartiteGraph(edgelist=edgelist_names)

myGraph = BipartiteGraph()
myGraph.set_edgelist(edgelist_names)


# And now let's simply compute the bicm! This should run instantly. The solver checks that the solution is correct automatically.

myGraph.solve_bicm()
dict_x, dict_y = myGraph.get_bicm_fitnesses()
print('Yielded data type is:', type(dict_x))


# The computed fitnesses are given as dictionaries, keeping track of the name of the nodes. If you are sure about the order of the fitnesses, you can get separately the fitness vectors and the dictionaries that keep track of the order of the nodes:

x = myGraph.x
y = myGraph.y
rows_dict = myGraph.rows_dict
cols_dict = myGraph.cols_dict


# Alternatively, you can have the bicm matrix of probabilities, whose entries i, j are the bicm probabilities of having a link between nodes i and j (who will correspond to rows_dict[i] and cols_dict[j]), or compute it on your own

avg_mat = myGraph.get_bicm_matrix()
print(avg_mat[0, 0] == x[0] * y[0] / (1 + x[0] * y[0]))

avg_mat = bicm_from_fitnesses(x, y)
print(avg_mat[0, 0] == x[0] * y[0] / (1 + x[0] * y[0]))


# And, if you like, you can check the solution by yourself:
# The data type conversions are generally consistent with the order, so this works

check_sol(biad_mat, avg_mat)


# You can then sample from the BiCM ensemble in two ways, via matrix sampling or edgelist sampling.

# Here we generate a sampling of 1000 networks and calculate the distribution of the number of links.
tot_links = []
for sample_i in range(1000):
    sampled_network = sample_bicm(avg_mat)
    tot_links.append(np.sum(sampled_network))

sampled_edgelist = sample_bicm_edgelist(x, y)

# Now let's compute the projection: it can take some seconds.

rows_projection = myGraph.get_rows_projection()
print(rows_projection)

# If you don't want to get the progress_bar you can set progress_bar=False. If you want to re-compute the projection with different settings, use compute_projection()

myGraph.compute_projection(rows=True, alpha=0.01, progress_bar=False)

rows_projection = myGraph.get_rows_projection()
print(rows_projection)

# These projections only contain links between nodes that behave similarly with respect to the expected behavior calculated from the BiCM. They could also be empty:

cols_projection = myGraph.get_cols_projection()

# Increasing alpha too much will make the projection too dense. 
# It is not suggested: if it is empty with a high alpha, try different projections.
# Check the documentation and the papers for more info.
myGraph.compute_projection(rows=False, alpha=0.2)

# Different projection methods are implemented. 
# They differ in the way of approximating the poisson-binomial distribution of the similarities between nodes.
# By default, the poisson approximation is used, since the poisson binomial is very expensive to compute. Check the docs for more!
# Here, an example of the poibin implementation. This can take even some minutes, and it's faster when using a biadjacency matrix instead of other data types. Note that the poisson approximation is fairly good.

myGraph.compute_projection(rows=False, method='poibin')
rows_projection = myGraph.get_rows_projection()
print(rows_projection)

# Now, let's have fun with other data structures and solver methods.

adj_list_names = adjacency_list_from_edgelist(edgelist_names, convert_type=False)[0]

myGraph2 = BipartiteGraph(adjacency_list=adj_list_names)

# Asking for the projection immediately will do everything automatically.

myGraph2.get_rows_projection()

rows_deg = biad_mat.sum(1)
cols_deg = biad_mat.sum(0)

myGraph3 = BipartiteGraph(degree_sequences=(rows_deg, cols_deg))

# Default (recommended) method is 'newton', three more methods are implemented

myGraph3.solve_bicm(method='fixed-point', tolerance=1e-5, exp=True)

myGraph3.solve_bicm(method='quasinewton', tolerance=1e-12, exp=True)

# This is the only method that works with a scipy solver, not recommended.

myGraph3.solve_bicm(method='root')

# If you want to retrieve the p-values of the projection on one layer, use one of the following functions:

# Use the rows arg to project on the rows or columns
adj_list = adjacency_list_from_adjacency_list(adj_list_names)[0]
pval_list = projection_calculator_light(edgelist, x, y, return_pvals=True, rows=True)
pval_list = projection_calculator_light_from_adjacency(adj_list, x, y, return_pvals=True)
pval_list = projection_calculator(biad_mat, avg_mat, return_pvals=True)


# Finally, if you want to manage the calculation of the pvalues for the projection on your own, you can use the class PvaluesHandler (see more in the docs)

# For the projection on the rows layer:
v_list = vmotifs_from_adjacency_list(adj_list)
pval_obj = PvaluesHandler()
pval_obj.set_fitnesses(x, y)
pval_obj.compute_pvals(v_list, method='poisson', threads_num=4, progress_bar=True)
pval_list = pval_obj.pval_list
# Contains a list of the couples of nodes that share at least a neighbor with the respective pvalue
print(pval_list[:5])

# For the projection on the opposite layer:
inv_adj_list = adjacency_list_from_adjacency_list(adj_list_names)[1]
v_list = vmotifs_from_adjacency_list(inv_adj_list)
pval_obj = PvaluesHandler()
pval_obj.set_fitnesses(y, x)
pval_obj.compute_pvals(v_list, method='poisson', threads_num=4, progress_bar=True)
pval_list = pval_obj.pval_list
# Contains a list of the couples of nodes that share at least a neighbor with the respective pvalue
print(pval_list[:5])
