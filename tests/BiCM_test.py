import numpy as np
import bicm

def main():
    biad_mat = np.loadtxt('test_network.csv', delimiter=',', dtype=np.ubyte)
    ## Using the matrix methods
    graph = bicm.BipartiteGraph(biadjacency=biad_mat)
    val_couples_poisson = graph.get_rows_projection()
    print(val_couples_poisson)
    np.savetxt('val_couples_poisson.csv', val_couples_poisson)
    
    ## Using the edgelist methods
    edgelist, rows_degs, cols_degs = bicm.edgelist_from_biadjacency(biad_mat)
    graph = bicm.BipartiteGraph(edgelist=edgelist)
    # We may get a warning if the graph contains fully connected or isolated nodes, but the solver should work
    graph.solve_bicm(method='newton')
    graph.compute_projection(method='poibin')
    val_couples_poibin = graph.get_rows_projection()
    print(val_couples_poibin)
    np.savetxt('val_couples_poibin.csv', val_couples_poibin)

    ## Using the sparse matrix methods
    # This function would return also rows and columns dictionaries if the edgelist is not numbered sequentially
    sparse_mat, rows_degs, cols_degs, _, _ = bicm.biadjacency_from_edgelist(edgelist, fmt='sparse')
    graph = bicm.BipartiteGraph(biadjacency=sparse_mat)
    # Customizable solver: newton, quasinewton, fixed-point or root
    graph.solve_bicm(method='quasinewton')
    graph.compute_projection(method='poibin', rows=True)
    print(graph.get_rows_projection())

    ## Using only the degree sequences
    graph = bicm.BipartiteGraph(degree_sequences=(rows_degs, cols_degs))
    graph.solve_bicm(method='fixed-point')
    # Can't compute the projection with only the degree sequences of course...
    x, y = graph.get_bicm_fitnesses()
    print(x, y)
    print(bicm.bicm_from_fitnesses(x, y))
    print(graph.get_bicm_matrix())


if __name__ == "__main__":
    main()