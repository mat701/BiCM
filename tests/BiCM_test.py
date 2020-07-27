import numpy as np
import sys
sys.path.append('../BiCM')
from BipartiteGraph import *

def main():
    biad_mat = np.loadtxt('test_network.csv', delimiter=',', dtype=np.ubyte)
    
    ## Using the matrix methods
    graph = BipartiteGraph(biadjacency=biad_mat)
    val_couples_poisson = graph.get_rows_projection()
    np.savetxt('val_couples_poisson.csv', val_couples_poisson)
    
    ## Using the edgelist methods
    edgelist, rows_degs, cols_degs = edgelist_from_biadjacency(biad_mat)
    graph = BipartiteGraph(edgelist=edgelist)
    graph.solve_bicm(method='newton')
    graph.compute_projection(method='poibin')
    val_couples_poibin = graph.get_rows_projection()
    np.savetxt('val_couples_poibin.csv', val_couples_poibin)

if __name__ == "__main__":
    main()