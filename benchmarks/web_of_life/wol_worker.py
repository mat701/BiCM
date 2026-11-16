"""
Run the BiCM and its projections on every Web of Life network with one version of bicm.

It is launched by ``web_of_life_benchmark.ipynb`` once for every version to compare,
each in its own process:

    python wol_worker.py --networks data/networks.json --output results/new.pkl
    python wol_worker.py --networks data/networks.json --output results/old.pkl --bicm-path bicm_old

``--bicm-path`` is a folder containing the ``bicm`` package to import instead of the installed one.
"""
import argparse
import contextlib
import io
import json
import pickle
import signal
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

METHODS = ('poisson', 'poibin', 'normal', 'rna')


class TimeLimit(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    """Raise TimeLimit if the block runs for more than the given seconds (where SIGALRM exists)."""
    if not hasattr(signal, 'SIGALRM') or not seconds:
        yield
        return

    def handler(signum, frame):
        raise TimeLimit()
    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def load_networks(path):
    """Binary biadjacency matrices {name: (matrix, rows names, columns names)} from the Web of Life edges."""
    edges = defaultdict(list)
    with open(path) as handle:
        for edge in json.load(handle):
            if float(edge['connection_strength']) > 0:
                edges[edge['network_name']].append((edge['species1'], edge['species2']))
    networks = {}
    for name, links in edges.items():
        rows = sorted({link[0] for link in links})
        cols = sorted({link[1] for link in links})
        row_index = {node: i for i, node in enumerate(rows)}
        col_index = {node: i for i, node in enumerate(cols)}
        matrix = np.zeros((len(rows), len(cols)), dtype=int)
        for row, col in links:
            matrix[row_index[row], col_index[col]] = 1
        networks[name] = (matrix, rows, cols)
    return dict(sorted(networks.items(), key=lambda item: (item[1][0].size, item[0])))


def pairs_arrays(pvals):
    """Flatten {node: {neighbor: p-value}} in three arrays sorted by couple."""
    flat = sorted((i, j, p) for i, neighbors in pvals.items() for j, p in neighbors.items())
    if not flat:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0)
    i, j, p = zip(*flat)
    return np.array(i, dtype=int), np.array(j, dtype=int), np.array(p, dtype=float)


def poibin_cost(matrix, rows):
    """Rough cost of the exact method of version 3: one distribution of n_opposite variables,
    quadratic in time, for every couple of distinct degrees of the projected layer."""
    degrees = matrix.sum(1) if rows else matrix.sum(0)
    n_degrees = len(np.unique(degrees))
    n_opposite = matrix.shape[1] if rows else matrix.shape[0]
    return n_degrees * (n_degrees + 1) / 2 * n_opposite ** 2


def run(args):
    if args.bicm_path:
        sys.path.insert(0, args.bicm_path)
    import bicm
    from bicm import BipartiteGraph
    print('bicm', bicm.__version__, 'from', bicm.__file__, flush=True)
    warnings.simplefilter('ignore')
    networks = load_networks(args.networks)
    if args.limit:
        networks = dict(list(networks.items())[:args.limit])

    # Compile the jitted functions before timing anything
    rng = np.random.default_rng(0)
    warm = (rng.random((12, 15)) < 0.4).astype(int)
    warm[0, 0] = 1
    warm = warm[warm.sum(1) > 0][:, warm.sum(0) > 0]
    with contextlib.redirect_stdout(io.StringIO()):
        for method in METHODS:
            graph = BipartiteGraph(biadjacency=warm)
            graph.solve_tool(print_error=False)
            for rows in (True, False):
                graph.compute_projection(rows=rows, approx_method=method, progress_bar=False)

    results = {'__version__': bicm.__version__}
    start = time.time()
    for count, (name, (matrix, _, _)) in enumerate(networks.items(), 1):
        record = {'shape': matrix.shape, 'n_edges': int(matrix.sum()), 'projections': {}}
        results[name] = record
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                graph = BipartiteGraph(biadjacency=matrix)
                tic = time.perf_counter()
                graph.solve_tool(print_error=False)
                record['solve_time'] = time.perf_counter() - tic
            record['x'] = np.array(graph.x, dtype=float)
            record['y'] = np.array(graph.y, dtype=float)
        except Exception as error:  # noqa: BLE001 - the notebook reports every failure
            record['error'] = repr(error)
            continue
        for rows in (True, False):
            layer = 'rows' if rows else 'columns'
            for method in METHODS:
                out = {}
                record['projections'][(layer, method)] = out
                if args.max_poibin_cost and method == 'poibin' and poibin_cost(matrix, rows) > args.max_poibin_cost:
                    out['skipped'] = 'too slow'
                    continue
                try:
                    with contextlib.redirect_stdout(io.StringIO()), time_limit(args.timeout):
                        tic = time.perf_counter()
                        graph.compute_projection(rows=rows, alpha=args.alpha, approx_method=method,
                                                 progress_bar=False)
                        out['time'] = time.perf_counter() - tic
                    pvals = graph.rows_pvals if rows else graph.cols_pvals
                    out['pairs_i'], out['pairs_j'], out['pvals'] = pairs_arrays(pvals)
                    adj_list = graph.projected_rows_adj_list if rows else graph.projected_cols_adj_list
                    out['links'] = sorted((i, j) for i, neighbors in adj_list.items() for j in neighbors)
                except TimeLimit:
                    out['skipped'] = 'time limit'
                except Exception as error:  # noqa: BLE001
                    out['error'] = repr(error)
        if count % 20 == 0 or count == len(networks):
            print('{:4d} / {} networks, {:.0f} s'.format(count, len(networks), time.time() - start), flush=True)
    with open(args.output, 'wb') as handle:
        pickle.dump(results, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--networks', required=True, help='JSON file with the edges of all the networks')
    parser.add_argument('--output', required=True, help='pickle file to write')
    parser.add_argument('--bicm-path', default=None, help='folder with the bicm package to test')
    parser.add_argument('--alpha', type=float, default=0.05, help='significance level of the projections')
    parser.add_argument('--timeout', type=int, default=0, help='time limit in seconds of one projection')
    parser.add_argument('--max-poibin-cost', type=float, default=0,
                        help='skip the exact method above this estimated cost (version 3 only)')
    parser.add_argument('--limit', type=int, default=0, help='run only the smallest networks')
    run(parser.parse_args())
