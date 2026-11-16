# Web of Life benchmark

`web_of_life_benchmark.ipynb` downloads all the networks of the [Web of Life](https://www.web-of-life.es/) database and runs the BiCM solver and the validated projections on each of them with bicm 3.5.0 and with the installed version, comparing running times, p-values (also against a brute force reference) and validated links. It also runs the weighted models BiWCM_d and BiWCM_c on the weighted networks, checking the convergence, the p-values of the links against scipy and, if PyTorch is installed, the solution against `bicm.wbnm`.

To run it:

    pip install bicm pandas matplotlib tqdm jupyter
    jupyter notebook web_of_life_benchmark.ipynb

`wol_worker.py` is the script that the notebook launches once for each version. Downloads and results are cached in `data/` and `results/`, which are not part of the repository.
