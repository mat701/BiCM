How to install the BiCM package
=====================================

Install it via pip on your machine:

.. code-block:: bash

    pip install bicm

The package requires Python 3.10 or later and depends on numpy, scipy and numba.

The null models for weighted networks of :mod:`bicm.wbnm` are implemented with PyTorch, which is an optional dependency. To install it together with the package:

.. code-block:: bash

    pip install "bicm[weighted]"

The results of :mod:`bicm.fast_projection` can be exported to a ``pandas.DataFrame`` if pandas is installed.

To install the package from source together with the test dependencies and run the tests:

.. code-block:: bash

    git clone https://github.com/mat701/BiCM
    cd BiCM
    pip install -e ".[test]"
    pytest

Visit https://github.com/mat701/BiCM to download the package instead.
