APPARENT 
======================

.. image:: _static/referral-network-example.png
   :width: 250
   :height: 250
   :alt: Referral network
   :align: right
   
**Apparent** is a Python toolkit for analyzing patient referral flows within US healthcare systems using **medical claims data** (Medicare). We provide functionality for building and analyzing patient referral networks. In particular, we provide functionality to analyze these networks via **discrete curvature** and **persistent homology**, in hopes of supporting further research developments into using network analysis to improve efficiency and equity of the US healthcare system.

Prototype & Publication
-----------------------

- **Prototype Tool**: `apparent.topology.rocks <https://apparent.topology.rocks/>`_
- **Paper**: `Analyzing Physician-Patient Referral Networks Using Discrete Curvature and Persistent Homology <https://arxiv.org/abs/2408.16022>`_

Features
--------

- **Build Networks**: Generate directed and undirected graphs representing referral relationships between physicians and patients.
- **Describe Networks**: Compute node- and edge-level features like degree, clustering coefficients, betweenness centrality, curvature measures, and persistence diagrams.
- **Compare Networks**: Measure pairwise similarity or distances between networks using curvature metrics and topological features.
- **Embed Networks**: Map networks into a lower-dimensional vector space for visualization and machine learning tasks.
- **Cluster Networks**: Group networks into clusters based on structural or functional similarity, leveraging techniques like k-means, hierarchical clustering, and DBSCAN.

Installation
------------

APPARENT uses `uv <https://github.com/astral-sh/uv>`_ as the package manager, which provides faster dependency resolution and installation.

**Step 1: Clone the Repository**

.. code-block:: bash

    git clone https://github.com/aidos-lab/apparent.git
    cd apparent

**Step 2: Install Dependencies and Activate Virtual Environment**

If you don't already have ``uv``, install with pip:

.. code-block:: bash

    pip install uv

To install dependencies, run:

.. code-block:: bash

    uv sync

You'll notice this creates a ``.venv`` folder in the root directory.

We activate that new virtual environment as such:

.. code-block:: bash

    source .venv/bin/activate

**Step 3: Specify Environment Variables in .env**

.. code-block:: bash

    touch .env
    echo APPARENT_URL="https://apparent.topology.rocks/us_physician_referral_networks.csv" >> .env

This points the directory to the location where the database is stored.

Usage
-----

Most actions can completed using the ``Apparent`` object, including the following functionality:

1. *Pull Data:* Extract data from our datasette tool (or local instance).
2. *Build Networks:* Construct physician-patient referral graphs from the extracted data.
3. *Describe Networks:* Compute network features such as curvature, centrality, and clustering coefficients.
4. *Compare Networks:* Analyze pairwise distances between networks using metrics like Forman curvature and Ollivier-Ricci curvature.
5. *Embed Networks:* Reduce dimensionality for visualization and machine learning.
6. *Cluster Networks:* Group similar networks using clustering algorithms like ``KMeans``, ``DBSCAN``, and hierarchical clustering.

Quick Example
~~~~~~~~~~~~~

Here's a quick example for how you can pull specific Physician Referral Networks using ``apparent``!

.. code-block:: python

    from apparent import Apparent

    # Initialize Apparent
    A = Apparent()

    # Example SQL query for fetching data
    my_query = """
              SELECT
                hospital_atlas_data.hsa,
                hospital_atlas_data.year,
                hospital_atlas_data.latitude,
                hospital_atlas_data.longitude
              FROM
                hospital_atlas_data
              WHERE
                hospital_atlas_data.year = 2017
              LIMIT
                10;
            """

    # Pull data from the database
    A.pull(my_query)

    # Build referral networks
    A.build_networks()

    # Compare networks based on Forman curvature
    A.compare(measure="forman_curvature")

    # Embed networks into a lower-dimensional space
    A.embed()

    # Cluster networks based on structural similarity
    A.cluster_networks()

Working with a Local Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're in an environment with connectivity issues, firewall restrictions, or need offline access, you can download the database and run a local Datasette instance:

.. code-block:: python

    from apparent import Apparent
    from apparent.utils import download_and_launch_local_datasette, stop_local_datasette

    # Download the database and start a local Datasette server
    local_url = download_and_launch_local_datasette(verbose=True)

    print(f"Local Datasette server running at: {local_url}")
    # Now you can use Apparent as normal, it will automatically use the local URL
    app = Apparent(base_url=local_url)

    # Simple sample query that mirrors the test patterns
    # This gets basic network info for small networks from 2017
    query = """
    SELECT
        hospital_atlas_data.hsa,
        hospital_atlas_data.year,
        hospital_atlas_data.latitude,
        hospital_atlas_data.longitude,
        referral_network_features.nnodes,
        referral_network_features.nedges,
        referral_network_features.forman_mean
    FROM
        hospital_atlas_data
        JOIN referral_network_features ON hospital_atlas_data.hsa = referral_network_features.hsa
        AND hospital_atlas_data.year = referral_network_features.year
    WHERE
        hospital_atlas_data.year = 2017
        AND referral_network_features.nnodes > 10
        AND referral_network_features.nnodes < 50
        AND referral_network_features.nedges < 200
    ORDER BY
        referral_network_features.nnodes
    LIMIT 5;
    """

    app.pull(query)
    print(f"Retrieved {len(app.data)} networks")
    print(app.data.head())

    # Stop the local Datasette server when done
    stop_local_datasette(port=8001)
 

Contributing
------------

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (``git checkout -b feature-name``).
3. Commit changes (``git commit -m 'Add feature'``).
4. Push to your branch (``git push origin feature-name``).
5. Open a Pull Request.

Testing
-------

This project uses ``pytest`` for testing. The tests are divided into two categories: ``unit`` and ``integration``.

Unit Tests
~~~~~~~~~~

Unit tests run against the live ``apparent.topology.rocks`` service. These tests are run automatically in CI on pushes to ``main`` and ``develop``.

To run the unit tests locally, you will need to set the ``APPARENT_URL`` environment variable in a ``.env`` file in the root of the project:

.. code-block:: bash

    echo APPARENT_URL="https://apparent.topology.rocks/us_physician_referral_networks.csv" >> .env

Then, you can run the unit tests:

.. code-block:: bash

    pytest -m unit

.. warning::
    The remote service has the following known limitations:

    - It is not possible to pull the largest networks.
    - There may be HTTP errors for oversized queries.

Integration Tests
~~~~~~~~~~~~~~~~~~~

You can run integration tests in two ways:

**Option 1: Using the helper script**

A script is provided to simplify running the integration tests. This script handles:

1. Downloading the raw dataset (under ``data/us_physician_referral_networks.db``).
2. Launching a local Datasette server.
3. Executing the integration test suite.

.. warning::
   The dataset is large (approximately 8 GB) and may take considerable time to download depending on your internet speed.

To execute the script, run the following command from the root directory:

.. code-block:: bash

    bash tests/run-integration-tests.sh

**Option 2: Using the Python API**

You can also use the Python API to set up the local database and run tests:

.. code-block:: python

    from apparent.utils import download_and_launch_local_datasette, stop_local_datasette
    import subprocess

    # Download DB and start Datasette
    local_url = download_and_launch_local_datasette(
        db_path="data/us_physician_referral_networks.db",
        port=8001,
        update_env=True
    )

    # Run integration tests
    subprocess.run(["python", "-m", "pytest", "tests/", "-v", "-m", "integration"])

    # Stop the local Datasette server when done
    stop_local_datasette(port=8001)




License
-------

This project is licensed under the BSD-3 License. See the LICENSE file for details.

Contact
-------

For questions, feedback, or collaboration opportunities please contact the AIODS Lab.

Table of Contents
------------------


.. toctree::
   :maxdepth: 2
   :caption: APPARENT

   apparent

.. toctree::
   :maxdepth: 2
   :caption:  -> Networks
   
   networks/build
   networks/cluster
   networks/compare
   networks/describe
   networks/embed

|

.. image:: _static/aidos_logo.png
   :width: 120
   :height: 120
   :alt: SCOTT logo
   :align: left


|
| **Interested in more of our work?**
|
| See what we are working on at `AIDOS Lab <https://aidos.group>`_ or check out our `GitHub <https://github.com/aidos-lab>`_.


