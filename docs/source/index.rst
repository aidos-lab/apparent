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
~~~~~~~~~~~~~~~~

A script is provided to simplify running the integration tests. This script handles downloading the raw dataset, launching a local Datasette server, and executing the integration test suite.

.. warning::
    The dataset is large (approximately 8 GB) and may take considerable time to download depending on your internet speed.

Integration tests require a local instance of the ``apparent`` database. To set up and run the integration tests, follow these steps:

1. Download the raw dataset.
2. Set up a local Datasette instance.

To execute the script, use the following command:

.. code-block:: bash

    bash tests/run-integration-tests.sh




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


