.. Eryn documentation master file, created by
   sphinx-quickstart on Thu Jul 22 20:03:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Eryn's documentation!
================================

.. image:: img/ents.jpg
  :width: 400
  :alt: Alternative text
  :align: center

.. include:: readme.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 4
   :caption: Documentation:

   user/ensemble
   user/state
   user/backend
   user/moves
   user/flows
   user/prior
   user/temper
   user/utils

.. toctree::
  :maxdepth: 4
  :caption: Tutorial:

  Eryn_tutorial
  tutorial/Group_moves
..   more_tutorials

.. toctree::
  :maxdepth: 4
  :caption: General Information:
  
  general/todos
