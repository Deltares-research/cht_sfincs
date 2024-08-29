# Coastal Hazards Toolkit - SFINCS

Welcome to the GitHub page of the Deltares CHT SFINCS.

This package contains various scripts for reading, writing and manipulating SFINCS schematizations. It package will at some point be completely superceded by HydroMT_SFINCS.

For an editable package, install the package with e.g.:

cd d:\checkouts\github\cht_sfincs
pip install -e .

or something like:

pip install -e d:\checkouts\github\cht_sfincs



To build the cht_sfincs package, open Anaconda Powershell.

To make sure you have the latest version of build:

python -m pip install --upgrade build

and to build the package, e.g.: 

cd d:\checkouts\github\cht_sfincs

python -m build

Upload to Pypi with:

cd d:\checkouts\github\cht_sfincs

python -m pip install --upgrade twine

python -m twine upload dist/*
