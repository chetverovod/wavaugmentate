#!/bin/sh
#
cd docs
make clean
make html
cd ..
git commit -am "Build docs." 
python3 -m build
git commit -am "Build package." 
git push

# Private file with tokens
shell tokens.sh

# Upload to TestPyPi
python3 -m twine upload --repository testpypi dist/* --username __token__ --password $TEST_PYPI_TOKEN --verbose
