#!/bin/sh

# Update local docs
cd docs
make clean
make html
cd ..
git commit -am "Build docs." 

# Update package
cd dist
rm *.whl
rm *.gz
cd ..
python3 -m build
git commit -am "Build package." 

# Private file with tokens
source tokens.sh

# Upload to TestPyPi
python3 -m twine upload --repository testpypi dist/* --username __token__ --password $TEST_PYPI_TOKEN --verbose
