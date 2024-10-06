#!/bin/sh


# Update package



# Private file with tokens
source tokens.sh

# Push to github
git push

# Upload to PyPi
python3 -m twine upload --repository pypi dist/* --username __token__ --password $PYPI_TOKEN --verbose
