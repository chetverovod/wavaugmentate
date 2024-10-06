#!/bin/sh
cd docs
make clean
make html
cd ..
git commit -am "Build docs." 
