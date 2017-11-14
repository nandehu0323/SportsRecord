#!/usr/bin/env sh
set -e

./bin/caffe.exe train --solver=examples/mnist/lenet_solver.prototxt $@
