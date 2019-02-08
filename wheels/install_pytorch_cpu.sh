#!/bin/bash
set -e;

PYTORCH_WHL_PREFIX="http://download.pytorch.org/whl/cpu";
PYTHON_VERSIONS=(
  cp27-cp27mu
  cp35-cp35m
  cp36-cp36m
  cp37-cp37m
);
PYTORCH_WHEELS=(
  ${PYTORCH_WHL_PREFIX}/torch-1.0.1-cp27-cp27mu-linux_x86_64.whl
  ${PYTORCH_WHL_PREFIX}/torch-1.0.1-cp35-cp35m-linux_x86_64.whl
  ${PYTORCH_WHL_PREFIX}/torch-1.0.1-cp36-cp36m-linux_x86_64.whl
  ${PYTORCH_WHL_PREFIX}/torch-1.0.1-cp37-cp37m-linux_x86_64.whl
);

for i in $(seq ${#PYTHON_VERSIONS[@]}); do
  export PYTHON=/opt/python/${PYTHON_VERSIONS[i - 1]}/bin/python;
  $PYTHON -m pip install -r <(awk '$0 !~ /^torch/' requirements.txt);
  $PYTHON -m pip install "${PYTORCH_WHEELS[i - 1]}";
done;
