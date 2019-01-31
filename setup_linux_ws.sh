 #!/bin/bash
set -x
virtualenv -p /usr/bin/python2.7 venv
source ./venv/bin/activate
python ./venv/bin/pip install --upgrade pip
python ./venv/bin/pip install -r ./requirements.txt