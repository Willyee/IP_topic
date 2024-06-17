# /bin/bash

BASE="/home/mrl/ws/IP_topic/input/datas"
DATA_DIR="models"
OBJ_NAME="model_normalized.obj"
cd ${BASE}
pwd
ls | while read line; do
    echo $line
    cd $line
    cd $DATA_DIR
    # ## The `-d` test command option see if FILE exists and is a directory ##
    # [ -d "picture" ] && echo "Directory picture exists."
    # ## OR ##
    # [ ! -d "picture" ] && cd ${BASE} && rm -rf $line
    ~/binvox -cb -down -down -down $OBJ_NAME
    pwd
    cd ${BASE}
    pwd
done
# -