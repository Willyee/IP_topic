# /bin/bash

BASE=~/Documents/ws/IP_topic
DATAS_DIR=${BASE}/unitest/unit_input

ls ${DATAS_DIR} | while read -r line; do
    echo ${line}
    FOLDER_NAME=$line
    TARGET=$DATAS_DIR/$FOLDER_NAME/models

    if [ -f "$TARGET/model_normalized.binvox" ]; then
        echo "model_normalized.binvox exist"
    else
        echo "model_normalized.binvox does not exist"
        $BASE/scripts/binvox_tools/binvox -cb -down -down -down $TARGET/model_normalized.obj
    fi

done
