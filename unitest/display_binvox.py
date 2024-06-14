from lib.binvox_rw_py import binvox_rw
import os
import matplotlib.pyplot as plt
import numpy as np

def read_binvox(base,
                datas_dir,
                floder_name):
    os.chdir(base+"/"+datas_dir+"/"+floder_name+"/"+"models")

    with open('model_normalized.binvox', 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)

    return np.array(model.data).astype(np.float32)


if __name__ == '__main__':
    base = "/Users/zhangyuchen/Documents/ws/IP_topic"
    # os.chdir(base)
    # base_ls = os.listdir(base)
    # for file in base_ls:
    #     if os.path.isdir(base+"/"+file):
    #         print(file)
    datas_dir = "unitest/unit_input"
    floder_name = "test_data"
    voxel = read_binvox(base, datas_dir, floder_name)

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel, edgecolor='k')

    plt.show()