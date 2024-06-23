from transformer import FeatureExtractor, TransformerModel, extract_features
from display_binvox import display_binvox
import torch
import os


# 初始化模型
print("初始化模型....")
feature_extractor = FeatureExtractor()
transformer_model = TransformerModel()


print("設定路徑....")
base = "/Users/zhangyuchen/Documents/ws/IP_topic"
# base = "/home/mrl/ws/IP_topic"
datas_dir = "unitest/unit_input"
floder_name = "test_data"

output_dir = "unitest/unit_output"

feature_extractor_path = base+'/'+output_dir+'/'+'feature_extractor.pth'
transformer_model_path = base+'/'+output_dir+'/'+'transformer_model.pth'


# 加載模型參數
print("加載模型參數....")
feature_extractor.load_state_dict(torch.load(feature_extractor_path))
transformer_model.load_state_dict(torch.load(transformer_model_path))

# 將模型設置為評估模式
feature_extractor.eval()
transformer_model.eval()


print("輸入圖片....")
os.chdir(base+"/"+datas_dir+"/"+floder_name+"/"+"models/picture")

image_dict = {i:'_r_'+str(i*15).zfill(3)+'.png' for i in range(24)} # for example 0:'_r_000.png', 5:'_r_075.png'
image_paths = [image_dict[i]for i in list(range(24))]#sorted(random.choices(list(range(24)),k=24))]

# 提取特徵

print("提取特徵....")
image_features = [extract_features(image_path) for image_path in image_paths]
image_features = torch.cat(image_features, dim=0)  # 合併為一個 tensor，形狀為 (M, 1280)
print("=================Image Features =================")
print(image_features)
print(image_features.shape)

print("初始化decoder輸入...")
tgt = torch.ones((1, 1, 4096))


# 測試過程（與訓練類似，但不進行梯度計算）

with torch.no_grad():
    features = feature_extractor(image_features)
    src = features.unsqueeze(1)
    output = transformer_model(src, tgt)
    print("=================nonreshared output=================")
    print(output)
    print(output.shape)
    output = output.reshape(32, 32, 32)
    # 可以進行後續的處理或評估

print("=================reshared output=================")
print(output)
print(output.shape)
voxel = output.numpy()
voxel = voxel > 0.5
display_binvox(voxel=voxel)
