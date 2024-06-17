from transformer import FeatureExtractor, TransformerModel, extract_features
from display_binvox import display_binvox
import torch
import os


# 初始化模型
feature_extractor = FeatureExtractor()
transformer_model = TransformerModel()

base = "/Users/zhangyuchen/Documents/ws/IP_topic"
# base = "/home/mrl/ws/IP_topic"
datas_dir = "unitest/unit_input"
floder_name = "test_data"

output_dir = "unitest/unit_output"

feature_extractor_path = base+'/'+output_dir+'/'+'feature_extractor.pth'
transformer_model_path = base+'/'+output_dir+'/'+'transformer_model.pth'


# 加载模型参数
feature_extractor.load_state_dict(torch.load(feature_extractor_path))
transformer_model.load_state_dict(torch.load(transformer_model_path))

# 将模型设置为评估模式
feature_extractor.eval()
transformer_model.eval()


os.chdir(base+"/"+datas_dir+"/"+floder_name+"/"+"models/picture")

image_dict = {i:'_r_'+str(i*15).zfill(3)+'.png' for i in range(24)} # for example 0:'_r_000.png', 5:'_r_075.png'
image_paths = [image_dict[i]for i in list(range(3,5))]#sorted(random.choices(list(range(24)),k=24))]

# 提取特徵
image_features = [extract_features(image_path) for image_path in image_paths]
image_features = torch.cat(image_features, dim=0)  # 合併為一個 tensor，形狀為 (M, 1280)

tgt = torch.ones((1, 1, 4096))

# 测试过程（与训练类似，但不进行梯度计算）
with torch.no_grad():
    features = feature_extractor(image_features)
    src = features.unsqueeze(1)
    output = transformer_model(src, tgt)
    output = output.reshape(32, 32, 32)
    # 可以进行后续的处理或评估

print(output)
voxel = output.numpy()
voxel = voxel > 0.3
display_binvox(voxel=voxel)