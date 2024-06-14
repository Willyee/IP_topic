import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import os
from display_binvox import read_binvox
import matplotlib.pyplot as plt

# EfficientNet-b0 模型
efficientnet = models.efficientnet_b0(pretrained=True)
efficientnet.classifier = nn.Identity()  # 移除最後的分類層

# 定義輸入圖像的轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = efficientnet(image)
    return features  # 返回的是 1280 維的特徵向量

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=1280, output_dim=4096):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.fc(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim=4096, num_heads=8, num_layers=6, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.Sigmoid()
        )
    
    def forward(self, src, tgt):
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)
        return output



feature_extractor = FeatureExtractor()
transformer_model = TransformerModel()


# 假設我們有 M 張照片，ground truth 為 (32, 32, 32) 的 Voxel 模型
# 輸入照片的路徑
base = "/Users/zhangyuchen/Documents/ws/IP_topic"
datas_dir = "unitest/unit_input"
floder_name = "test_data"
os.chdir(base+"/"+datas_dir+"/"+floder_name+"/"+"models/picture")
image_paths = ['_r_000.png', '_r_015.png', '_r_030.png', '_r_045.png', '_r_060.png', '_r_075.png', '_r_090.png', '_r_105.png']

# 提取特徵
image_features = [extract_features(image_path) for image_path in image_paths]
image_features = torch.cat(image_features, dim=0)  # 合併為一個 tensor，形狀為 (M, 1280)

# 轉換特徵
features = feature_extractor(image_features)  # 形狀為 (M, 4096)

# ground truth Voxel 模型
ground_truth_voxel = torch.from_numpy(read_binvox(base,datas_dir,floder_name))


# 準備 transformer 的輸入
src = features.unsqueeze(1)  # 形狀為 (M, 1, 4096)
tgt_start = torch.zeros((1, 1, 4096))  # 預設為 8 個初始的解碼輸入，形狀為 (8, 1, 4096)
tgt_ground_truth = ground_truth_voxel
tgt_ground_truth = torch.reshape(tgt_ground_truth,(8, 1, 4096))
tgt = torch.cat([tgt_start, tgt_ground_truth], dim=0)  # 形狀為 (9, 1, 4096)

# 訓練過程
optimizer = optim.Adam(list(feature_extractor.parameters()) + 
                       list(transformer_model.parameters()), lr=1e-4)
criterion = nn.BCELoss()

num_epochs = 20  # 訓練的總迭代次數

for epoch in range(num_epochs):
    feature_extractor.train()
    transformer_model.train()
    optimizer.zero_grad()
    features = feature_extractor(image_features)  # 確保在每個epoch中重新計算特徵
    src = features.unsqueeze(1) 
    output = transformer_model(src, tgt)  # 形狀為 (9, 1, 4096)
    output = output.squeeze(1)  # 形狀為 (9, 4096)

    # 將輸出移除最後一層
    output = output[:-1]  # 形狀為 (8, 4096)

    # 將輸出重塑為 (32, 32, 4) 並比較 ground truth
    output = output.view(8, 32, 32, 4).permute(1, 2, 0, 3).reshape(32, 32, 32)
    loss = criterion(output, ground_truth_voxel)
    loss.backward(retain_graph=True)  # 確保在需要時保留計算圖
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 測試過程（與訓練類似，但不進行梯度計算）
transformer_model.eval()
with torch.no_grad():
    output = transformer_model(src, tgt)
    output = output[:-1]  # 形狀為 (8, 4096)
    output = output.squeeze(1).view(8, 32, 32, 4).permute(1, 2, 0, 3).reshape(32, 32, 32)
    # 可以進行後續的處理或評估
    print(output)
    voxel = output.numpy()
    voxel = voxel > 0.5
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel, edgecolor='k')

    plt.show()