import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import os
from display_binvox import read_binvox,display_binvox
import math


# EfficientNet-b0 模型
# efficientnet = models.efficientnet_b0(pretrained=True)
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim=4096, num_heads=8, num_layers=6, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len=9)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward,batch_first=True)
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
        tgt = self.pos_encoder(tgt)
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

# ground truth Voxel 模型
ground_truth_voxel = torch.from_numpy(read_binvox(base,datas_dir,floder_name))

# 準備 transformer 的輸入
tgt_start = torch.ones((1, 1, 4096))
tgt_ground_truth = ground_truth_voxel.permute(2,0,1).reshape(8,1,4096)
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
    src = features.unsqueeze(1) # 形狀為 (M, 1, 4096)
    output = transformer_model(src, tgt)  # 形狀為 (9, 1, 4096)
    output = output.squeeze(1)  # 形狀為 (9, 4096)

    # 將輸出移除最後一層
    output = output[:-1]  # 形狀為 (8, 4096)

    # 將輸出重塑為 (32, 32, 4) 並比較 ground truth
    output = output.reshape(32, 32, 32).permute(1, 2, 0)
    loss = criterion(output, ground_truth_voxel)
    loss.backward(retain_graph=True)  # 確保在需要時保留計算圖
    optimizer.step()
    
    if epoch % 10 == 0:
        i = epoch // 10
        exp_lr = -4 - 2*i
        optimizer = optim.Adam(list(feature_extractor.parameters()) + 
                               list(transformer_model.parameters()), lr=10**exp_lr)
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 測試過程（與訓練類似，但不進行梯度計算）
transformer_model.eval()
with torch.no_grad():
    output = transformer_model(src, tgt)
    output = output[:-1]  # 形狀為 (8, 4096)
    output = output.reshape(32, 32, 32).permute(1, 2, 0)
    # 可以進行後續的處理或評估
    print(output)
    voxel = output.numpy()
    voxel = voxel > 0.5
    display_binvox(voxel=voxel)
