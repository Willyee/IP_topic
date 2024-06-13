import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.optim as optim

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
        self.fc_out = nn.Linear(input_dim, 4096)
    
    def forward(self, src, tgt):
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)
        return output



feature_extractor = FeatureExtractor()
transformer_model = TransformerModel()


# 假設我們有 M 張照片，ground truth 為 (32, 32, 32) 的 Voxel 模型
# 輸入照片的路徑
image_paths = ['_r_000.png', '_r_015.png', '_r_030.png', '_r_045.png', '_r_060.png', '_r_075.png', '_r_090.png', '_r_105.png']
# ground truth Voxel 模型
ground_truth_voxel = torch.rand((32, 32, 32))  # 這裡用隨機數據代替，實際應為真實的 Voxel 模型

# 提取特徵
features = [extract_features(image_path) for image_path in image_paths]
features = torch.cat(features, dim=0)  # 合併為一個 tensor，形狀為 (M, 1280)

# 轉換特徵
features = feature_extractor(features)  # 形狀為 (M, 4096)

# 準備 transformer 的輸入
src = features.unsqueeze(1)  # 形狀為 (M, 1, 4096)
tgt = torch.zeros((8, 1, 4096))  # 預設為 8 個初始的解碼輸入，形狀為 (8, 1, 4096)

# 訓練過程
optimizer = optim.Adam(transformer_model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 1  # 訓練的總迭代次數

for epoch in range(num_epochs):
    transformer_model.train()
    optimizer.zero_grad()
    output = transformer_model(src, tgt)  # 形狀為 (8, 1, 4096)
    output = output.squeeze(1)  # 形狀為 (8, 4096)

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
    print(output.size())
    output = output.squeeze(1).view(8, 32, 32, 4).permute(1, 2, 0, 3).reshape(32, 32, 32)
    # 可以進行後續的處理或評估