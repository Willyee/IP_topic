import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import os
from display_binvox import read_binvox, display_binvox
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EfficientNet-b0 模型
# efficientnet = models.efficientnet_b0(pretrained=True)
print("載入EfficientNet-b0 預訓練模型參數")
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

class PicturePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PicturePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float() * math.pi * 2 / max_len
        div_term = torch.arange(0, d_model, 2).float() * math.pi * 2 / d_model
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
        self.pos_encoder = PicturePositionalEncoding(input_dim, max_len=24)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, 32*32*32),
            nn.Sigmoid()
        )
    
    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)
        return output


if __name__ == "__main__":
    # 初始化模型
    print("初始化模型....")
    feature_extractor = FeatureExtractor()
    transformer_model = TransformerModel()


    # 假設我們有 M 張照片，ground truth 為 (32, 32, 32) 的 Voxel 模型
    # 輸入照片的路徑
    print("設定路徑....")
    base = "/Users/zhangyuchen/Documents/ws/IP_topic"
    # base = "/home/mrl/ws/IP_topic"
    datas_dir = "unitest/unit_input"
    floder_name = "test_data"

    output_dir = "unitest/unit_output"

    feature_extractor_path = base + '/' + output_dir + '/' + 'feature_extractor.pth'
    transformer_model_path = base + '/' + output_dir + '/' + 'transformer_model.pth'

    os.chdir(base + "/" + datas_dir + "/" + floder_name + "/" + "models/picture")
    
    print("輸入圖片....")
    image_dict = {i: '_r_' + str(i * 15).zfill(3) + '.png' for i in range(24)}  # 例如 0:'_r_000.png', 5:'_r_075.png'
    image_paths = [image_dict[i] for i in list(range(24))]  # sorted(random.choices(list(range(24)),k=24))]

    # 提取特徵
    print("將圖片轉換成CNN模型預設輸入....")
    image_features = [extract_features(image_path) for image_path in image_paths]
    image_features = torch.cat(image_features, dim=0)  # 合併為一個 tensor，形狀為 (M, 1280)

   
    # 準備 transformer 的輸入
    print("初始化decoder輸入...")
    tgt = torch.ones((1, 1, 4096))

    # ground truth Voxel 模型
    print("讀取decoder輸出groud ture...")
    ground_truth_voxel = torch.from_numpy(read_binvox(base, datas_dir, floder_name))


    # 訓練過程
    print("設定優化器與損失函數...")
    optimizer = optim.Adam(list(feature_extractor.parameters()) + 
                        list(transformer_model.parameters()), lr=1e-4)
    schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.BCELoss()

    print("開始訓練...")
    num_epochs = 10  # 訓練的總迭代次數
    for epoch in range(num_epochs):
        print("========Epoch:========", epoch+1)
        feature_extractor.train()
        transformer_model.train()
        optimizer.zero_grad()
        print("提取圖片特徵....")
        features = feature_extractor(image_features)  # 確保在每個 epoch 中重新計算特徵
        src = features.unsqueeze(1)  # 形狀為 (M, 1, 4096)
        print("訓練transformer....")
        output = transformer_model(src, tgt)  # 形狀為 (9, 1, 4096)

        # 將輸出重塑為 (32, 32, 32) 並比較 ground truth
        output = output.reshape(32, 32, 32)
        print("計算損失....")
        loss = criterion(output, ground_truth_voxel)
        print("反向傳播....")
        loss.backward(retain_graph=True)  # 確保在需要時保留計算圖
        print("優化....")
        optimizer.step()
        schedule.step()  # 調整學習率
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}")

    # 測試過程（與訓練類似，但不進行梯度計算）
    transformer_model.eval()
    with torch.no_grad():
        print("測試transformer....")
        output = transformer_model(src, tgt)
        output = output.reshape(32, 32, 32)
        # 可以進行後續的處理或評估
        print(output)
        voxel = output.numpy()
        voxel = voxel > 0.5
        print("展示....(記得關閉視窗後才能繼續執行下一步，下一步是保存模型參數)")
        display_binvox(voxel=voxel)

    # 保存模型參數
    print("保存模型參數....")
    torch.save(feature_extractor.state_dict(), feature_extractor_path)
    torch.save(transformer_model.state_dict(), transformer_model_path)
