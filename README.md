# IP_topic
由於尚未完成專案所以程式仍放在unitest

## download
將城市複製到您指定的位置
```
https://github.com/Willyee/IP_topic.git
cd IP_topic
```

在我的專案使用了<br>
[panmari/stanford-shapenet-renderer](https://github.com/panmari/stanford-shapenet-renderer)<br>
[dimatura/binvox-rw-py](https://github.com/dimatura/binvox-rw-py)<br>
但由於執行程式只需使用binvox-rw-py，所以我只帶您設定binvox-rw-py<br>
移動到lib -><br>
刪除資料夾並複製依賴工具 -><br>
更改名稱(方便之後python呼叫) -><br>
改動branch(因為我有更動一些東西)
```
cd lib
rm -rf binvox_rw_py
https://github.com/Willyee/binvox-rw-py.git
mv binvox-rw-py binvox_rw_py
cd binvox_rw_py
git checkout IP_topic_use
```

移動到根目錄進行初始化、並且安裝與我相符的python套件版本
```
cd ../../ # 您現在應該在IP_topic
pip install -r requirements.txt
python3 setup.py install --user
```

## usage
由於模型的參數太大了，並且我這個並非完成的模型，所以我並未上傳模型參數
![截圖 2024-06-23 10.41.49](https://hackmd.io/_uploads/rkcYKFB8A.png)
所以我讓您自行訓練模型參數，完整訓練完大該花費5-10分鐘，因為只有放一個訓練資料而已

```
cd unitest 
python3 transformer.py # 訓練模型
python3 test.py # 使用模型
```

