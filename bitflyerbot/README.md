# 機械学習によるビットコイン自動売買ボット(BitFlyer用) -richmanbtcさんチュートリアル版- 

richmanbtcさんが作成された機械学習ボットチュートリアルで作成されたモデルを利用したビットコインの自動売買ボットです。  
https://github.com/richmanbtc/mlbot_tutorial

mlbotの初心者向けチュートリアル(BitFlyer版)で作成したモデルを利用しており、BitFlyerのBTC-FX/JPYのペアで動作する自動売買ボットとなります。
https://gist.github.com/amdapsi/517ddaaa6f5731d6abc995dc4d491060

## 起動方法

### 前提条件

mlbotの初心者向けチュートリアル(BitFlyer版)が実行できていること  
BitFlyerにアカウントがあること

### BitFlyerでAPIキーとシークレットキーを取得

APIについては以下を参照  
https://bitflyer.com/ja-jp/api

### モデルの作成と配置

mlbotの初心者向けチュートリアル(BitFlyer版)で作成したLightGBMのモデルである model_y_buy_bffx.xzとmodel_y_sell_bffx.xz をmodelフォルダに格納。

### 以下の形式で本フォルダ直下に.envファイルを作成

```
API_KEY={BitFlyerで取得したAPIキー}
SECRET_KEY={BitFlyerで取得したシークレットキー}
LOT=0.01 //発注ロット
MAX_LOT=0.03 //最大ロット
INTERVAL=15(売買間隔)
```

### dockerビルド
```bash
docker build . -t paibot-bffx
```

### dockerコンテナ起動
```bash
docker run -d -it --env-file=.env --name=paibot-bffx paibot-bffx
```

### 起動状況確認
```bash
docker container logs -t paibot-bffx
```

### dockerコンテナの停止・削除
```bash
docker stop paibot-bffx
docker container prune
```
