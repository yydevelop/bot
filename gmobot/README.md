# 機械学習によるビットコイン自動売買ボット<br>-richmanbtcさんチュートリアル版-

richmanbtcさんが作成された機械学習ボットチュートリアルで作成されたモデルを利用したビットコインの自動売買ボットです。  
https://github.com/richmanbtc/mlbot_tutorial

## 起動方法

### 前提条件

richmanbtcさんが作成された機械学習ボットチュートリアルが実行できていること

### GMOコインでAPIキーとシークレットキーを取得

API取得方法は以下を参照  
https://coin.z.com/jp/corp/product/info/api/

### モデルの作成と配置

 機械学習ボットチュートリアルで作成したLightGBMのモデルである model_y_buy.xzとmodel_y_sell.xz をmodelフォルダ
 に格納。

### 以下の形式で本フォルダ直下に.envファイルを作成

```
API_KEY={GMOコインで取得したAPIキー}
SECRET_KEY={GMOコインで取得したシークレットキー}
LOT=0.01 //発注ロット
MAX_LOT=0.03 //最大ロット
INTERVAL=15(売買間隔)
```

### dockerビルド
```bash
docker build . -t gmobot
```

### dockerコンテナ起動
```bash
docker run -d -it --env-file=.env --name=gmobot gmobot
```

### 起動状況確認
```bash
docker container logs -t gmobot
```

### dockerコンテナの停止・削除
```bash
docker stop gmobot
docker container prune
```
