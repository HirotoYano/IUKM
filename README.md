# python environment

pythonの開発環境を構築する。

## 構成について

Dockerで構築しているため、Dockerが整備されているPCであれば同じ環境を作成することができる。
また、VScodeのdev containerを採用している。
詳しくは、[https://github.com/microsoft/vscode-dev-containers](https://github.com/microsoft/vscode-dev-containers)を確認してください。

### python

Pythonのバージョンは、```3.9.5```を採用している。
また、ライブラリ管理のツールとして、[poetry](https://github.com/python-poetry/poetry)を利用している。
poetryは、仮想環境を作成することができ、その仮想環境内でライブラリ管理をおこうことができるが、
今回作成した構成ではDocekrがあるため、仮想環境を作成する必要がない。
そのため、Docker内のLocalにライブラリをインストールする構成に鳴っている。（Docker内に仮想環境の作成は行わない）

## 実行方法について

<span style="color: red; ">実行する前に、少し設定を行う必要がある。</span>

### 前準備

#### USER IDの設定

`.devcontainer/.env`に書き込む必要がある。
`USER_UID`, `USER_GID`にPC USERのIDを書き込む必要がある。
`network setting`はProxy内で動作しているPCのみ設定が必要です。

```env
# USER setting
USER_NAME=vscode
USER_UID=1000 # ここと
USER_GID=1000 # ここ
# network setting # 必要があれば設定する
HTTP_PROXY="" 
HTTPS_PROXY=""
```

Vscodeのdev containerを利用する方法と、docker composeで起動する方法の2通りある。

### dev container

Vscodeを利用しているなら、こちらをおすすめする。

`Respon in Container`から起動することで開発環境に入ることができる。

### docker compose

以下のコマンドで起動して開発環境内に入ることができる。

``` shell
cd .devcontainer
docker compose build
docker compose up -d #このコマンドでContainerが起動する。
docker exec -it devcontainer-python_environment-1 bash
```

