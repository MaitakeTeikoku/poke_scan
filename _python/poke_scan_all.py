import os
import requests
import logging

start_id = 1
end_id = 1025

dir = "./PAGES/poke_scan/_python"

# 画像をダウンロードする関数
def download_image(url, name, download_dir):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # 画像の拡張子を取得
        extension =url.split('.')[-1]

        # ファイルの保存
        file_path = os.path.join(download_dir, f"{name}.{extension}")
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(name)
    except requests.exceptions.RequestException as e:
        print(e)
        logging.error(e)

def download_images(sprites, download_dir, id):
    try:
        if isinstance(sprites, dict):
            for index, (key, value) in enumerate(sprites.items()):
                if isinstance(value, dict):
                    # 値が辞書なら再帰的に処理
                    download_images(value, download_dir, f"{id}_{index}")
                elif isinstance(value, str) and value.endswith(('.png')) and not any(x in value for x in ['shiny', 'gray']):
                    # URLが文字列であればダウンロード
                    download_image(value, f"{id}_{index}_{key}", download_dir)

    except Exception as e:
        # その他のエラーをログに記録
        print(e)
        logging.error(e)

def download_data(id, dir):
    download_dir = f"{dir}/images_all/{id}"

    # 画像保存先のディレクトリを作成
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    try:
        # PokeAPIからポケモン情報を取得
        url = f"https://pokeapi.co/api/v2/pokemon/{id}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"No. {id} の情報を取得できませんでした。: {response.status_code}")

        data = response.json()
        sprites = data.get("sprites")

        download_images(sprites, download_dir, id)
    
    except Exception as e:
        # その他のエラーをログに記録
        print(e)
        logging.error(e)

def main(start_id, end_id, dir):
    # ログ設定
    if not os.path.exists(f"{dir}/logs"):
        os.makedirs(f"{dir}/logs")
    logging.basicConfig(filename=f"{dir}/logs/error.log", level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(f"{dir}/images_all"):
        os.makedirs(f"{dir}/images_all")

    for id in range(start_id, end_id + 1):
        download_data(id, dir)

if __name__ == "__main__":
    main(start_id, end_id, dir)
