import os
import shutil
import requests
import logging

start_id = 1
end_id = 1025

dir = "./PAGES/poke_scan/_python/poke_scan_types"

# 画像をコピーする関数
def copy_image(id, dir, type_name):
    try:
        image_dir = f"./PAGES/poke_scan/_python/images_all/{id}"

        if not os.path.exists(image_dir):
            raise Exception(f"No. {id} のフォルダが見つかりません: {image_dir}")
        
        # フォルダ内のすべてのファイルを取得
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png'))]

        if not image_files:
            raise Exception(f"No. {id} の画像がフォルダ内に見つかりません: {image_dir}")

        for image_file in image_files:
            download_dir = f"{dir}/types_images_1/{type_name}"

            # 画像保存先のディレクトリを作成
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            # コピー元ファイルパスとコピー先ファイルパスを生成
            source_file_path = os.path.join(image_dir, image_file)
            target_file_path = os.path.join(download_dir, image_file)

            # ファイルをコピー
            shutil.copy(source_file_path, target_file_path)
        print(f"{id}")
    except Exception as e:
        error = f"[ERROR] copy_image {id}: {e}"
        print(error)
        logging.error(error)

def download_data(id, dir):
    try:
        # PokeAPIからポケモン情報を取得
        url = f"https://pokeapi.co/api/v2/pokemon/{id}"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        types = data.get("types")
        type_name = types[0]["type"]["name"]

        copy_image(id, dir, type_name)
    
    except Exception as e:
        # その他のエラーをログに記録
        error = f"[ERROR] download_data {id}: {e}"
        print(error)
        logging.error(error)

def main(start_id, end_id, dir):
    # ログ設定
    if not os.path.exists(f"{dir}/logs"):
        os.makedirs(f"{dir}/logs")
    logging.basicConfig(filename=f"{dir}/logs/error.log", level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(f"{dir}/types_images_1"):
        os.makedirs(f"{dir}/types_images_1")

    for id in range(start_id, end_id + 1):
        download_data(id, dir)

if __name__ == "__main__":
    main(start_id, end_id, dir)
