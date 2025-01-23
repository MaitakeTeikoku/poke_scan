import os
import shutil
import requests
import logging

start_id = 1
end_id = 1025

dir = "./PAGES/poke_scan/_python/poke_scan_types"

# 画像をコピーする関数
def copy_image(id, dir, types):
    try:
        image_dir = f"./PAGES/poke_scan/_python/images/{id}"

        if not os.path.exists(image_dir):
            raise Exception(f"No. {id} のフォルダが見つかりません: {image_dir}")
        
        # フォルダ内のすべてのファイルを取得
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png'))]

        if not image_files:
            raise Exception(f"No. {id} の画像がフォルダ内に見つかりません: {image_dir}")

        for image_file in image_files:
            for type_info in types:
                type_name = type_info["type"]["name"]
                download_dir = f"{dir}/types_images/{type_name}"

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
        error = f"[ERROR] download_image {id}: {e}"
        print(error)
        logging.error(error)

def download_data(id, dir):
    try:
        # PokeAPIからポケモン情報を取得
        url = f"https://pokeapi.co/api/v2/pokemon/{id}"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        #sprites = data.get("sprites")
        types = data.get("types")

        copy_image(id, dir, types)
        #download_images(sprites, dir, id, types)
    
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

    if not os.path.exists(f"{dir}/types_images"):
        os.makedirs(f"{dir}/types_images")

    for id in range(start_id, end_id + 1):
        download_data(id, dir)

if __name__ == "__main__":
    main(start_id, end_id, dir)


'''
# 画像をダウンロードする関数
def download_image(url, name, dir, types):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # 画像の拡張子を取得
        extension =url.split('.')[-1]

        for type_info in types:
            type_name = type_info["type"]["name"]
            download_dir = f"{dir}/types_images/{type_name}"

            # 画像保存先のディレクトリを作成
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            # ファイルの保存
            file_path = os.path.join(download_dir, f"{name}.{extension}")
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"{name}: {type_name}")
    except requests.exceptions.RequestException as e:
        error = f"[ERROR] download_image {name}: {e}"
        print(error)
        logging.error(error)

def download_images(sprites, dir, id, types):
    try:
        if isinstance(sprites, dict):
            for index, (key, value) in enumerate(sprites.items()):
                if isinstance(value, dict):
                    # 値が辞書なら再帰的に処理
                    download_images(value, dir, f"{id}_{index}", types)
                elif isinstance(value, str) and value.endswith(('.png')) and not any(x in value for x in ['shiny', 'gray']):
                    # URLが文字列であればダウンロード
                    download_image(value, f"{id}_{index}_{key}", dir, types)

    except Exception as e:
        # その他のエラーをログに記録
        error = f"[ERROR] download_images {id}: {e}"
        print(error)
        logging.error(error)
'''
