import os
import shutil
import requests
import logging

start_id = 1
end_id = 1025

dir = "./PAGES/poke_scan/_python/poke_scan_types"

# 画像をコピーする関数
def download_image(id, dir, types):
    try:
        image_paths = [
            {
                "name": "officialArtwork",
                "path": "/other/official-artwork",
            },
            {
                "name": "front",
                "path": "",
            },
            {
                "name": "back",
                "path": "/back",
            },
            {
                "name": "home",
                "path": "/other/home",
            }
        ]
        
        for path in image_paths:
            image_url = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon{path['path']}/{id}.png"
            
            response = requests.get(image_url)
            if response.status_code == 404:
                continue

            # 画像の拡張子を取得
            extension =image_url.split('.')[-1]

            for type_info in types:
                type_name = type_info["type"]["name"]
                download_dir = f"{dir}/types_images/{type_name}"

                # 画像保存先のディレクトリを作成
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir, exist_ok=True)

                # ファイルの保存
                file_path = os.path.join(download_dir, f"{id}_{path['name']}.{extension}")
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"{id}: {type_name}")

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

        download_image(id, dir, types)
        #download_images(sprites, dir, id, types)
    
    except Exception as e:
        # その他のエラーをログに記録
        error = f"[ERROR] download_data {id}: {e}"
        print(error)
        logging.error(error)

def main(start_id, end_id, dir):
    # ログ設定
    if not os.path.exists(f"{dir}/logs"):
        os.makedirs(f"{dir}/logs", exist_ok=True)
    logging.basicConfig(filename=f"{dir}/logs/error.log", level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(f"{dir}/types_images"):
        os.makedirs(f"{dir}/types_images", exist_ok=True)

    for id in range(start_id, end_id + 1):
        download_data(id, dir)

if __name__ == "__main__":
    main(start_id, end_id, dir)
