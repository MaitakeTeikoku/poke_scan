import os
import requests
import logging

start_id = 1
end_id = 151

dir = "./PAGES/poke_scan/_python"

def download_images(id, dir):
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

            download_dir = f"{dir}/images/{id}"

            # 画像保存先のディレクトリを作成
            if not os.path.exists(download_dir):
                os.makedirs(download_dir, exist_ok=True)

            # ファイルの保存
            file_path = os.path.join(download_dir, f"{id}_{path['name']}.{extension}")
            with open(file_path, 'wb') as file:
                file.write(response.content)
        
        print(f"{id}")
    except Exception as e:
        # その他のエラーをログに記録
        error = f"[ERROR] download_images {id}: {e}"
        print(error)
        logging.error(error)

def main(start_id, end_id, dir):
    # ログ設定
    if not os.path.exists(f"{dir}/logs"):
        os.makedirs(f"{dir}/logs", exist_ok=True)
    logging.basicConfig(filename=f"{dir}/logs/error.log", level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(f"{dir}/images"):
        os.makedirs(f"{dir}/images", exist_ok=True)

    for id in range(start_id, end_id + 1):
        download_images(id, dir)

if __name__ == "__main__":
    main(start_id, end_id, dir)
