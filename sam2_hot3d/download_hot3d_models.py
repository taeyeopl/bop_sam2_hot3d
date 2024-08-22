import requests
import os

# 기본 URL
base_url = "https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/hot3d/object_models_eval/"

# 다운로드할 파일 목록
files = [
    "models_info.json",
    "obj_000001.glb", "obj_000002.glb", "obj_000003.glb", "obj_000004.glb",
    "obj_000005.glb", "obj_000006.glb", "obj_000007.glb", "obj_000008.glb",
    "obj_000009.glb", "obj_000010.glb", "obj_000011.glb", "obj_000012.glb",
    "obj_000013.glb", "obj_000014.glb", "obj_000015.glb", "obj_000016.glb",
    "obj_000017.glb", "obj_000018.glb", "obj_000019.glb", "obj_000020.glb",
    "obj_000021.glb", "obj_000022.glb", "obj_000023.glb", "obj_000024.glb",
    "obj_000025.glb", "obj_000026.glb", "obj_000027.glb", "obj_000028.glb",
    "obj_000029.glb", "obj_000030.glb", "obj_000031.glb", "obj_000032.glb",
    "obj_000033.glb"
    ]

# 다운로드 폴더 생성
download_folder = "./hot3d_object_models_eval"
os.makedirs(download_folder, exist_ok=True)

# 각 파일 다운로드
for file in files:
    url = base_url + file
    response = requests.get(url)

    if response.status_code == 200:
        file_path = os.path.join(download_folder, file)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file}")
    else:
        print(f"Failed to download: {file}")

print("Download completed.")