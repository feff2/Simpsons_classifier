from imports import *

def search_dub(data_path):
    image_hashes = {}
    duplicates = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                with open(os.path.join(root, file), 'rb') as f:
                    image_data = f.read()
                image_hash = hashlib.md5(image_data).hexdigest()
                if image_hash in image_hashes:
                    duplicates.append({"Original": image_hashes[image_hash], "Duplicate": os.path.join(root, file)})
                else:
                    image_hashes[image_hash] = os.path.join(root, file)

    df_duplicates = pd.DataFrame(duplicates)
    return df_duplicates


