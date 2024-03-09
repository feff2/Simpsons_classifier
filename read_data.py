from imports import *


def read_data(data_path):
    image_data = {}

    for root, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                class_name = os.path.basename(root)
                if class_name not in image_data:
                    image_data[class_name] = []
                image_data[class_name].append(os.path.join(root, file))

    df_images = pd.DataFrame(image_data.items(), columns=['Class', 'Images'])

    return df_images



