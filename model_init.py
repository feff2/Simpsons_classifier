from imports import *

def set_parameter_requires_grad(model, feature_extracting = False):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(df_images):
    model = None
    images_lst = sorted(df_images['Images'].tolist())
    train_val_files_lst = [Path(item) for sublist in images_lst for item in sublist]
    train_val_labels = [path.parent.name for path in train_val_files_lst]
    num_classes = len(pd.unique(train_val_labels))
    model = models.resnet50()
    set_parameter_requires_grad(model)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model