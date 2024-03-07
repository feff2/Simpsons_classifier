from imports import *

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'edge')


class SimpsonsDataset(Dataset):
    
    """
    Данный класс подгтавливает наш датасет к обучению:
    загружает изображений, аугментрует их, нормализует, 
    затем передает модели изображение с его меткой
    """
    
    def __init__(self, files,  mode):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode
        self.l_e = LabelEncoder()
        self.rescale_size = 244
        self.labels=[]
        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.l_e.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.l_e, le_dump_file)
        else:
            self.filenames = [path.name for path in files]
            for file in self.filenames:
                self.class_name=""
                self.parts = str(file).split('_')
                for part in self.parts:
                    if '.' in part:
                        break
                    self.class_name += part + '_'
                self.labels.append(self.class_name.rstrip('_'))
            
            with open('label_encoder.pkl', 'rb') as file:
                self.l_e = pickle.load(file)
            
            
            
    def __len__(self):
        return len(self.files)
    
    def create_transformer(self):
        if self.mode == "train":
            transformer = transforms.Compose([
                SquarePad(),
                transforms.Resize(size=(self.rescale_size, self.rescale_size)),
                transforms.RandomRotation(15),
                transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    transforms.RandomApply([transforms.RandomHorizontalFlip(
                        p=0.5), transforms.ColorJitter(contrast=0.9)], p=0.5),
                    transforms.RandomApply([transforms.RandomHorizontalFlip(
                        p=0.5), transforms.ColorJitter(brightness=0.1)], p=0.5),
                ]),
                transforms.ToTensor(),
                transforms.Normalize([0.4622, 0.4075, 0.3523], 
                                     [0.2170, 0.1963, 0.2254])
            ])
        else:
            transformer = transforms.Compose([
                transforms.Resize(size=(self.rescale_size, self.rescale_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.4622, 0.4075, 0.3523], 
                                     [0.2170, 0.1963, 0.2254])
            ])
        return transformer
    
    def load_sample(self, im_file):
        image = Image.open(im_file)
        image.load()
        return image
    
    def __getitem__(self, idx):
        transform = self.create_transformer()    
        x = self.load_sample(self.files[idx])
        x = transform(x)
        label = self.labels[idx]
        label_id = self.l_e.transform([label])
        y = label_id.item()
        return x, y