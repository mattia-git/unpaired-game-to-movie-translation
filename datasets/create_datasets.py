import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

class FrameToPIL:
    def __call__(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

class SquarePad:
    def __call__(self, image):
        width, height = image.size
        max_dim = max(width, height)
        pad = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        pad.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
        return pad

class FrameDataset(Dataset):
    def __init__(self, video_path, transform=None):
        super(FrameDataset, self).__init__()

        self.cap = cv2.VideoCapture(video_path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.transform = transform

    def __len__(self):
        return self.length
    
    def _transform(self, frame):
        return self.transform(frame) if self.transform else frame

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"Could not read frame {idx}")

        return self._transform(frame)

    def __str__(self):
        return f"FrameDataset: {self.length} frames at {self.fps} fps"
    
class PatchDataset(Dataset):
    def __init__(self, video_path, transform=None):
        super(PatchDataset, self).__init__()

        # Crawl video directory
        patch_files = []

        for root, dirs, files in os.walk(video_path):
            for file in files:
                if file.endswith('.png'):
                    patch_files.append(os.path.join(root, file))

        self.patch_files = patch_files
        self.length = len(patch_files)
        self.transform = transform

    def __len__(self):
        return self.length
    
    def _transform(self, frame):
        return self.transform(frame) if self.transform else frame

    def __getitem__(self, idx):
        patch_path = self.patch_files[idx]
        patch = Image.open(patch_path)

        return {
            'patch': self._transform(patch),
            'patch_path': patch_path
        }

    def __str__(self):
        return f"PatchDataset: {self.length} patches"

def create_frame_dataset(video_path, dim=(256, 256)):
    transform = Compose([
        FrameToPIL(),
        SquarePad(),
        Resize(dim),
        ToTensor()
    ])

    return FrameDataset(video_path, transform)

def create_patch_dataset(video_path, dim=(256, 256)):
    transform = Compose([
        SquarePad(),
        Resize(dim),
        ToTensor()
    ])

    return PatchDataset(video_path, transform)