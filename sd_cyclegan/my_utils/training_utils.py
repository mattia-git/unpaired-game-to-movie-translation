import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob
import numpy as np

def parse_args_unpaired_training():
    parser = argparse.ArgumentParser()

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)

    # args for the model
    parser.add_argument('--use_feature_conditioning', action='store_true', help='Use conditional unet, integrating extra features such as pose estimation')
    parser.add_argument("--use_wrapper", action="store_true", help="Whether to use the conditional wrapper or simple UNet.")
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--validation_steps", type=int, default=500,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=5e-6,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    args = parser.parse_args()
    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T

class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()

        # Set source and target folder based on the split
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer

        # Load fixed captions for both domains
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        # Find all images in the source and target folders
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "img_path_src": img_path_src,
            "img_path_tgt": img_path_tgt,
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }

class UnpairedFeatureDataset(UnpairedDataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__(dataset_folder, split, image_prep, tokenizer)

        self.poses_folder = os.path.join(dataset_folder, "poses")
        self.l_poses = []

        self.faces_folder = os.path.join(dataset_folder, "faces")
        self.l_faces = []

        # For each l_imgs_src, and l_imgs_tgt, find the corresponding pose and face images
        self.l_poses_src, self.l_faces_src, prune_indices_src = self.get_poses_and_faces(self.l_imgs_src)
        self.l_poses_tgt, self.l_faces_tgt, prune_indices_tgt = self.get_poses_and_faces(self.l_imgs_tgt)

        for i in prune_indices_src:
            self.l_imgs_src.pop(i)
        for i in prune_indices_tgt:
            self.l_imgs_tgt.pop(i)

        self.filter_poses()
        self.filter_faces()

    def get_poses_and_faces(self, l_imgs):
        prune_indices = []
        l_poses = {}
        l_faces = {}

        for i, img_path in enumerate(l_imgs):
            img_name = os.path.basename(img_path)
            pose_path = os.path.join(self.poses_folder, img_name)
            face_path = os.path.join(self.faces_folder, img_name)

            pose_path = os.path.splitext(pose_path)[0] + ".pt"
            face_path = os.path.splitext(face_path)[0] + ".npy"

            if not os.path.exists(pose_path) or not os.path.exists(face_path):
                prune_indices.append(i)

            l_poses[img_path] = pose_path
            l_faces[img_path] = face_path

        return l_poses, l_faces, prune_indices
    
    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
            - "features_src": The source domain's features (pose and face)
            - "features_tgt": The target domain's features (pose and face)
        """

        base_dict = super().__getitem__(index)

        img_src_path = base_dict["img_path_src"]
        img_tgt_path = base_dict["img_path_tgt"]

        pose_src_path = self.l_poses_src[img_src_path]
        face_src_path = self.l_faces_src[img_src_path]

        pose_tgt_path = self.l_poses_tgt[img_tgt_path]
        face_tgt_path = self.l_faces_tgt[img_tgt_path]

        # pose is .pt
        pose_src = torch.load(pose_src_path)
        pose_tgt = torch.load(pose_tgt_path)

        # face is .npy
        face_src = np.load(face_src_path, allow_pickle=True).item()
        face_tgt = np.load(face_tgt_path, allow_pickle=True).item()

        if face_src is None:
            face_src = {
                'boxes': np.zeros((1, 4)),
            }

        if face_tgt is None:
            face_tgt = {
                'boxes': np.zeros((1, 4)),
            }

        face_src = face_src['boxes']
        face_tgt = face_tgt['boxes']

        if face_src is None:
            face_src = np.zeros((1, 4))
        
        if face_tgt is None:
            face_tgt = np.zeros((1, 4))

        face_src = torch.from_numpy(face_src.astype(np.float32))
        face_tgt = torch.from_numpy(face_tgt.astype(np.float32))

        # Squeeze batch dimension
        pose_src = pose_src.squeeze(0)
        pose_tgt = pose_tgt.squeeze(0)
        face_src = face_src.squeeze(0)
        face_tgt = face_tgt.squeeze(0)

        # Convert pose and face to maps
        pose_src = pose_to_map(pose_src)
        pose_tgt = pose_to_map(pose_tgt)
        face_src = face_to_map(face_src)
        face_tgt = face_to_map(face_tgt)

        # merge features
        features_src = torch.stack([pose_src, face_src], dim=0)
        features_tgt = torch.stack([pose_tgt, face_tgt], dim=0)

        return {
            **base_dict,
            "features_src": features_src,
            "features_tgt": features_tgt
        }
    
def pose_to_map(pose):
    try:
        map = torch.zeros((256, 256))

        if pose.ndim == 3:
            pose = pose[0]

        num_joints = pose.shape[0]
        for i in range(num_joints):
            x, y = pose[i]
            x = int(x)
            y = int(y)

            x = max(0, min(x, 255))
            y = max(0, min(y, 255))

            map[x, y] = i + 1

        return map
    except:
        return torch.zeros((256, 256))

def face_to_map(face):
    try:
        map = torch.zeros((256, 256))

        if face.ndim == 2:
            face = face[0]

        if face.shape != (4,):
            return map

        x1, y1, x2, y2 = face

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        x1 = max(0, min(x1, 255))
        y1 = max(0, min(y1, 255))
        x2 = max(0, min(x2, 255))
        y2 = max(0, min(y2, 255))

        map[x1:x2, y1:y2] = 1

        return map
    except:
        return torch.zeros((256, 256))
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, image_prep):
        super().__init__()

        self.imgs = []
        patches_dir = os.path.join(dataset_folder, "patches")
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.imgs.extend(glob(os.path.join(patches_dir, ext)))

        # Find corresponding poses and faces
        self.poses_folder = os.path.join(dataset_folder, "poses")
        self.faces_folder = os.path.join(dataset_folder, "faces")

        self.poses = {}
        self.faces = {}

        for img_path in self.imgs:
            img_name = os.path.basename(img_path)
            pose_path = os.path.join(self.poses_folder, img_name)
            face_path = os.path.join(self.faces_folder, img_name)

            pose_path = os.path.splitext(pose_path)[0] + ".pt"
            face_path = os.path.splitext(face_path)[0] + ".npy"

            if not os.path.exists(pose_path) or not os.path.exists(face_path):
                print(f'[DATALOADER WARNING] Pose or face not found for {img_path}')
                continue

            self.poses[img_path] = pose_path
            self.faces[img_path] = face_path

        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_pil = Image.open(img_path).convert("RGB")
        img_t = F.to_tensor(self.T(img_pil))
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        pose_path = self.poses[img_path]
        face_path = self.faces[img_path]

        pose = torch.load(pose_path)

        if pose is None:
            pose = torch.zeros((1, 17, 2))
        elif pose.shape != (1, 17, 2):
            pose = torch.zeros((1, 17, 2))

        face = np.load(face_path, allow_pickle=True).item()

        if face is None:
            face = {
                'boxes': np.zeros((1, 4)),
            }

        face = face['boxes']

        if face is None:
            face = np.zeros((1, 4))

        face = torch.from_numpy(face.astype(np.float32))

        # Squeeze batch dimension
        pose = pose.squeeze(0)
        face = face.squeeze(0)

        # Convert pose and face to maps
        pose = pose_to_map(pose)
        face = face_to_map(face)

        # merge features
        features = torch.stack([pose, face], dim=0)

        return {
            "img": img_t,
            "img_path": img_path,
            "features": features
        }