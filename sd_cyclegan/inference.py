import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import TestDataset
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='path to the test dataset')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. Required for custom models.')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_256x256', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    parser.add_argument('--use_feature_conditioning', action='store_true', help='whether to use feature condition')
    parser.add_argument('--use_wrapper', action='store_true', help='whether to use the wrapper')

    args = parser.parse_args()

    if args.model_path is None:
        raise ValueError('model_path must be provided')

    if args.prompt is None:
        raise ValueError('prompt is required')
    
    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = CycleGAN_Turbo(pretrained_path=args.model_path, use_feature_conditioning=args.use_feature_conditioning, use_wrapper=args.use_wrapper)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()

    dataset = TestDataset(args.dataset_dir, image_prep=args.image_prep)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for (i, batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = batch['img'].cuda()
        img_path = batch['img_path']

        if args.use_feature_conditioning:
            features = batch['features'].cuda()
        else:
            features = None

        with torch.no_grad():
            output = model(img, direction=args.direction, caption=args.prompt, features=features, use_wrapper=args.use_wrapper)

            # Post-process the output
            output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
            output_pil = output_pil.resize((*img.shape[2:],), Image.LANCZOS)

            img_name = os.path.basename(img_path[0])
            output_pil.save(os.path.join(args.output_dir, img_name))

    print(f'Inference completed. The output images are saved in {args.output_dir}')