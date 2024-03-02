# import sys,os
# sys.path.append(".")
# sys.path.append('./taming-transformers')
# from taming.models import vqgan 
# import argparse
# from ldm.modules.attention import CrossAttention

# parser = argparse.ArgumentParser()
# parser.add_argument("--output", type=str, default='run')
# parser.add_argument("--loss_type", type=str, choices=["simple", "vlb", "hybrid", "rescale", "min_snr","diff-pruning","custom"],default='simple')
# parser.add_argument("--loss_type2", type=str, default='')
# args = parser.parse_args()

#@title loading utils
import torch
# from omegaconf import OmegaConf
import numpy as np 

# from ldm.util import instantiate_from_config

# import torch_pruning as tp

# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from torchvision import utils as tvu
import torch.distributed as dist

from sparsityV2 import ASP
import torch.optim as optim

import matplotlib.pyplot as plt

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

# from ldm.models.diffusion.ddim import DDIMSampler

# model = get_model()
# model1 = get_model()


# sampler = DDIMSampler(model)

from transformers import AutoImageProcessor, ResNetForImageClassification
# import torch
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
# model1 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.cuda()
# model1.cuda()
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

 # define classes to be sampled here
n_samples_per_class = 24

# ddim_steps = 20
# ddim_eta = 0.0
# scale = 3.0   # for unconditional guidance

print(model)

print("Pruning ...")
model.train()
# model1.train()
# breakpoint()
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# rank = 1
transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
dataset = ImageFolder('/home/cgao037/data/imagenet-1k/train', transform=transform)
# dataset_test = torchvision.datasets.ImageNet('/home/cgao037/data/imagenet-1k/',split='val',dowmload=True, transform=transform)

loader = DataLoader(
        dataset,
        batch_size=int(128),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

# test_loader = DataLoader(
#         dataset_test,
#         batch_size=int(128),
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         drop_last=True
#     )

model.zero_grad()
# model1.zero_grad()

# breakpoint()
# loader1 = enumerate(loader)
loader = enumerate(loader)
# test_loader = enumerate(test_loader)
# _, [x, y] = next(enumerate(loader))
# _, [x1, y1] = next(enumerate(loader))
# breakpoint()
import random
max_loss = -1
loss_list = []
epoch = 10
# for _, [x, y] in loader:
#     x = x.cuda()
#     y = y.cuda()
#     output = model(x,y)
#     breakpoint() 
# breakpoint()
from tqdm import tqdm
loss_before = []
for t in tqdm(range(100)):
    
    _, [x, y] = next(loader)
    # _, [x1, y1] = next(loader)
    x = x.cuda()
    y = y.cuda()
    # x1 = x1.cuda()
    # y1 = y1.cuda()
    
    # breakpoint()
    output = model(x,y)
    # output1 = model1(x1,y1)
    # breakpoint()
    loss_before.append(output.loss.item())
    output.loss.backward()
    # output1.loss.backward()
    # print(t)
    # breakpoint()
print("before loss ", np.asarray(loss_before).mean())
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
ASP.prune_trained_model(model, optimizer,mask_calculator='m4n2')
# optimizer1 = optim.SGD(model1.parameters(), lr=1e-4, momentum=0.9)
# ASP.prune_trained_model(model1, optimizer1,mask_calculator='m4n2')
# mask_list = ASP.cound_two_model_mask(0,1)
# breakpoint()  
loss_after = []  
for t in tqdm(range(100)):
    _, [x, y] = next(loader)
    x = x.cuda()
    y = y.cuda()
    output = model(x,y)
    loss_after.append(output.loss.item())
print("after loss " , np.asarray(loss_after).mean())
# for _, [x, y] in loader:
#     x = x.cuda()
#     y = y.cuda()
#     output = model(x,y)
#     breakpoint() 

# all_samples = list()
# show_sample = True
# if show_sample:
#     with torch.no_grad():
#         with model.ema_scope():
#             uc = model.get_learned_conditioning(
#                 {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
#                 )
            
#             for class_label in classes:
#                 print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
#                 xc = torch.tensor(n_samples_per_class*[class_label])
#                 c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
#                 samples_ddim, _ = sampler.sample(S=ddim_steps,
#                                                 conditioning=c,
#                                                 batch_size=n_samples_per_class,
#                                                 shape=[3, 64, 64],
#                                                 verbose=False,
#                                                 unconditional_guidance_scale=scale,
#                                                 unconditional_conditioning=uc, 
#                                                 eta=ddim_eta)

#                 x_samples_ddim = model.decode_first_stage(samples_ddim)
#                 x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
#                                             min=0.0, max=1.0)
#                 all_samples.append(x_samples_ddim)

#     # display as grid
#     grid = torch.stack(all_samples, 0)
#     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
#     grid = make_grid(grid, nrow=n_samples_per_class)

#     # to image
#     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
#     img = Image.fromarray(grid.astype(np.uint8))
#     img.save(args.loss_type+"_samples.png")
    
#     loss_list = np.asarray(loss_list)
#     plt.plot(loss_list[:,0],loss_list[:,1]);plt.savefig(args.loss_type+'_loss.png')

# sample_for_fid = False
# if sample_for_fid:    
#     if not os.path.exists(os.path.join(args.output,args.loss_type)):
#         os.mkdir(os.path.join(args.output,args.loss_type))
        
#     n_samples_per_class = 10
#     img_id = 0
#     classes = range(1000)
#     with torch.no_grad():
#         with model.ema_scope():
#             uc = model.get_learned_conditioning(
#                 {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
#                 )
            
#             for class_label in classes:
#                 print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
#                 xc = torch.tensor(n_samples_per_class*[class_label])
#                 c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
#                 samples_ddim, _ = sampler.sample(S=ddim_steps,
#                                                 conditioning=c,
#                                                 batch_size=n_samples_per_class,
#                                                 shape=[3, 64, 64],
#                                                 verbose=False,
#                                                 unconditional_guidance_scale=scale,
#                                                 unconditional_conditioning=uc, 
#                                                 eta=ddim_eta)

#                 x_samples_ddim = model.decode_first_stage(samples_ddim)
#                 x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
#                                             min=0.0, max=1.0)
#                     #all_samples.append(x_samples_ddim)
#                 for i in range(len(x_samples_ddim)):
#                     tvu.save_image(
#                         x_samples_ddim[i], os.path.join(args.output,args.loss_type, f"{class_label}_{img_id}.png")
#                     )
#                     img_id += 1

