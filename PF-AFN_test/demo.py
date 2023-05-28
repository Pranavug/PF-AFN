import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F

import gradio as gr
from data.base_dataset import *

def vton(person_img):
    # Invoke model with person img and return img with different clothes
    # Save image as person_img.jpg
    print(person_img)
    Image.fromarray(person_img).save("person_img.jpg")

    return person_img


def initialize(opt):
    warp_model = AFWM(opt, 3)
    warp_model.eval()
    warp_model.cuda()
    load_checkpoint(warp_model, opt.warp_checkpoint)

    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    gen_model.eval()
    gen_model.cuda()
    load_checkpoint(gen_model, opt.gen_checkpoint)

    return warp_model, gen_model

def preprocess_image(opt):
    im_name, c_name = "person_img.jpg", "cloth_img.jpg"
    data_dir = "server_store"
    I_path = os.path.join(data_dir,im_name)
    I = Image.open(I_path).convert('RGB')

    params = get_params(opt, I.size)
    transform = get_transform(opt, params)
    transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)

    I_tensor = transform(I)

    C_path = os.path.join(data_dir,c_name)
    C = Image.open(C_path).convert('RGB')
    C_tensor = transform(C)

    E_path = os.path.join(data_dir,c_name)
    E = Image.open(E_path).convert('L')
    E_tensor = transform_E(E)

    input_dict = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}
    return input_dict


if __name__ == '__main__':
    opt = TestOptions().parse()
    warp_model, gen_model = initialize(opt)
    
    demo = gr.Interface(fn=vton, inputs="image", outputs="image")
    demo.launch()


for _, data in enumerate(dataset):
    iter_start_time = time.time()

    real_image = data['image']
    clothes = data['clothes']
    ##edge is extracted from the clothes image with the built-in function in python
    edge = data['edge']
    edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
    clothes = clothes * edge        

    flow_out = warp_model(real_image.cuda(), clothes.cuda())
    warped_cloth, last_flow, = flow_out
    warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                        mode='bilinear', padding_mode='zeros', align_corners=True)

    gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
    gen_outputs = gen_model(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    path = 'results/' + opt.name
    os.makedirs(path, exist_ok=True)
    sub_path = path + '/PFAFN'
    os.makedirs(sub_path,exist_ok=True)

    # Save image files
    a = real_image.float().cuda()
    b= clothes.cuda()
    c = p_tryon
    combine = torch.cat([a[0],b[0],c[0]], 2).squeeze()
    cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
    rgb=(cv_img*255).astype(np.uint8)
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite(sub_path+'/'+str("test_img")+'.jpg',bgr)
