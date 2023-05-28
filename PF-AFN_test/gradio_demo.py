import numpy as np
import gradio as gr
import torch.nn as nn
from options.test_options import TestOptions
import numpy as np
from PIL import Image
import torch
import cv2
import torch.nn.functional as F

from data.base_dataset import get_params, get_transform
from models.afwm import AFWM
from models.networks import ResUnetGenerator, load_checkpoint

from edge_extraction_script import get_edges

opt = TestOptions().parse()

warp_model = AFWM(opt, 3)
print(warp_model)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, opt.warp_checkpoint)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
print(gen_model)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, opt.gen_checkpoint)

def get_combined_image(real_image, clothes, tryon):
    a = real_image.float().cuda()
    b = clothes.cuda()
    c = tryon
    combine = torch.cat([a[0], b[0], c[0]], 2).squeeze()
    cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    rgb = (cv_img * 255).astype(np.uint8)
    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb


def get_tryon_image(real_image, clothes):
    edges = get_edges(clothes)

    real_image = Image.fromarray(real_image.astype('uint8'))

    clothes = Image.fromarray(clothes.astype('uint8'))

    edges = Image.fromarray(edges.astype('uint8')).convert('L')

    params = get_params(opt, real_image.size)
    transform = get_transform(opt, None)
    transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)

    real_image_tensor = transform(real_image).unsqueeze(0)
    clothes_tensor = transform(clothes).unsqueeze(0)
    edges_tensor = transform_E(edges).unsqueeze(0)

    print(real_image_tensor.shape)
    print(clothes_tensor.shape)
    print(edges_tensor.shape)

    edge = torch.FloatTensor((edges_tensor.detach().numpy() > 0.5).astype(np.int64))
    clothes = clothes_tensor * edge

    flow_out = warp_model(real_image_tensor.cuda(), clothes_tensor.cuda())
    warped_cloth, last_flow, = flow_out
    warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                                mode='bilinear', padding_mode='zeros', align_corners=True)  

    gen_inputs = torch.cat([real_image_tensor.cuda(), warped_cloth, warped_edge], 1)
    print(gen_inputs.shape)
    gen_outputs = gen_model(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    return get_combined_image(real_image_tensor, clothes_tensor, p_tryon)


demo = gr.Interface(get_tryon_image, inputs=[gr.Image(shape=(192, 256)), gr.Image(shape=(192, 256))], outputs="image")

demo.launch()   
