import torch
import torchvision
import numpy as np
import cv2
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader import *

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

from bts import BtsModel
import struct
import os
import sys
import matplotlib as mpl
import matplotlib.cm as cm
import argparse
import PIL.Image as pil
from torch.autograd import Variable

input_focal = 707.0912


outputs = {} 

def hook(module,input,output):
    outputs[module] = output

def bin_write(f,data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt,*data)
    f.write(bin)


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description="BTS Pytorch implementnation exporter",fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument('--model-name',type=str,help='model name',default='bts_eigen_v2_pytorch_resnext50')
parser.add_argument('--encoder',type=str,help='type of encoder resnext50',default='resnext50_bts')
parser.add_argument('--data_path',type=str,help='path to data')
parser.add_argument('--filename_file',type=str,help='path to the filenames text file')
parser.add_argument('--max_depth',type=float,help='maximum depth',default=80)
parser.add_argument('--height',type=float,help='height',default=352)
parser.add_argument('--width',type=float,help='width',default=1216)
parser.add_argument('--checkpoint_path',type=str,help='checkpoint path',default='')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--dataset',type=str,default='kitti')

width = 1216
height = 352
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def exporter(params):
    args.mode = 'test'
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    for n,m in model.named_modules():
        m.register_forward_hook(hook)

    ##make tkdnn_bin folders

    if not os.path.exists('bts_tkdnn_bin'):
        os.makedirs('bts_tkdnn_bin')    
    if not os.path.exists('bts_tkdnn_bin/debug'):
        os.makedirs('bts_tkdnn_bin/debug')
    if not os.path.exists('bts_tkdnn_bin/layers'):
        os.makedirs('bts_tkdnn_bin/layers')
    if not os.path.exists('bts_tkdnn_bin/outputs'):
        os.makedirs('bts_tkdnn_bin/outputs')
    if not os.path.exists('bts_tkdnn_bin/inputs'):
        os.makedirs('bts_tkdnn_bin/inputs')

    model.eval()
    input_image = pil.open("dog.jpg").convert('RGB')
    input_image = input_image.resize((width,height),pil.Resampling.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = input_image/255.0

  
    image = Variable(input_image).cuda()
    focal = Variable(torch.tensor([707.0912])).cuda()
    lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_cropped = model(image,focal)


    #saving inputs

    input_image_array = np.array(input_image.cpu().detach().numpy(),dtype=np.float32)
    input_image_array.tofile("bts_tkdnn_bin/inputs/input.bin",format="f")

    #saving outputs
    lpg8x8 = lpg8x8.cpu().detach().numpy()
    lpg8x8_output = np.array(lpg8x8,dtype=np.float32)
    lpg8x8_output.tofile("bts_tkdnn_bin/outputs/lpg8x8_output.bin",format="f")

    lpg4x4 = lpg4x4.cpu().detach().numpy()
    lpg4x4_output = np.array(lpg4x4,dtype=np.float32)
    lpg4x4_output.tofile("bts_tkdnn_bin/outputs/lpg4x4_output.bin",format="f")

    lpg2x2 = lpg2x2.cpu().detach().numpy()
    lpg2x2_output = np.array(lpg2x2,dtype=np.float32)
    lpg2x2_output.tofile("bts_tkdnn_bin/outputs/lpg2x2_output.bin",format="f")

    reduc1x1 = reduc1x1.cpu().detach().numpy()
    reduc1x1_output = np.array(reduc1x1,dtype=np.float32)
    reduc1x1_output.tofile("bts_tkdnn_bin/outputs/reduc1x1_output.bin",format="f")

    depth_cropped = depth_cropped.cpu().detach().numpy()
    depth_cropped_output = np.array(depth_cropped,dtype=np.float32)
    depth_cropped_output.tofile("bts_tkdnn_bin/outputs/depth_cropped_output.bin",format="f")



    #debug layer

    for n,m in model.named_modules():
        t = '-'.join(n.split('.'))
        if m not in outputs:
            continue
        in_outputs = outputs[m]

        for i in in_outputs:
            a = []
            if len(str(n)) == 0 :
                continue
            for j in i:
                if(type(j) == str):
                    continue
                else:
                    o = j.cpu().detach().numpy()
                    a.append(o)
            tempLayerArray = np.array(a,dtype=np.float32)
            t = '-'.join(n.split('.'))
            tempLayerArray.tofile("bts_tkdnn_bin/debug/"+t+".bin",format="f")

    #saving weights

    f = None
    for n,m in model.named_modules():
        t = '-'.join(n.split('.'))

        if not('of Conv2d' in str(m.type) or 'of BatchNorm2d' in str(m.type)):
            continue

        if 'of Conv2d' in str(m.type):
            file_name = "bts_tkdnn_bin/layers/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')

            w = np.array([])
            b = np.array([])
            if 'weight' in m._parameters and m._parameters['weight'] is not None:
                w = m._parameters['weight'].cpu().data.numpy()
                w = np.array(w, dtype=np.float32)
                print("    weights shape:", np.shape(w))

            if 'bias' in m._parameters and m._parameters['bias'] is not None:
                b = m._parameters['bias'].cpu().data.numpy()
                b = np.array(b, dtype=np.float32)
                print("    bias shape:", np.shape(b))
                
            bin_write(f, w)
            bias_shape = w.shape[0]
            if b.size > 0:
                bin_write(f, b)
            f.close()
            print("close file")
            f = None
        if 'of BatchNorm2d' in str(m.type):
            file_name = "bts_tkdnn_bin/layers/" + t + ".bin"
            print("open file: ",file_name)
            f = open(file_name,mode='wb')
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            s = m._parameters['weight'].cpu().data.numpy()
            s = np.array(s, dtype=np.float32)
            rm = m.running_mean.cpu().data.numpy()
            rm = np.array(rm, dtype=np.float32)
            rv = m.running_var.cpu().data.numpy()
            rv = np.array(rv, dtype=np.float32)
            bin_write(f, b)
            bin_write(f, s)
            bin_write(f, rm)
            bin_write(f, rv)

            print("    b shape:", np.shape(b))
            print("    s shape:", np.shape(s))
            print("    rm shape:", np.shape(rm))
            print("    rv shape:", np.shape(rv))

            f.close()
            print("close file")
            f = None




    
if __name__ == '__main__':
    exporter(args)

