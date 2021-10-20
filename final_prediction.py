from networks import RefineModel,DCT_SE_DDNet3
from train_dataset_refine import SparseValDataset
import torch
import os
import numpy as np
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--sample_view', type=int, default=240)
parser.add_argument('--use_cuda', type=str, default='3')
parser.parse_args()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_cuda

#-----------------------------------------------------------------------------------------------------------------
net_g_path = './network_parameters/overall_train_lamb1e-07_bs4_lr1_420001B.pth'
device = 'cuda'

net_g1 = DCT_SE_DDNet3(num_features=32, growth_rate=32)
net_g1 = net_g1.to(device)
net_g1.load_state_dict(torch.load(net_g_path, map_location=device)['model_g1'])

net_g3 = RefineModel(in_channels=2, out_channels=1, n_resblocks=6, n_feats=64)
net_g3 = net_g3.to(device)
net_g3.load_state_dict(torch.load(net_g_path, map_location=device)['model_g3'])

#----------------------------------------------------------------------------------------------------------------
val_root = './dataset/sparse_' + str(args.sample_view)
val_dataset = SparseValDataset(gt_root='./dataset',
                               input_root=val_root,
                               batch_size=1,
                               ct_window=True)
val_dataloader = val_dataset.build_datasets()

#----------------------------------------------------------------------------------------------------------------------
def u_window_transform(u_img,window_center, window_width, u_min, u_max):
    u_img = u_img*(u_max-u_min)+u_min
    u_water = 0.02
    x = (u_img/u_water -1)*1000
    window_min = window_center-window_width//2
    return (x-window_min)/window_width

#----------------------------------------------------------------------------------
start_epoch = 0
current_iter = 0
total_epochs = 0

save_img_dir = os.path.join('/data1G/sc/sparse_view/test/sparse_'+str(args.sample_view), 'refine_result')
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)

for epoch in range(start_epoch, total_epochs + 1):
    epoch += 1
    for iteration, data in enumerate(val_dataloader):
        current_iter += 1

        lq = data["img_input"].to(device)
        gt = data["img_gt"].to(device)
        res_prob = data["res_prob"].to(device)

        net_g1.eval()
        net_g3.eval()

        split_name = data["split_name"]
        img_full_path = os.path.join(save_img_dir, split_name[0] + '.png')

        with torch.no_grad():
            SE_result = net_g1(lq.float())
            input = torch.cat((res_prob, SE_result), dim=1)
            output = net_g3(input.float())
            output_numpy = output.detach().cpu().squeeze().numpy()
            output_numpy[output_numpy > 1] = 1
            output_numpy[output_numpy < 0] = 0
            output_numpy = (output_numpy-np.min(output_numpy))/ (np.max(output_numpy)-np.min(output_numpy))
            cv2.imwrite(img_full_path, output_numpy * 255)
            print('finish images ', current_iter)