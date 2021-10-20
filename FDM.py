from networks import DCT_SE_DDNet3
from train_dataset import SparseValDataset
import torch
import os
import numpy as np
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--sample_view', type=int, default=60)
parser.add_argument('--use_cuda', type=str, default='3')

parser.parse_args()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_cuda

#-----------------------------------------------------------------------------------------------------------------
device = 'cuda'
net_g = DCT_SE_DDNet3(num_features=32, growth_rate=32)
net_g = net_g.to(device)
net_g_path = './network_parameters/overall_train_lamb1e-07_bs4_lr1_420001B.pth'
net_g.load_state_dict(torch.load(net_g_path, map_location=device)['model_g1'])


val_root = './dataset/sparse_' + str(args.sample_view)
val_dataset = SparseValDataset(gt_root='./dataset',
                               input_root=val_root,
                               batch_size=1)
val_dataloader = val_dataset.build_datasets()


start_epoch = 0
current_iter = 0
total_epochs = 0

save_img_dir = os.path.join('./dataset/sparse_'+str(args.sample_view), 'SE_result')
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)

for epoch in range(start_epoch, total_epochs + 1):
    epoch += 1
    for data in val_dataloader:
        current_iter += 1

        lq = data["img_input"].to(device)
        split_name = data["split_name"]
        img_full_path = os.path.join(save_img_dir, split_name[0] + '.png')

        net_g.eval()
        with torch.no_grad():
            output = net_g(lq)
            output_numpy = output.detach().cpu().squeeze().numpy()
            output_numpy[output_numpy > 1] = 1
            output_numpy[output_numpy < 0] = 0
            output = Image.fromarray(np.uint8(output_numpy * 255))
            output.save(img_full_path)
            print('finish images ', current_iter)

print('finish')
