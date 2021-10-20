from networks import Predict_Construction_Error
from train_dataset_construct_error import SparseValDataset
import torch
import os
import numpy as np
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--sample_view', type=int, default=60)
parser.add_argument('--use_cuda', type=str, default='1')

parser.parse_args()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_cuda

device = 'cuda'
net_g = Predict_Construction_Error()
net_g = net_g.to(device)
net_g_path = './network_parameters/predict_Construction_Error_bs4_lr1_1340001B.pth'
net_g.load_state_dict(torch.load(net_g_path, map_location=device)['model_g'])

val_root = './dataset/sparse_' + str(args.sample_view)
val_dataset = SparseValDataset(gt_root='./dataset',
                               input_root=val_root,
                               batch_size=1)
val_dataloader = val_dataset.build_datasets()
#----------------------------------------------------------------------------------------------------------------------
start_epoch = 0
current_iter = 0
total_epochs = 0

save_img_dir = os.path.join('./dataset/sparse_'+str(args.sample_view), 'res_prob')
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)

for epoch in range(start_epoch, total_epochs + 1):
    epoch += 1
    for data in val_dataloader:
        current_iter += 1
        lq = data["img_input"].to(device)
        SE_result = data["SE_result"].to(device)
        split_name = data["split_name"]
        net_g.eval()
        with torch.no_grad():
            input = torch.cat((lq, SE_result), dim=1)
            output = net_g(input.float())
            output = torch.sigmoid(output)
            output_numpy = output.detach().cpu().squeeze().numpy()
            output_numpy[output_numpy > 1] = 1
            output_numpy[output_numpy < 0] = 0
            img_full_path = os.path.join(save_img_dir, split_name[0]+'.png')
            img_input = Image.fromarray(np.uint8(output_numpy * 255))
            img_input.save(img_full_path)
            print('finish images ', current_iter)
print('finish')


