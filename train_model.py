import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import fastmri
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
from fastmri.losses import SSIMLoss

from models.mymodels import FastMRICVT
from models.mymodels import FastMRIEncoderDecoder
from models.cmt import CMTEncoder
from models.lineardecoder import LinearDecoder

import json

#hyperparameters defined here. Eventually, move to yaml files

spec ={
    'INIT': 'trunc_norm',
    'NUM_STAGES': 3,
    'PATCH_SIZE': [5, 3, 3],
    'PATCH_STRIDE': [2, 2, 2],
    'PATCH_PADDING': [1, 1,1],
    'DIM_EMBED': [32, 128, 256],
    'NUM_HEADS': [1, 4, 8],
    'DEPTH': [1, 2, 8],
    'MLP_RATIO': [4.0, 4.0,4.0],
    'ATTN_DROP_RATE': [0.0, 0.0,0.0],
    'DROP_RATE': [0.0, 0.0,0,0],
    'DROP_PATH_RATE': [0.0, 0.0,0.0],
    'QKV_BIAS': [True, True,True],
    'CLS_TOKEN': [False, False, False],
    'POS_EMBED': [False, False, False],
    'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
    'KERNEL_QKV': [3, 3, 3],
    'PADDING_KV': [1, 1,1],
    'STRIDE_KV': [2, 2,2],
    'PADDING_Q': [1, 1,1],
    'STRIDE_Q': [1, 1,1]
}
epochs = 25
learning_rate = 1e-4
weight_decay = 1e-16
batch_size = 4
device = torch.device('cuda')
#device = torch.device('cpu')
encoder = CMTEncoder(
    img_size=320, in_chans=1, num_classes=0, patch_size=4, embed_dim=64,
    depths=[1, 2, 3], kv_scale=[8, 4, 2], num_heads=[2, 4, 8], mlp_ratio=4.,
    qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    use_rel_pos_embed=False, use_bn_in_attn=True, use_irffn=False, use_lpu=False,
    use_ghost_ffn=False, use_multi_merge=True, norm_layer=nn.LayerNorm
).to(device)
model = FastMRIEncoderDecoder(encoder).to(device)
#model = FastMRICVT(spec=spec).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate, weight_decay=weight_decay)
scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
log_every = 25
criterion = SSIMLoss()
criterion.w = criterion.w.to(device)


def train_loop(epoch, model, loader):
    model.train()
    i = 0
    results = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for inputs, targets, _ , _, _, _, max_val in loader:
        
        optimizer.zero_grad()
        pred = model(inputs.to(device)[:,None,:,:])
        loss = criterion(pred, targets.to(device)[:,None,:,:], torch.Tensor(max_val).to(device))

        loss.backward()
        optimizer.step()

        results['loss'] += loss.item() * len(inputs)
        results['counter'] += len(inputs)
        results['loss_arr'].append(loss.item())
        if i % log_every == 0:
            print("Train: Epoch: %d \t Iteration: %d \t loss: %.4f" % (epoch, i, sum(results['loss_arr'])/len(results['loss_arr'])))
            results['loss_arr'] = []
        i +=1

    scheduler.step()
    return 
def val_loop(epoch, model, loader):
   model.eval()
   i = 0
   results = {'loss': 0, 'counter': 0, 'loss_arr':[]}
   with torch.no_grad():
      for inputs, targets, _ , _, _, _ , max_val in loader:
         pred = model(inputs.to(device)[:,None,:,:])
         
         loss = criterion(pred, targets.to(device)[:,None,:,:], torch.Tensor(max_val).to(device))
         results['loss'] += loss.item() * len(inputs)
         results['counter'] += len(inputs)
         results['loss_arr'].append(loss.item())
         
         if i % log_every == 0:
             print("Val: Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(results['loss_arr'][-10:])/len(results['loss_arr'][-10:])))
         i += 1
         
   return results['loss']/results['counter']

if __name__ == "__main__":
    

    mask_func = subsample.RandomMaskFunc(
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8]    
        )   
    
    train = mri_data.SliceDataset(
        root='/home/wesselljack00/mybucket/singleCoil/singlecoil_train',
        transform=transforms.UnetDataTransform('singlecoil', mask_func=mask_func),
        challenge='singlecoil'
        )

    val = mri_data.SliceDataset(
        root='/home/wesselljack00/mybucket/singleCoil/singlecoil_val',
        transform=transforms.UnetDataTransform('singlecoil', mask_func=mask_func),
        challenge='singlecoil'
        )
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers =2)
    val_loader = DataLoader(val,batch_size=batch_size, shuffle = True, num_workers=2)
    
    results = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_epoch': 0}
    for epoch in range(0, epochs):
        train_loop(epoch, model, train_loader)

        val_loss = val_loop(epoch, model, val_loader)

        results['epochs'].append(epoch)
        results['losess'].append(val_loss)

        if val_loss < results['best_val']:
            results['best_val'] = val_loss
            results['best_epoch'] = epoch
            torch.save(model.state_dict(), "trained_models/best_model_" + str(epoch) + ".pt")
            
            print("Val loss: %.4f  \t epoch %d" % (val_loss,epoch))
            print("Best: val loss: %.4f \t epoch %d" % (results['best_val'], results['best_epoch']))


        json_object = json.dumps(results, indent=4)
        with open("losess.json", "w") as outfile:
            outfile.write(json_object)