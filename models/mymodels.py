from typing import Any
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.cvt import VisionTransformer
from pytorch_lightning import LightningModule
from fastmri.losses import SSIMLoss
from models.lineardecoder import LinearDecoder
from models.assppdecoder import ASSPP

#encoder-decoder structure as described in: https://arxiv.org/pdf/2302.12416.pdf
#There are specific decoders for each feature map in the model.
#We use a pre-defined decoder structure to mimic the model described in the paper
#Based on code from:
#https://github.com/CIRS-Girona/s3Tseg/blob/main/models/linear_decoder.py

class FastMRIEncoderDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        #These first 3 decoders take as input the feature maps from our decoder
        self.decoder1 = LinearDecoder(img_size= (80,80), num_classes = 64, in_features=64)
        self.decoder2 = LinearDecoder(img_size=(80,80), num_classes = 64, in_features=128)
        self.decoder3 = LinearDecoder(img_size=(80, 80), num_classes=64, in_features=256)
        #this decoder takes in the concatenated (or added) feature maps from the first 3 decoder layers
        self.decoder4 = LinearDecoder(img_size = (80, 80), num_classes = 64, in_features=192)
        #this decoder takes in feature map 2 and performs a different type of feature extraction
        self.decoderASSPP = ASSPP(in_features=128, out_features=64)
        #this final decoder takes the concatenated outputs of decoder 4 and ASSPP and outputs our image
        self.decoderFinal = LinearDecoder(img_size= (320,320), num_classes=  1, in_features=128)
    def forward(self, x):
        x, x_trace = self.encoder(x)
        dec1 = self.decoder1(x_trace[0])
        dec2 = self.decoder2(x_trace[1])
        dec3 = self.decoder3(x)

        concat = torch.cat((dec1,dec2,dec3), dim=1)
        concat = rearrange(torch.flatten(concat, start_dim=2), 'b c l-> b l c') 
        dec4 = self.decoder4(concat)

        input = rearrange(x_trace[1], 'b (h w) c -> b c h w ', h = 40)
        dec5 = self.decoderASSPP(input)
        concat = torch.cat((dec4, dec5), dim=1)
       
        concat = rearrange(torch.flatten(concat, start_dim=2), 'b c l -> b l c')
        x = self.decoderFinal(concat)
        return x

class FastMRICVT(nn.Module):
    def __init__(self, in_chans=1, out_size=320, act_layer=nn.GELU, norm_layer=nn.LayerNorm, init='trunc_norm', spec=None):

        super().__init__()
        self.out_size = out_size
        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.gelu = nn.GELU()

        #upsampling head to re-create our images.
        #keep number of layers in this equal to number of blocks. 
        #seemed to not work well, use seq2img instead and do not reshape before calling
        
        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_channels = dim_embed, out_channels = dim_embed//4, kernel_size = 2, stride = 2, padding = 0, output_padding = 0),
            #nn.LeakyReLU(negative_slope=.2),
            nn.ConvTranspose2d(in_channels = dim_embed//4, out_channels = dim_embed//8, kernel_size = 2, stride=2, padding=0, output_padding=0),
            nn.ConvTranspose2d(dim_embed//4, out_channels = dim_embed//16, kernel_size = 3, stride =2, padding=1, output_padding=1),
            #nn.LeakyReLU(negative_slope=.2),
            nn.ConvTranspose2d(dim_embed//16, out_channels = 1, kernel_size=2, stride=2, padding=0,output_padding=0),
        )
        
    def forward_features(self, x):
        for i in range(self.num_stages):
            x, _ = getattr(self, f'stage{i}')(x)
        width = x.shape[3]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w ', h = width)
        #return feature maps, used in transpose convolution upsampling head 
        return x
    
    def forward(self, x):      
        x = self.forward_features(x)
        return self.head(x)

 #The FastMRI model with a UNet-style decoder. Identical to FastMRICVT with the exception of the decoder architecture
class FastMRIUNetDec(nn.Module):
    def __init__(self, in_chans=1, out_size=320, act_layer=nn.GELU, norm_layer=nn.LayerNorm, init='trunc_norm', spec=None):

        super().__init__()
        self.out_size = out_size
        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.gelu = nn.GELU()

        self.head = nn.ModuleList([
            nn.ConvTranspose2d(in_channels = dim_embed, out_channels = dim_embed//4, kernel_size = 2, stride = 2, padding = 0, output_padding = 0),

            nn.Conv2d(in_channels= dim_embed//4 + 128, out_channels= dim_embed//4, kernel_size = 3, stride = 1, padding =1 ),
            nn.Conv2d(in_channels= dim_embed//4, out_channels= dim_embed//8, kernel_size = 3, stride = 1, padding =1 ),
            nn.ConvTranspose2d(in_channels = dim_embed//8, out_channels = dim_embed//16, kernel_size = 3, stride =2, padding=1, output_padding=0),

            nn.Conv2d(in_channels = dim_embed//16 + 32, out_channels = dim_embed//16, kernel_size = 3, stride = 1, padding = 1),
            nn.Conv2d(in_channels = dim_embed//16, out_channels = dim_embed//16, kernel_size = 3, stride = 1, padding = 1),
            nn.ConvTranspose2d(dim_embed//16, out_channels = 1, kernel_size=2, stride=2, padding=0,output_padding=0),

            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 2),
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)
        ])
        
    def forward_features(self, x):
        #feature maps for skip connections
        feature_maps = []
        for i in range(self.num_stages):
            x, _ = getattr(self, f'stage{i}')(x)
            feature_maps.append(x)
        width = x.shape[3]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w ', h = width)
        #return feature maps, used in transpose convolution upsampling head 
        return x, feature_maps
    
    def forward(self, x):      
        x, feature_maps = self.forward_features(x)
        i = 1
        j = 0
        for module in self.head:
            #if on last layer, dont apply gelu
            if j == 9:
                x = module(x)
                break
            x = self.gelu(module(x))
            #concatenate feature maps only after up-sampling
            if j % 3 == 0 and not i < 0:
                x = torch.concatenate((x, feature_maps[i]), dim = 1)
                i -= 1
            j += 1

        return x   
class FastMRICVTModule(LightningModule):
    def __init__(self, learning_rate, weight_decay, kwargs):
        super().__init__()
        self.model = FastMRICVT(spec=kwargs)
        self.loss = SSIMLoss()
        self.loss.w = self.loss.w.to(torch.device("cuda"))

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        inputs, targets, _, _, _, _, max_val = batch
        pred = self.forward(inputs[:,None,:,:])
        loss = self.loss(pred, targets[:,None,:,:], max_val)
        return loss
    def validation_step(self, batch):
        inputs, targets, _, _, _, _, max_val = batch
        pred = self.forward(inputs[:,None,:,:])
        loss = self.loss(pred, targets[:,None,:,:], max_val)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr =self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
        return {'optimizer': optimizer, 'scheduler':scheduler}
        