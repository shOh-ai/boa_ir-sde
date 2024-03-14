import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys

from DenoisingUNet_arch import ConditionalUNet

#Matching loss
class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()
    

#################################### ContentLoss추가(OSH) ####################################
#############################################################################################
    
#cluster model
model_cluster = ConditionalUNet(in_nc=1, out_nc=1, nf=64, depth=4, upscale=1)

#sparse model
model_sparse = ConditionalUNet(in_nc=1, out_nc=1, nf=64, depth=4, upscale=1)
state_dict = torch.load('path_to_pretrained_model.pth') #Cluster에 대한 pretrained_model.pth 가져와서 세팅할 부분
model_sparse.load_state_dict(state_dict) # 로드한 상태 사전을 sparse model 인스턴스에 적용
model_sparse.eval() # 평가 모드로 설정

#content loss
class ContentLoss(nn.Module):
    def __init__(self, loss, model_cluster, model_sparse, out_channels=1):
        super(ContentLoss, self).__init__()
        self.criterion = loss  # L1, L2 선택하기
        self.model1 = model_cluster
        self.model2 = model_sparse

        # 1x1 convolution layer 추가
        self.feature_conv = nn.Conv2d(in_channels=1024, out_channels=out_channels, kernel_size=1)
        
    def calculate_content_loss(self, pred, target, cond, time):  # cond, time 파라미터 추가
        pred_f = self.extract_features(self.model1, pred, cond, time)  # cond, time 전달
        target_f = self.extract_features(self.model2, target, cond, time)  # cond, time 전달
        loss = self.criterion(pred_f, target_f)
        
        return loss.mean()

    def extract_features(self, model, xt, cond, time):
        # Initial processing
        x = xt - cond
        x = torch.cat([x, cond], dim=1)
        
        H, W = x.shape[2:]
        x = model.check_image_size(x, H, W)
        
        x = model.init_conv(x)
        t = model.time_mlp(time)

        # downsampling layer를 반복, but, 마지막 연산 전에 중지하여 feature extraction
        for module_list in model.downs[:-1]:  # 마지막 downsampling 작업 세트를 제외하고 모든 것 반복
            for layer in module_list:
                if isinstance(layer, Residual):
                    x = layer(x, t)  # 시간 임베딩이 필요한 레이어의 경우
                else:
                    x = layer(x)
                    
        # 마지막 downsampling 단계를 다르게 처리하여 바로 다음의 출력을 추출
        last_downsampling_layers = model.downs[-1]
        for layer in last_downsampling_layers[:-1]:  # 최종 downsampling 모듈 리스트의 마지막 레이어를 제외하고 모든 layer 처리
            if isinstance(layer, Residual):
                x = layer(x, t)  # 시간 임베딩이 필요한 레이어의 경우
            else:
                x = layer(x)        
            
        # 마지막 downsampling 작업, 즉 default_conv 바로 다음의 특징 추출
        extracted_features = last_downsampling_layers[-1](x)
        extracted_features = self.feature_conv(extracted_features)
        
        return extracted_features
    
####Usage
#content_loss = ContentLoss(loss=torch.nn.MSELoss(), model1=model, model2=model2) #loss:torch.nn.L1Loss() 으로 바꿔끼울 수 있음
#loss = content_loss.calculate_content_loss(pred_image, target_image, )