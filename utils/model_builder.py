import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.utils.model_zoo as model_zoo
from models.AWCANet import AWCANet


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'vit_base_patch16_224_in21k':'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
}

def build_model(model_name, num_classes=1, backbone='resnet50', pretrained=False, out_stride=16, mult_grid=False):
    if model_name == 'AWCANet':
        model = AWCANet(8, 1)


    if pretrained:
        os.environ['TORCH_HOME'] = '../pretrained'
        pretrain_dict = model_zoo.load_url(model_urls[backbone])
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    return model
