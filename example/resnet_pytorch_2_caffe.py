import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
import pytorch_to_caffe

if __name__=='__main__':
    name='resnet18'
    resnet18=resnet.resnet18()
    for param in resnet18.parameters():
        param.requires_grad=False 
    num_fc = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_fc,7)
    checkpoint = torch.load('/media/yyk/My Passport/DL/PytorchToCaffe-master/example/res18.pth')
    
    resnet18.load_state_dict(checkpoint)
    resnet18.eval()
    input=torch.ones([1,3,224,224])
     #input=torch.ones([1,3,224,224])
    pytorch_to_caffe.trans_net(resnet18,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))