import torch
import torch.nn as nn
import pdb
# import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(Block, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.LN1 = nn.LayerNorm(input_dim)
        self.LN2 = nn.LayerNorm(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # first layer
        self.linear = nn.Sequential(nn.Linear(input_dim, 256))
        self.project = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.Linear = nn.Sequential(nn.Linear(output_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask = torch.ceil(self.A * 0.00001)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        H = self.LN1(H)
        H_xx1 = self.linear(H)
        e = torch.softmax(torch.matmul(H_xx1, H_xx1.t()), dim=1)

        # e = torch.softmax(torch.matmul(H, H.t()), dim=1)
        zero_vec = -9e15 * torch.ones_like(e)
        A = torch.where(self.mask > 0, e, zero_vec) + self.I
        # A = e + self.I
        if model != 'normal': A = torch.clamp(A, 0.1)  # This is a trick for the Indian Pines.
        A = torch.softmax(A, dim=1)
        H = torch.mm(A, H)

        output = self.Activition(self.project(self.LN2(H)))

        return output, A


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class CEGCN(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,
                 model='normal'):
        super(CEGCN, self).__init__()
        self.class_count = class_count  # class number
        self.channel = channel  # output dimension
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # column normalization on Q

        layers_count_CNN, layers_count_former = 2, 2
        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count_CNN):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        # Pixel-level Convolutional Sub-Network
        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count_CNN):
            if i < layers_count_CNN - 1:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 128, kernel_size=5))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))

        # Superpixel-level Graph Sub-Network
        self.SemanticFormer = nn.Sequential()
        for i in range(layers_count_former):
            if i < layers_count_former - 1:
                self.SemanticFormer.add_module('SemanticFormer' + str(i), Block(128, 128, self.A))
            else:
                self.SemanticFormer.add_module('SemanticFormer' + str(i), Block(128, 128, self.A))
                
        # Softmax layer
        self.Softmax_linear_CNN = nn.Sequential(nn.Linear(64, self.class_count))
        self.Softmax_linear_former = nn.Sequential(nn.Linear(128, self.class_count))


    def forward(self, x: torch.Tensor, train=False):
        '''
        :param x: H*W*C torch.Size([145, 145, 200])
        :return: probability_map
        '''
        (h, w, c) = x.shape

        # remove noise
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  

        clean_x_flatten = clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # low frequency
        hx = clean_x

        # CNN and SemanticFormer branch
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        # SemanticFormer
        H = superpixels_flatten
        if self.model == 'normal':
            for i in range(len(self.SemanticFormer)): H, _ = self.SemanticFormer[i](H)
        else:
            for i in range(len(self.SemanticFormer)): H, _ = self.SemanticFormer[i](H, model='smoothed')

        former_result = torch.matmul(self.Q, H)  # self.norm_row_Q == self.Q

        
        Y = CNN_result
        Y = self.Softmax_linear_CNN(Y)
        Y = torch.softmax(Y, -1)
        former_result = self.Softmax_linear_former(former_result)
        former_result = torch.softmax(former_result, -1)
        CNN_result = self.Softmax_linear_CNN(CNN_result)
        CNN_result = torch.softmax(CNN_result, -1)
        return CNN_result, former_result, Y