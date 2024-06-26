import torch
from torch.autograd import Function
import numpy as np
import math
from torch.nn import Module
import pywt
import torch.nn as nn


class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH
    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None

class IDWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input_LL, input_LH, input_HL, input_HH,
                matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()), torch.matmul(input_LH, matrix_High_1.t()))
        H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()), torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), L), torch.matmul(matrix_High_0.t(), H))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        grad_LH = torch.matmul(grad_L, matrix_High_1)
        grad_HL = torch.matmul(grad_H, matrix_Low_1)
        grad_HH = torch.matmul(grad_H, matrix_High_1)
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None


class DWT_2D(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)

class IDWT_2D(Module):
    """
    input:  lfc -- (N, C, H/2, W/2)
            hfc_lh -- (N, C, H/2, W/2)
            hfc_hl -- (N, C, H/2, W/2)
            hfc_hh -- (N, C, H/2, W/2)
    output: the original 2D data -- (N, C, H, W)
    """
    def __init__(self, wavename):
        """
        2D inverse DWT (IDWT) for 2D image reconstruction
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        """
        recontructing the original 2D data
        the original 2D data = \mathcal{L}^T * lfc * \mathcal{L}
                             + \mathcal{H}^T * hfc_lh * \mathcal{L}
                             + \mathcal{L}^T * hfc_hl * \mathcal{H}
                             + \mathcal{H}^T * hfc_hh * \mathcal{H}
        :param LL: the low-frequency component
        :param LH: the high-frequency component, hfc_lh
        :param HL: the high-frequency component, hfc_hl
        :param HH: the high-frequency component, hfc_hh
        :return: the original 2D data
        """
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        self.get_matrix()
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWT(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(DWT, self).__init__()
        # self.dwt = DWT_2D_tiny(wavename = wavename) #return LL
        self.dwt = DWT_2D(wavename = wavename)   #return LL,LH,HL,HH

    def forward(self, input):
        # LL = self.dwt(input)
        # return LL
        # inputori= input
        LL,LH,HL,HH = self.dwt(input)
        # LL = LL+LH+HL+HH
        # result = torch.sum(LL, dim=[2, 3])  # x:torch.Size([64, 256, 56, 56])
        return LL,LH,HL,HH  ###torch.Size([64, 256])



if __name__ == '__main__':
    from datetime import datetime
    from torch.autograd import gradcheck
    import cv2

    dummy_x = torch.randn([16,64,16,16]).cuda()
    mm = DWT().cuda()
    LL,LH,HL,HH = mm(dummy_x)
    print(LL.shape)
    adf






    wavelet = pywt.Wavelet('bior1.1')
    h = wavelet.rec_lo
    g = wavelet.rec_hi
    h_ = wavelet.dec_lo
    g_ = wavelet.dec_hi
    h_.reverse()
    g_.reverse()


    # image_full_name = '/home/li-qiufu/Pictures/standard_test_images/lena_color_512.tif'
    # image = cv2.imread(image_full_name, flags = 1)

    image_full_name = '/disk1/imed_hjq/data/WenzhouMedicalUniversity/Parkinsonism/pd_dataset_R2/1/0_OD'
    image = cv2.imread(image_full_name + "/0_OD_2021-01-25_09-38-22_Star 18Line R16_139.754_F12.01A3.01_B-scan_PixelRatio_1.png")

    image = image[0:512,0:512,:]
    print(image.shape)

    height, width, channel = image.shape
    #image = image.reshape((1,height,width))
    t0 = datetime.now()
    for index in range(100):
        m0 = DWT_2D(band_low = h, band_high = g)
        image_tensor = torch.Tensor(image)
        image_tensor.unsqueeze_(dim = 0)
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.transpose_(1,3)
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.transpose_(2,3)
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.requires_grad = False
        LL, LH, HL, HH = m0(image_tensor)
        matrix_low_0 = torch.Tensor(m0.matrix_low_0)
        matrix_low_1 = torch.Tensor(m0.matrix_low_1)
        matrix_high_0 = torch.Tensor(m0.matrix_high_0)
        matrix_high_1 = torch.Tensor(m0.matrix_high_1)

        image_tensor.requires_grad = True
        input = (image_tensor.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_high_0.double(), matrix_high_1.double())
        test = gradcheck(DWTFunction_2D.apply, input)
        print(test)
        print(LL.requires_grad)
        print(LH.requires_grad)
        print(HL.requires_grad)
        print(HH.requires_grad)
        LL.requires_grad = True
        input = (LL.double(), LH.double(), HL.double(), HH.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_high_0.double(), matrix_high_1.double())
        test = gradcheck(IDWTFunction_2D.apply, input)
        print(test)

        m1 = IDWT_2D(band_low = h_, band_high = g_)
        image_re = m1(LL,LH,HL,HH)
        break

    t1 = datetime.now()
    image_re.transpose_(2,3)
    image_re.transpose_(1,3)
    image_re_np = image_re.detach().numpy()
    print('image_re shape: {}'.format(image_re_np.shape))

    image_zero = image - image_re_np[0]
    print(np.max(image_zero), np.min(image_zero))
    print(image_zero[:,8])
    print('taking {} secondes'.format(t1 - t0))
    cv2.imshow('reconstruction', image_re_np[0]/255)
    cv2.imshow('image_zero', image_zero/255)
    cv2.waitKey(0)

    """
    image_full_name = '/home/liqiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(image_full_name, flags = 1)
    image = image[0:512,0:512,:]
    print(image.shape)
    image_3d = np.concatenate((image, image, image, image, image, image), axis = 2)
    print(image_3d.shape)
    image_tensor = torch.Tensor(image_3d)
    #image_tensor = image_tensor.transpose(dim0 = 2, dim1 = 1)
    #image_tensor = image_tensor.transpose(dim0 = 1, dim1 = 0)
    image_tensor.unsqueeze_(dim = 0)
    image_tensor.unsqueeze_(dim = 0)
    t0 = datetime.now()
    for index in range(10):
        m0 = DWT_3D(wavename = 'haar')
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.requires_grad = False
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = m0(image_tensor)
        matrix_low_0 = torch.Tensor(m0.matrix_low_0)
        matrix_low_1 = torch.Tensor(m0.matrix_low_1)
        matrix_low_2 = torch.Tensor(m0.matrix_low_2)
        matrix_high_0 = torch.Tensor(m0.matrix_high_0)
        matrix_high_1 = torch.Tensor(m0.matrix_high_1)
        matrix_high_2 = torch.Tensor(m0.matrix_high_2)

        #image_tensor.requires_grad = True
        #input = (image_tensor.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_low_2.double(),
        #                                matrix_high_0.double(), matrix_high_1.double(), matrix_high_2.double())
        #test = gradcheck(DWTFunction_3D.apply, input)
        #print('testing dwt3d -- {}'.format(test))
        #LLL.requires_grad = True
        #input = (LLL.double(), LLH.double(), LHL.double(), LHH.double(),
        #         HLL.double(), HLH.double(), HHL.double(), HHH.double(),
        #         matrix_low_0.double(), matrix_low_1.double(), matrix_low_2.double(),
        #         matrix_high_0.double(), matrix_high_1.double(), matrix_high_2.double())
        #test = gradcheck(IDWTFunction_3D.apply, input)
        #print('testing idwt3d -- {}'.format(test))

        m1 = IDWT_3D(wavename = 'haar')
        image_re = m1(LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH)
    t1 = datetime.now()
    image_re.squeeze_(dim = 0)
    image_re.squeeze_(dim = 0)
    #image_re.transpose_(0,1)
    #image_re.transpose_(1,2)
    image_re_np = image_re.detach().numpy()
    print('image_re shape: {}'.format(image_re_np.shape))

    image_zero = image - image_re_np[:,:,0:3]
    print(np.max(image_zero), np.min(image_zero))
    #print(image_zero[:,8,0])
    print('taking {} secondes'.format(t1 - t0))
    cv2.imshow('reconstruction', image_re_np[:,:,0:3]/255)
    cv2.imshow('image_zero', image_zero/255)
    cv2.waitKey(0)
    """

    """
    import matplotlib.pyplot as plt
    import numpy as np
    vector_np = np.array(list(range(1280)))#.reshape((128,1))

    print(vector_np.shape)
    t0 = datetime.now()
    for index in range(100):
        vector = torch.Tensor(vector_np)
        vector.unsqueeze_(dim = 0)
        vector.unsqueeze_(dim = 0)
        m0 = DWT_1D(band_low = h, band_high = g)
        L, H = m0(vector)

        #matrix_low = torch.Tensor(m0.matrix_low)
        #matrix_high = torch.Tensor(m0.matrix_high)
        #vector.requires_grad = True
        #input = (vector.double(), matrix_low.double(), matrix_high.double())
        #test = gradcheck(DWTFunction_1D.apply, input)
        #print('testing 1D-DWT: {}'.format(test))
        #print(L.requires_grad)
        #print(H.requires_grad)
        #L.requires_grad = True
        #H.requires_grad = True
        #input = (L.double(), H.double(), matrix_low.double(), matrix_high.double())
        #test = gradcheck(IDWTFunction_1D.apply, input)
        #print('testing 1D-IDWT: {}'.format(test))

        m1 = IDWT_1D(band_low = h_, band_high = g_)
        vector_re = m1(L, H)
    t1 = datetime.now()
    vector_re_np = vector_re.detach().numpy()
    print('image_re shape: {}'.format(vector_re_np.shape))

    vector_zero = vector_np - vector_re_np.reshape(vector_np.shape)
    print(np.max(vector_zero), np.min(vector_zero))
    print(vector_zero[:8])
    print('taking {} secondes'.format(t1 - t0))
    """
