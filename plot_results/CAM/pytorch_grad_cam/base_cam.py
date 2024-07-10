import cv2
import numpy as np
import torch
import ttach as tta
import sys
sys.path.append('/disk1/imed_hjq/code/3-PD_analysis/02_baseline/')
from CAM.pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from CAM.pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class BaseCAM:
    def __init__(self, 
                 model, 
                 target_layer,
                 use_cuda=False,
                 reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model, 
            target_layer, reshape_transform)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):   # output is tuple,len()=3
            """
            20221027
            grad cam存在一个问题，当真实标签为1的时候，base_cam.py中的def get_loss(self, output, target_category):会报错
            这是因为output的shape是[bs,1]，label =[1,1]的时候，target_category[i]=1，
            导致output[i, target_category[i]]=output[1,1]，显然，output没有这个值，因为output的shape是[bs,1]

            因此，我需要把模型输出[bs, 1]，改成[bs, 2]，注意，这里多了sigmoid
            """
            if output.shape[1] == 1:
                output = torch.sigmoid(output)
                output = torch.concat([output, 1-output], dim=1)

            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):
        """
        核心代码
        :param input_tensor:
        :param target_category:
        :param eigen_smooth:
        :return:
        """

        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            # 0513 output --> output[0]
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        cam = self.get_cam_image(input_tensor, target_category, 
            activations, grads, eigen_smooth)

        # relu激活
        cam = np.maximum(cam, 0)

        result = []
        for img in cam:
            img = cv2.resize(img, input_tensor.shape[-2:][::-1])
            # 归一化处理
            img = img - np.min(img)
            img = img / np.max(img)
            result.append(img)
        result = np.float32(result)
        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor,
                target_category, eigen_smooth)

        return self.forward(input_tensor,
            target_category, eigen_smooth)