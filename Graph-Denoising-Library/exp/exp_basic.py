import os
import torch

from models import DropEdge, RGIB, Edmot,DGMM


# 任务基类
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # 这里填写已经适配好的模型
        self.model_dict = {
            'DropEdge': DropEdge,
            'RGIB' : RGIB,
            'Edmot' : Edmot,
            'DGMM' : DGMM
        }

        self.device = self._acquire_device()
        # self.model = self._build_model().to(self.device)
        # 进行模型构建 这里调用的是子类的方法
        self.model = self._build_model()


    def _build_model(self):
        raise NotImplementedError
        return None

    # 判断是否使用gpu选择device
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
