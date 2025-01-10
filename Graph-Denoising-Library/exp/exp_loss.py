import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import sys
from sklearn.model_selection import train_test_split
import torch.sparse as ts


warnings.filterwarnings('ignore')


# 节点分类实验类  在类中定义的实例方法，它应该至少接受 self 作为第一个参数
class Exp_Link_Prediction(Exp_Basic):
    def __init__(self, args):
        super(Exp_Link_Prediction, self).__init__(args)