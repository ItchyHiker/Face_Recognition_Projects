import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormFace(nn.Module):
    def __init__(self, feature_dim, num_class, scale=16):
        super(NormFace, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(feature_dim, num_class))
        nn.init.xavier_uniform(self.weight)
        self.weight.data.uniform_(-1, 1).renorm(2, 1, 1e-5).mul_(1e5)
        self.scale = scale
    def forward(self, x):
        cosine = F.normalize(x).mm(F.normalize(self.weight, dim=0))
        return cosine*self.scale
    def __str__(self):
        return 'NormFace'

class SphereFace(nn.Module):
    def __init__(self, feature_dim, num_class, m=4, base=1000.0, gamma=0.001, 
            power=2, lambda_min=5.0):
        super(SphereFace, self).__init__()
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.m = m
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min
        self.iter = 0
        self.weight = nn.Parameter(torch.Tensor(num_class, feature_dim))
        nn.init.xavier_uniform(self.weight)
        
        self.margin_formula = [
                lambda x: x ** 0,
                lambda x: x ** 1,
                lambda x: 2 * x ** 2 - 1,
                lambda x: 4 * x ** 3 - 3 * x,
                lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
                lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
            ]
    def forward(self, x, label):
        self.iter += 1
        self.cur_lambda = max(self.lambda_min, 
                self.base * (1 + self.gamma*self.iter)**(-1*self.power))
        cos_theta  = F.linear(F.normalize(x), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.margin_formula[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = ((self.m * theta) / math.pi).floor()
        phi_theta = ((-1.0)**k)*cos_m_theta - 2*k
        phi_theta_ = (self.cur_lambda * cos_theta + phi_theta) / (1 + self.cur_lambda)
        norm_of_feature = torch.norm(x, 2, 1)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = one_hot*phi_theta_ + (1 - one_hot)*cos_theta
        output *= norm_of_feature.view(-1, 1)

        return output

    def __str__(self):
        return 'SphereFace'

class CosFace(nn.Module):
    def __init__(self, feature_dim, num_class, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.feature_dims = feature_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(num_class, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1,  label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        return output
    def __str__(self):
        return 'CosFace'

class ArcFace(nn.Module):
    def __init__(self, feature_dim, num_class, s=32.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(num_class, feature_dim))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        # make the function cos(theta_m) monotonic decreasing while theta in [0, 180]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output

    def __str__(self):
        return 'ArcFace'
