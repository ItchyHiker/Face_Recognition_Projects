import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormFace(nn.Module):
    def __init__(self, feature_dim, num_class, scale=16):
        super(NormFace, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(feature_dim, num_class))
        nn.init.xavier_uniform_(self.weight)
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

class ArcFace2(nn.Module):
    # Additive Margin Softmax: https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(ArcFace2, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        nn.init.xavier_uniform_(self.kernel)
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0) # normalize for each column
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output
    def __str__(self):
        return "ArcFace2"
