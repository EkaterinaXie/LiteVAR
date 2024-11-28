import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseQuantizer(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.scale = None
        self.zero_point = None
        self.n_levels = 2**self.n_bits

        self.do_calibration = False
        self.name = 'default quantizer'

    def forward(self, x):
        if self.do_calibration:
            self.calibration(x)
        x_int = (x/self.scale+self.zero_point).round().clamp(0, self.n_levels-1)
        x_dequant = (x_int-self.zero_point)*self.scale

        return x_dequant
    
    def calibration(self, x):
        self.scale = (x.max()-x.min())/self.n_levels
        self.zero_point = x.min()

# perchannel weight quantizer
class WeightPCQuantizer(nn.Module):
    # Per output channel quantization (weights in an output channel share a scale and a zero point)
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.scale = None
        self.zero_point = None

        self.do_calibration = False
        self.name = 'default weight quantizer'

    def forward(self, x):
        if self.do_calibration:
            self.calibration(x)

        x_int = (x/self.scale + self.zero_point).round().clamp(0, self.n_levels-1)
        x_dequant = (x_int - self.zero_point) * self.scale

        return x_dequant
    
    def calibration(self, x):
        """
        weight: shape [out_channel, in_channel]
        """
        # broadcast

        self.scale = (torch.amax(x,dim=1)-torch.amin(x,dim=1))/self.n_levels
        self.zero_point = torch.amin(x,dim=1)
        """
        [out_channel, in_channel]/ [out_channel] -> [out_channel, in_channel]/ [out_channel, 1]
        """
        oc = self.scale.shape[0] # output channel nums
        self.scale = self.scale.reshape(oc, 1)
        self.zero_point = self.zero_point.reshape(oc, 1)




class QuantLinear(nn.Module):
    def __init__(self, raw_layer):
        super().__init__()
        self.raw_layer = raw_layer
        self.w_quantizer = BaseQuantizer(8)
        self.a_quantizer = BaseQuantizer(8)

    def forward(self, x):
        w_quant = self.w_quantizer(self.raw_layer.weight)
        b = self.raw_layer.bias

        x_quant = self.a_quantizer(x)
        out = F.linear(x_quant, w_quant, b)

        return out
    
    @property
    def weight(self):
        # 这里返回量化后的权重
        return self.w_quantizer(self.raw_layer.weight)
    

        

class DemoBlock(nn.Module):
    def __init__(self, feat_dims):
        super().__init__()
        self.layernorm = nn.LayerNorm(feat_dims)
        self.f1 = nn.Linear(feat_dims, feat_dims, bias = False)
        self.act_func = nn.GELU()
        self.f2 = nn.Linear(feat_dims, feat_dims, bias = False)
    
    def forward(self, x):
        res = x
        x = self.layernorm(x)
        x = self.f1(x)
        x = self.act_func(x)
        x = self.f2(x)
        out = res + x

        return out
    
class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32,64)
        self.blocks = nn.ModuleList()
        for i in range(3):
            block = DemoBlock(64)
            self.blocks.append(block)
        
    def forward(self, x):
        x = self.linear1(x)
        for block in self.blocks:
            x = block(x)

        return x
    
def replace_to_quantize_layer(our_net):
    for name, child in our_net.named_children():
        if isinstance(child, nn.Linear):
            quant_layer = QuantLinear(child)
            setattr(our_net, name, quant_layer)
            print(f'replace {name} to {quant_layer}')
        else:
            replace_to_quantize_layer(child)

# net = DemoNet()
# replace_to_quantize_layer(net)
# net.eval()

# for name, module in net.named_modules():
#     if isinstance(module, QuantLinear):
#         module.w_quantizer.do_calibration = True
#         module.a_quantizer.do_calibration = True
#         module.w_quantizer.name = f'calib {name} wq'
#         module.a_quantizer.name = f'calib {name} aq'

# calib_data = torch.randn([2,32])
# output = net(calib_data)

# end calibration

# for name, module in net.named_modules():
#     if isinstance(module, QuantLinear):
#         module.w_quantizer.do_calibration = False
#         module.a_quantizer.do_calibration = False

# new_input = torch.randn([2,32])
# output = net(new_input)
