import torch
import torch.nn as nn
import torch.nn.functional as F
eps = 1e-4
quantile_scale = 0.99
torch.random.manual_seed(3)
DEBUG=True
DEBUG=False

''' base quantizer'''
class BaseQuantizer(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.zero_point = None
        self.scale = None

        self.do_calibration = False
        self.name = 'base quantizer'

    def forward(self,x):
        if self.do_calibration:
            self.calibration(x)
        x_int = (x/self.scale-self.zero_point).round().clamp(0,self.n_levels-1)
        x_dequant = (x_int+self.zero_point)*self.scale
        return x_dequant
    
    def calibration(self,x):
        self.scale = (x.max()-x.min())/self.n_levels
        if torch.any(self.scale == 0):
            print(f'Warning: scale contains zero value(s)! Replacing with eps = {eps}')
            self.scale = torch.where(self.scale == 0, eps, self.scale)
        self.zero_point = x.min()/self.scale

class ActStaticScaleQuantizer(nn.Module):
    def __init__(self, n_bits=8, dim=-1):
        super().__init__()
        self.n_bits = n_bits
        self.dim = dim
        self.n_levels = 2**self.n_bits

        self.name = 'for param after softmax -- we can set scale and zeropoint'

    def forward(self,x):
        scale = 1/self.n_levels
        zero_point = 0
        x_int = (x/scale-zero_point).round().clamp(0,self.n_levels-1)
        x_dequant = (x_int+zero_point)*scale
        return x_dequant
    
'''Activation Dynamic Quantization ———— per tensor'''
# NOTICE: for activation, x:[batch, feat_num], so per tensor mean each batch has different quant data
class ActDynamicQuantizer(nn.Module):
    def __init__(self, n_bits=8, dim=-1):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.dim=dim

        self.name = 'activation dynamic quantizer'

    def forward(self,x):
        scale = (torch.amax(x,dim=self.dim,keepdim=True)-torch.amin(x,dim=self.dim,keepdim=True))/self.n_levels
        if torch.any(scale == 0):
            print(f'Warning: scale contains zero value(s)! Replacing with eps = {eps}')
            scale = torch.where(scale == 0, eps, scale)
        zero_point = torch.amin(x,dim=self.dim,keepdim=True)/scale
        if torch.isinf(zero_point).any():
            # breakpoint()
            print(f'Warning: zeropoint contains inf value(s)! Replacing with max_num = {int(2**15 - 1)}')
            zero_point = torch.where(torch.isposinf(zero_point), int(2**15 - 1), zero_point)
            zero_point = torch.where(torch.isneginf(zero_point), -2**15, zero_point)
        # breakpoint()
        x_int = (x/scale-zero_point).round().clamp(0,self.n_levels-1)
        x_dequant = (x_int+zero_point)*scale
        if torch.isnan(x_dequant).any():
            breakpoint()
        return x_dequant
    
'''Activation Dynamic Quantization ———— per group'''
class ActPGDynamicQuantizer(nn.Module):
    def __init__(self, group_size, dim, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.group_size = group_size
        self.dim = dim

        self.name = 'activation dynamic quantizer ———— per group'

    def forward(self,x):
        # print(f'x.shape={x.shape}')
        x_shape = x.shape # 16,30,1,64
        batch=x_shape[0] # 16
        feat_num=1
        for i in x_shape[1:]:
            feat_num*=i
        assert batch*feat_num % self.group_size == 0, breakpoint()
        x = x.reshape(batch*feat_num//self.group_size, self.group_size)

        scale = (torch.amax(x,dim=self.dim,keepdim=True)-torch.amin(x,dim=self.dim,keepdim=True))/self.n_levels
        if torch.any(scale == 0):
            print(f'Warning: scale contains zero value(s)! Replacing with eps = {eps}')
            scale = torch.where(scale == 0, eps, scale)
        zero_point = torch.amin(x,dim=self.dim,keepdim=True)/scale
        x_int = (x/scale-zero_point).round().clamp(0, self.n_levels-1)
        x_dequant = (x_int+zero_point)*scale

        x_dequant=x_dequant.reshape(x_shape)
        if torch.isnan(x_dequant).any():
            breakpoint()
        return x_dequant
    

'''Weight Quantization(Quantile) ———— per group'''
class WeightPGQuantileQuantizer(nn.Module):
    def __init__(self, n_bits=8, group_size = 32):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.group_size = group_size
        self.zero_point = None
        self.scale = None

        self.do_calibration = False
        self.name = 'weight per_channel quantizer'

    def forward(self,x):
        OC,IC=x.shape
        assert OC*IC%self.group_size==0
        x=x.reshape(OC*IC//self.group_size,self.group_size)
        if self.do_calibration:
            self.calibration(x)
        x_int = (x/self.scale-self.zero_point).round().clamp(0,self.n_levels-1)
        x_dequant = (x_int+self.zero_point)*self.scale
        x_dequant = x_dequant.reshape(OC,IC)

        return x_dequant
    
    def calibration(self,x):
        k_min = int((1-quantile_scale)*x.shape[1])
        k_max = int(quantile_scale*x.shape[1])
        if k_min == 0:
            k_min = 1
        min_vals, _ = torch.kthvalue(x, k_min, keepdim=True)
        max_vals, _ = torch.kthvalue(x, k_max, keepdim=True)

        self.scale = (max_vals-min_vals)/self.n_levels
        if torch.any(self.scale == 0):
            print(f'Warning: scale contains zero value(s)! Replacing with eps = {eps}')
            self.scale = torch.where(self.scale == 0, eps, self.scale)
        self.zero_point = min_vals/self.scale

'''Weight Quantization ———— per group'''
class WeightPGQuantizer(nn.Module):
    def __init__(self, group_size, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.group_size = group_size
        self.zero_point = None
        self.scale = None

        self.do_calibration = False
        self.name = 'weight per_group quantizer'

    def forward(self,x):
        OC,IC=x.shape
        assert OC*IC%self.group_size==0
        x=x.reshape(OC*IC//self.group_size,self.group_size)
        if self.do_calibration:
            self.calibration(x)
        x_int = (x/self.scale-self.zero_point).round().clamp(0,self.n_levels-1)
        if torch.isnan(x_int).any():
            breakpoint()
        x_dequant = (x_int+self.zero_point)*self.scale
        x_dequant = x_dequant.reshape(OC,IC)
        if torch.isnan(x_dequant).any():
            breakpoint()

        return x_dequant
    
    def calibration(self,x):
        self.scale = (torch.amax(x,dim=1,keepdim=True)-torch.amin(x,dim=1,keepdim=True))/self.n_levels
        if torch.any(self.scale == 0):
            print(f'Warning: scale contains zero value(s)! Replacing with eps = {eps}')
            self.scale = torch.where(self.scale == 0, eps, self.scale)
        self.zero_point = torch.amin(x,dim=1,keepdim=True)/self.scale

'''Weight Quantization ———— per channel'''
class WeightPCQuantizer(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.zero_point = None
        self.scale = None

        self.do_calibration = False
        self.name = 'weight per_channel quantizer'

    def forward(self,x):
        if self.do_calibration:
            self.calibration(x)
        x_int = (x/self.scale-self.zero_point).round().clamp(0,self.n_levels-1)
        x_dequant = (x_int+self.zero_point)*self.scale
        return x_dequant
    
    def calibration(self,x):
        # PER CHANNEL torch.amax
        self.scale = (torch.amax(x,dim=1,keepdim=True)-torch.amin(x,dim=1,keepdim=True))/self.n_levels
        # 检查是否有任何零值需要被替换
        if torch.any(self.scale == 0):
            print(f'Warning: scale contains zero value(s)! Replacing with eps = {eps}')
            self.scale = torch.where(self.scale == 0, eps, self.scale)
        self.zero_point = torch.amin(x,dim=1,keepdim=True)/self.scale

'''Weight Quantization(Quantile) ———— per channel'''
class WeightPCQuantileQuantizer(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.zero_point = None
        self.scale = None

        self.do_calibration = False
        self.name = 'weight per_channel quantizer + quantile'

    def forward(self,x):
        if self.do_calibration:
            self.calibration(x)
        x_int = (x/self.scale-self.zero_point).round().clamp(0,self.n_levels-1)
        x_dequant = (x_int+self.zero_point)*self.scale
        return x_dequant
    
    def calibration(self,x):
        # PER CHANNEL + quantile torch.kthvalue()
        k_min = int(0.01*x.shape[1])
        k_max = int(0.99*x.shape[1])
        if k_min == 0:
            k_min = 1
        min_vals, _ = torch.kthvalue(x,k_min,keepdim=True)
        max_vals, _ = torch.kthvalue(x,k_max,keepdim=True)
        self.scale = (max_vals-min_vals)/self.n_levels
        if torch.any(self.scale == 0):
            print(f'Warning: scale contains zero value(s)! Replacing with eps = {eps}')
            self.scale = torch.where(self.scale == 0, eps, self.scale)
        self.zero_point = min_vals/self.scale



class DemoBlock(nn.Module):
    def __init__(self, feat_dims):
        super().__init__()
        self.layernorm = nn.LayerNorm(feat_dims)
        self.f1 = nn.Linear(feat_dims, feat_dims, bias=False)
        self.f1.weight.data[2]=torch.randn_like(self.f1.weight.data[2])
        self.f1.weight.data[1][5] = self.f1.weight.data.max()*2
        self.act_func = nn.GELU()
        self.f2 = nn.Linear(feat_dims, feat_dims, bias=False)

    def forward(self,x):
        res = x
        x = self.layernorm(x)
        x = self.f1(x)
        x = self.act_func(x)
        x = self.f2(x)
        out = res+x
        return out

class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 128)
        self.blocks = nn.ModuleList()
        for i in range(3):
            block = DemoBlock(128)
            self.blocks.append(block)
    
    def forward(self,x):
        x = self.linear1(x)
        for block in self.blocks:
            x = block(x)
            
        return x


def replace_to_quantize_layer(our_net):
    for name, child in our_net.named_children():
        if isinstance(child, nn.Linear):
            quant_layer = QuantLinear(child, name)
            setattr(our_net,name,quant_layer)
            print(f'replace {name} to {quant_layer}')
        else:
            replace_to_quantize_layer(child)

if __name__ == "__main__":

    '''quantize all linear layers of our net'''
    net = DemoNet()
    validation_data = torch.randn([100,32])
    # fp16 output
    raw_valid_output = net(validation_data)

    replace_to_quantize_layer(net)

    # get calibration
    net.eval()

    # 
    # for name,module in net.named_modules():
    #     module.name=name

    for name, module in net.named_modules():
        if isinstance(module, QuantLinear):
            module.w_quantizer.do_calibration = True
            module.a_quantizer.do_calibration = True
            module.w_quantizer.name = f'calib {name} wq'
            module.a_quantizer.name = f'calib {name} aq'

    calib_data = torch.randn([2, 32])
    a = net(calib_data)

    # validation with quantized layer
    for name, module in net.named_modules():
        if isinstance(module, QuantLinear):
            module.w_quantizer.do_calibration = False
            module.a_quantizer.do_calibration = False

    quanted_valid_output=net(validation_data)

    # calculate MSE between fp16 layer output and int8 layer output
    MSE=F.mse_loss(quanted_valid_output,raw_valid_output)
    print(f"MSE  {MSE}")



