import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from modules import ResnetBlock, CondInstanceNorm, TwoInputSequential, CINResnetBlock, InstanceNorm2d
from torch.optim import lr_scheduler

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(InstanceNorm2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, norm='instance', which_model_netG='resnet',
             use_dropout=False, gpu_ids=[]):

    netG = None
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())

    norm_layer = get_norm_layer(norm_type=norm)

    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                           use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG


def define_stochastic_G(nlatent, input_nc, output_nc, ngf, norm='instance',
                        which_model_netG='resnet', use_dropout=False, gpu_ids=[]):

    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    norm_layer = CondInstanceNorm

    netG = CINResnetGenerator(nlatent, input_nc, output_nc, ngf, norm_layer=norm_layer,
                              use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG




def define_D_B(input_nc, ndf, which_model_netD, norm, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = Discriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD



def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    if out_f is not None:
        out_f.write(net.__repr__()+"\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()


##############################################################################
# Network Classes
##############################################################################

######################################################################
# Modified version of ResnetGenerator that supports stochastic mappings
# using Conditonal instance norm (can support CBN easily)
######################################################################
class CINResnetGenerator(nn.Module):
    def __init__(self, nlatent, input_nc, output_nc, ngf=64, norm_layer=CondInstanceNorm,
                 use_dropout=False, n_blocks=9, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(CINResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        instance_norm = functools.partial(InstanceNorm2d, affine=True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2*ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(2*ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4*ngf, nlatent),
            nn.ReLU(True)
        ]
        
        for i in range(n_blocks):
            model += [CINResnetBlock(x_dim=4*ngf, z_dim=nlatent, padding_type=padding_type,
                                     norm_layer=norm_layer, use_dropout=use_dropout, use_bias=True)]

        model += [

            nn.ConvTranspose2d(4*ngf, 2*ngf,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=True),
            norm_layer(2*ngf, nlatent),
            nn.ReLU(True),

            nn.ConvTranspose2d(2*ngf, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]


        self.model = TwoInputSequential(*model)

    def forward(self, input, noise):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, noise), self.gpu_ids)
        else:
            return self.model(input, noise)


######################################################################
# ResnetGenerator for deterministic mappings
######################################################################
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=InstanceNorm2d, use_dropout=False,
                 n_blocks=9, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2*ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(2*ngf),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4*ngf),
            nn.ReLU(True),
        ]

        for i in range(n_blocks):
            model += [ResnetBlock(4*ngf, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=True)]

        model += [

            nn.ConvTranspose2d(4*ngf, 2*ngf,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=True),
            norm_layer(2*ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(2*ngf, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

######################################################################
# Discriminator that supports stochastic mappings
# using Conditonal instance norm (can support CBN easily)
######################################################################
class CINDiscriminator(nn.Module):
    def __init__(self, nlatent, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, gpu_ids=[]):
        """
        nlatent: number of channles in both latent codes (or one of them - depending on the model)
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(CINDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        use_bias = True

        kw = 4
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, 2*ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2*ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(2*ndf, 4*ndf,
                      kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(4*ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4*ndf, 5*ndf,
                      kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(5*ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(5*ndf, 1, kernel_size=kw, stride=1, padding=1)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = TwoInputSequential(*sequence)

    def forward(self, input, noise):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, noise), self.gpu_ids)
        else:
            return self.model(input, noise)


######################################################################
# Discriminator network
######################################################################

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        n_layers=3
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        #if target_is_real:
        #    create_label = ((self.real_label_var is None) or
        #                    (self.real_label_var.numel() != input.numel()))
        #    if create_label:
        #        real_tensor = self.Tensor(input.size()).fill_(target_is_real) #########change from self.real_label to target_is_real April 20, 21:20#####################
        #        self.real_label_var = Variable(real_tensor, requires_grad=False)
        #    target_tensor = self.real_label_var
        #else:
        #    create_label = ((self.fake_label_var is None) or
        #                    (self.fake_label_var.numel() != input.numel()))
        #    if create_label:
        #        fake_tensor = self.Tensor(input.size()).fill_(target_is_real)
        #        self.fake_label_var = Variable(fake_tensor, requires_grad=False)
        #    target_tensor = self.fake_label_var
        if isinstance(target_is_real, list):
            # target_tensor = Variable(self.Tensor(target_is_real), requires_grad=False)
            # print(input.size()[2], input.size()[3])
            target_is_real_arr = np.asarray(target_is_real)
            target_is_real_arr = np.expand_dims(target_is_real_arr, axis = -1)
            target_is_real_arr = np.expand_dims(target_is_real_arr, axis = -1)
            target_is_real_arr = np.tile(target_is_real_arr,(1,1,30,30))
            # print(target_is_real_arr.shape)
            # print(target_is_real_arr[0,0:4,0,0])
            target_is_real_arr = list(target_is_real_arr)
            # print(target_is_real_arr.shape)
            target_tensor = Variable(self.Tensor(target_is_real_arr), requires_grad=False)
        else:
            # print(input.size()[0], input.size()[1])
            target_tensor = Variable(self.Tensor(input.size()).fill_(target_is_real), requires_grad=False)
        # print(target_tensor)
        # if ((target_is_real>0.01) and (target_is_real <0.99)):
        #    print('target_is_real:', target_is_real)
        #    print('target tensor:', target_tensor)
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class InitialDeconvolution(nn.Module):
    def __init__(self, input_nc = 1, output_nc =16, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(InitialDeconvolution, self).__init__()
        self.gpu_ids = gpu_ids
        self.input_nc = input_nc
        self.output_nc = output_nc
        model = []
        model += [nn.ConvTranspose2d(4, 16, kernel_size=(1,1), stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

def define_InitialDeconv(init_type='normal', gpu_ids=[]):
    net_Deconv = None
    net_Deconv = InitialDeconvolution()
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        net_Deconv.cuda(gpu_ids[0])
    # init_weights(net_Deconv, init_type=init_type)
    net_Deconv.apply(weights_init)
    return net_Deconv

def define_D_Regression4(input_nc, ndf, which_model_netD, norm, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = Discriminator_edges_Regression4(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD

class Discriminator_edges_Regression4(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(Discriminator_edges_Regression4, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        n_layers=3
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 4, kernel_size=kw, stride=1, padding=padw)]



        self.model = nn.Sequential(*sequence)
        

    def forward(self, input):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
