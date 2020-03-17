from torch.autograd import Variable
from collections import OrderedDict
from .base_model import BaseModel
from . import networks
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.label_intensity = opt.label_intensity
        self.label_intensity_styletransfer = opt.label_intensity_styletransfer
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # self.visual_names = ['real_A', 'fake_B']
        self.visual_names = ['fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G', 'DeConv']

        self.netG = networks.define_stochastic_G(nlatent=opt.nlatent, input_nc=opt.input_nc,
                                                     output_nc=opt.output_nc, ngf=opt.ngf,
                                                     which_model_netG=opt.which_model_netG,
                                                     norm=opt.norm, use_dropout=opt.use_dropout,
                                                     gpu_ids=opt.gpu_ids)
        self.netDeConv = networks.define_InitialDeconv(gpu_ids=self.gpu_ids)
        self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, input, beta, alpha):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']
        self.add_item = self.netDeConv(Variable(torch.FloatTensor([self.label_intensity_styletransfer]).view(1,4,1,1)).cuda(self.gpu_ids[0], async=True))

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A, self.add_item)

    def get_current_visuals(self):
        # return OrderedDict([('real_A', self.real_A), ('fake_B', self.fake_B)])
        return OrderedDict([('fake_B', self.fake_B)])
