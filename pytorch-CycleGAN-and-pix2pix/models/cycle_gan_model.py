import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        opt.use_sigmoid = opt.no_lsgan
        self.nlatent = opt.nlatent
        self.label_intensity = opt.label_intensity
        # self.label_intensity_styletransfer = opt.label_intensity_styletransfer
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['cycle_A', 'cycle_B', 'G_A_Classification0', 'G_B_Classification0','D_A_Classification0', 'D_B_Classification0', 'G_A_Classification1', 'G_B_Classification1', 'D_A_Classification1', 'D_B_Classification1', 'G_A_Classification2', 'G_B_Classification2', 'D_A_Classification2', 'D_B_Classification2', 'G_A_Classification3', 'G_B_Classification3', 'D_A_Classification3', 'D_B_Classification3']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
#        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        visual_names_B = []
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A_Classification0', 'D_B_Classification0', 'D_A_Classification1', 'D_B_Classification1', 'D_A_Classification2', 'D_B_Classification2', 'D_A_Classification3', 'D_B_Classification3', 'DeConv']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'DeConv']
            self.label_intensity_styletransfer = opt.label_intensity_styletransfer


        self.netG_A = networks.define_stochastic_G(nlatent=opt.nlatent, input_nc=opt.input_nc,
                                                     output_nc=opt.output_nc, ngf=opt.ngf,
                                                     which_model_netG=opt.which_model_netG,
                                                     norm=opt.norm, use_dropout=opt.use_dropout,
                                                     gpu_ids=opt.gpu_ids)

        self.netG_B = networks.define_stochastic_G(nlatent=opt.nlatent, input_nc=opt.input_nc,
                                                     output_nc=opt.output_nc, ngf=opt.ngf,
                                                     which_model_netG=opt.which_model_netG,
                                                     norm=opt.norm, use_dropout=opt.use_dropout,
                                                     gpu_ids=opt.gpu_ids)
        self.netDeConv = networks.define_InitialDeconv(gpu_ids=self.gpu_ids)
        enc_input_nc = opt.output_nc
        if opt.enc_A_B:
            enc_input_nc += opt.input_nc

        if self.isTrain:
            use_sigmoid = opt.no_lsgan

            self.netD_A_Classification0 = networks.define_D_B(input_nc=opt.input_nc,
                                              ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)
            self.netD_A_Classification1 = networks.define_D_B(input_nc=opt.input_nc,
                                              ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)
            self.netD_A_Classification2 = networks.define_D_B(input_nc=opt.input_nc,
                                              ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)
            self.netD_A_Classification3 = networks.define_D_B(input_nc=opt.input_nc,
                                              ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)

            self.netD_B_Classification0 = networks.define_D_B(input_nc=opt.input_nc,
                                              ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)
            self.netD_B_Classification1 = networks.define_D_B(input_nc=opt.input_nc,
                                              ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)
            self.netD_B_Classification2 = networks.define_D_B(input_nc=opt.input_nc,
                                              ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)
            self.netD_B_Classification3 = networks.define_D_B(input_nc=opt.input_nc,
                                              ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)



        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionRegression = networks.GANLoss(use_lsgan=True, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DeConv = torch.optim.Adam(self.netDeConv.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Classification0 = torch.optim.Adam(itertools.chain(self.netD_A_Classification0.parameters(), self.netD_B_Classification0.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Classification1 = torch.optim.Adam(itertools.chain(self.netD_A_Classification1.parameters(), self.netD_B_Classification1.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Classification2 = torch.optim.Adam(itertools.chain(self.netD_A_Classification2.parameters(), self.netD_B_Classification2.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Classification3 = torch.optim.Adam(itertools.chain(self.netD_A_Classification3.parameters(), self.netD_B_Classification3.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))



            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_DeConv)
            self.optimizers.append(self.optimizer_D_Classification0)
            self.optimizers.append(self.optimizer_D_Classification1)
            self.optimizers.append(self.optimizer_D_Classification2)
            self.optimizers.append(self.optimizer_D_Classification3)


            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, input, sign):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_B_original = input_B
        input_C = input['C' if AtoB else 'A']
        input_D = input['D' if AtoB else 'A']
        input_E = input['E' if AtoB else 'A']  
        self.sign = sign
        if self.isTrain:
            if sign == '0':
                self.random_number = np.zeros((1,4,1,1))
                self.random_number[0,0,0,0] = 1
            elif sign == '1':
                self.random_number = np.zeros((1,4,1,1))
                self.random_number[0,1,0,0] = 1
            elif sign == '2':
                self.random_number = np.zeros((1,4,1,1))
                self.random_number[0,2,0,0] = 1
            elif sign == '3':
                self.random_number = np.zeros((1,4,1,1))
                self.random_number[0,3,0,0] = 1
            elif sign == '4':
                self.random_number = np.random.rand(1,4,1,1)
                self.random_number = self.random_number/(self.random_number[0,0,0,0]+self.random_number[0,1,0,0]+self.random_number[0,2,0,0]+self.random_number[0,3,0,0])
            else:
                print("Setinput: Error occur when getting the 0, 1, 0to1")
            self.add_item = self.netDeConv(Variable(torch.FloatTensor([self.random_number]).view(1,4,1,1)).cuda(self.gpu_ids[0], async=True))

        else:
            # print(self.label_intensity_styletransfer)
            
            self.add_item = self.netDeConv(Variable(torch.FloatTensor(self.label_intensity_styletransfer).view(1,4,1,1)).cuda(self.gpu_ids[0], async=True))
        if self.isTrain:
            input_B = self.random_number[0,0,0,0]*input_B + self.random_number[0,1,0,0]*input_C + self.random_number[0,2,0,0]*input_D + self.random_number[0,3,0,0]*input_E
        else:
            input_B = self.label_intensity_styletransfer[0]*input_B + self.label_intensity_styletransfer[1]*input_C + self.label_intensity_styletransfer[2]*input_D + self.label_intensity_styletransfer[3]*input_E

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_C = input_C.cuda(self.gpu_ids[0], async=True)
            input_D = input_D.cuda(self.gpu_ids[0], async=True)
            input_E = input_E.cuda(self.gpu_ids[0], async=True)
            input_B_original = input_B_original.cuda(self.gpu_ids[0], async=True)
            # self.add_item = self.add_item.cuda(self.gpu_ids[0], async=True)

        self.input_A = input_A
        self.input_B = input_B
        self.input_B_original = input_B_original
        self.input_C = input_C
        self.input_D = input_D
        self.input_E = input_E
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_B_original = Variable(self.input_B_original)
        self.real_C = Variable(self.input_C)
        self.real_D = Variable(self.input_D)
        self.real_E = Variable(self.input_E)
        # self.add_item = Variable(self.add_item)

    def test(self):
        # self.add_item = Variable(self.add_item, volatile=True)
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A(self.real_A, self.add_item)
        self.rec_A = self.netG_B(self.fake_B, self.add_item)


    def backward_D_basic(self, netD, real, fake, random_number):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5 *random_number
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_basic_regression(self, netD, real, fake, truelabel):
        # Real
        pred_real = netD.forward(real)
        # print("pred_real",pred_real)
        loss_D_real = self.criterionRegression(pred_real, truelabel)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionRegression(pred_fake, [0,0,0,0])
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * 0.25
        # backward
        loss_D.backward()
        return loss_D


    def backward_D_A_Classification0(self, random_number):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A_Classification0 = self.backward_D_basic(self.netD_A_Classification0, self.real_B_original, fake_B, random_number)
        # print("backward_D_A_classification:", random_number)

    def backward_D_B_Classification0(self, random_number):
        if self.sign!='4':
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B_Classification0 = self.backward_D_basic(self.netD_B_Classification0, self.real_A, fake_A, random_number)
        elif self.sign == '4':
            fake_A0 = self.fake_A_pool.query(self.fake_A0)
            self.loss_D_B_Classification0 = self.backward_D_basic(self.netD_B_Classification0, self.real_A, fake_A0, random_number)
        else:
            print("error occur when backward DB classification0")
        # print("backward_D_B_classification:", random_number)

    def backward_D_A_Classification1(self, random_number):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A_Classification1 = self.backward_D_basic(self.netD_A_Classification1, self.real_C, fake_B, random_number)
        # print("backward_D_A_classification_0:", random_number)

    def backward_D_B_Classification1(self, random_number):
        if self.sign!='4':
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B_Classification1 = self.backward_D_basic(self.netD_B_Classification1, self.real_A, fake_A, random_number)
        elif self.sign == '4':
            fake_A1 = self.fake_A_pool.query(self.fake_A1)
            self.loss_D_B_Classification1 = self.backward_D_basic(self.netD_B_Classification1, self.real_A, fake_A1, random_number)
        else:
            print("error occur when backward DB classification1")

    def backward_D_A_Classification2(self, random_number):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A_Classification2 = self.backward_D_basic(self.netD_A_Classification2, self.real_D, fake_B, random_number)
        # print("backward_D_A_classification_0:", random_number)

    def backward_D_B_Classification2(self, random_number):
        if self.sign!='4':
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B_Classification2 = self.backward_D_basic(self.netD_B_Classification2, self.real_A, fake_A, random_number)
        elif self.sign == '4':
            fake_A2 = self.fake_A_pool.query(self.fake_A2)
            self.loss_D_B_Classification2 = self.backward_D_basic(self.netD_B_Classification2, self.real_A, fake_A2, random_number)
        else:
            print("error occur when backward DB classification2")


    def backward_D_A_Classification3(self, random_number):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A_Classification3 = self.backward_D_basic(self.netD_A_Classification3, self.real_E, fake_B, random_number)
        # print("backward_D_A_classification_0:", random_number)

    def backward_D_B_Classification3(self, random_number):
        if self.sign!='4':
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B_Classification3 = self.backward_D_basic(self.netD_B_Classification3, self.real_A, fake_A, random_number)
        elif self.sign == '4':
            fake_A3 = self.fake_A_pool.query(self.fake_A3)
            self.loss_D_B_Classification3 = self.backward_D_basic(self.netD_B_Classification3, self.real_A, fake_A3, random_number)
        else:
            print("error occur when backward DB classification3")


    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_GA = self.opt.lambda_GA
        lambda_GB = self.opt.lambda_GB
        lambda_GA_classification = self.opt.lambda_GA_classification
        lambda_GB_classification = self.opt.lambda_GB_classification
        lambda_GA_classification_0 = self.opt.lambda_GA_classification_0
        lambda_GB_classification_0 = self.opt.lambda_GB_classification_0
        lambda_GA_classification_0to1 = self.opt.lambda_GA_classification_0to1
        lambda_GB_classification_0to1 = self.opt.lambda_GB_classification_0to1

        # GAN loss D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A, self.add_item)

        if self.sign == '0':
            # print("Asign == 0:", self.random_number)
            self.loss_G_A_Classification0 = self.criterionGAN(self.netD_A_Classification0.forward(self.fake_B), True)*lambda_GA_classification
            self.loss_G_A_Classification1 = 0
            self.loss_G_A_Classification2 = 0
            self.loss_G_A_Classification3 = 0
#            self.loss_G_A_Classification0to1 = 0
        elif self.sign == '1':
            # print("Asign == 1:", self.random_number)
            self.loss_G_A_Classification0 = 0
            self.loss_G_A_Classification1 = self.criterionGAN(self.netD_A_Classification1.forward(self.fake_B), True)*lambda_GA_classification
            self.loss_G_A_Classification2 = 0
            self.loss_G_A_Classification3 = 0
#            self.loss_G_A_Classification0to1 = 0
        elif self.sign == '2':
            self.loss_G_A_Classification0 = 0
            self.loss_G_A_Classification1 = 0
            self.loss_G_A_Classification2 = self.criterionGAN(self.netD_A_Classification2.forward(self.fake_B), True)*lambda_GA_classification
            self.loss_G_A_Classification3 = 0
#            self.loss_G_A_Classification0to1 = 0
        elif self.sign == '3':
            self.loss_G_A_Classification0 = 0
            self.loss_G_A_Classification1 = 0
            self.loss_G_A_Classification2 = 0
            self.loss_G_A_Classification3 = self.criterionGAN(self.netD_A_Classification3.forward(self.fake_B), True)*lambda_GA_classification
#            self.loss_G_A_Classification0to1 = 0
        elif self.sign == '4':
            # print("Asign == 0.5:", self.random_number)
            self.loss_G_A_Classification0 = self.criterionGAN(self.netD_A_Classification0.forward(self.fake_B), True)*lambda_GA_classification*self.random_number[0,0,0,0]
            self.loss_G_A_Classification1 = self.criterionGAN(self.netD_A_Classification1.forward(self.fake_B), True)*lambda_GA_classification*self.random_number[0,1,0,0]
            self.loss_G_A_Classification2 = self.criterionGAN(self.netD_A_Classification2.forward(self.fake_B), True)*lambda_GA_classification*self.random_number[0,2,0,0]
            self.loss_G_A_Classification3 = self.criterionGAN(self.netD_A_Classification3.forward(self.fake_B), True)*lambda_GA_classification*self.random_number[0,3,0,0]
#            self.loss_G_A_Classification0to1 = self.criterionGAN(self.netD_A_Classification0to1.forward(self.fake_B), True)*lambda_GA_classification_0to1
        else:
            print("Error occurs when calculating the A GAN Loss")


        # GAN loss D_B(G_B(B))
        if self.sign != '4':
            self.fake_A = self.netG_B.forward(self.real_B, self.add_item)
        elif self.sign == '4':
            self.fake_A0 = self.netG_B.forward(self.real_B_original, self.add_item)
            self.fake_A1 = self.netG_B.forward(self.real_C, self.add_item)
            self.fake_A2 = self.netG_B.forward(self.real_D, self.add_item)
            self.fake_A3 = self.netG_B.forward(self.real_E, self.add_item)
        else:
            print('error occur when generate fakeA')
        if self.sign == '0':
            # print("Bsign == 0:", self.random_number)
            self.loss_G_B_Classification0 = self.criterionGAN(self.netD_B_Classification0.forward(self.fake_A), True)*lambda_GB_classification
            self.loss_G_B_Classification1 = 0
            self.loss_G_B_Classification2 = 0
            self.loss_G_B_Classification3 = 0

        elif self.sign == '1':
            # print("Bsign == 1:", self.random_number)
            self.loss_G_B_Classification0 = 0
            self.loss_G_B_Classification1 = self.criterionGAN(self.netD_B_Classification1.forward(self.fake_A), True)*lambda_GB_classification
            self.loss_G_B_Classification2 = 0
            self.loss_G_B_Classification3 = 0

        elif self.sign == '2':
            self.loss_G_B_Classification0 = 0
            self.loss_G_B_Classification1 = 0
            self.loss_G_B_Classification2 = self.criterionGAN(self.netD_B_Classification2.forward(self.fake_A), True)*lambda_GB_classification
            self.loss_G_B_Classification3 = 0

        elif self.sign == '3':
            self.loss_G_B_Classification0 = 0
            self.loss_G_B_Classification1 = 0
            self.loss_G_B_Classification2 = 0
            self.loss_G_B_Classification3 = self.criterionGAN(self.netD_B_Classification3.forward(self.fake_A), True)*lambda_GB_classification

        elif self.sign == '4':
            # print("Bsign == 0.5:", self.random_number)
            self.loss_G_B_Classification0 = self.criterionGAN(self.netD_B_Classification0.forward(self.fake_A0), True)*lambda_GB_classification*self.random_number[0,0,0,0]
            self.loss_G_B_Classification1 = self.criterionGAN(self.netD_B_Classification1.forward(self.fake_A1), True)*lambda_GB_classification*self.random_number[0,1,0,0]
            self.loss_G_B_Classification2 = self.criterionGAN(self.netD_B_Classification2.forward(self.fake_A2), True)*lambda_GB_classification*self.random_number[0,2,0,0]
            self.loss_G_B_Classification3 = self.criterionGAN(self.netD_B_Classification3.forward(self.fake_A3), True)*lambda_GB_classification*self.random_number[0,3,0,0]

        else:
            print("Error occurs when calculating the B GAN Loss")



        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B, self.add_item)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        if self.sign != '4':
            self.rec_B = self.netG_A.forward(self.fake_A, self.add_item)
        elif self.sign == '4':
            self.rec_B0 = self.netG_A.forward(self.fake_A0, self.add_item)
            self.rec_B1 = self.netG_A.forward(self.fake_A1, self.add_item)
            self.rec_B2 = self.netG_A.forward(self.fake_A2, self.add_item)
            self.rec_B3 = self.netG_A.forward(self.fake_A3, self.add_item)
        else:
            print("error occur when generate reconstruction B")
        if self.sign != '4':
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        elif self.sign == '4':
            self.loss_cycle_B0 = self.criterionCycle(self.rec_B0, self.real_B_original) * lambda_B * self.random_number[0,0,0,0]
            self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_C) * lambda_B * self.random_number[0,1,0,0]
            self.loss_cycle_B2 = self.criterionCycle(self.rec_B2, self.real_D) * lambda_B * self.random_number[0,2,0,0]
            self.loss_cycle_B3 = self.criterionCycle(self.rec_B3, self.real_E) * lambda_B * self.random_number[0,3,0,0]
            self.loss_cycle_B = self.loss_cycle_B0 + self.loss_cycle_B1 + self.loss_cycle_B2 + self.loss_cycle_B3
        else:
            print("error occur when generate loss cycleB")
        # combined loss
        self.loss_G = self.loss_cycle_A + self.loss_cycle_B + self.loss_G_A_Classification1 + self.loss_G_B_Classification1 + self.loss_G_A_Classification0 + self.loss_G_B_Classification0 + self.loss_G_A_Classification2 + self.loss_G_B_Classification2 + self.loss_G_A_Classification3 + self.loss_G_B_Classification3
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_DeConv.zero_grad()
        # G_A and G_B



        self.optimizer_G.zero_grad()
        self.backward_G()

        self.optimizer_G.step()
        # D_A and D_B
        if self.sign == '0':
            self.optimizer_D_Classification0.zero_grad()
            self.backward_D_A_Classification0(1)
            self.backward_D_B_Classification0(1)
            gnorm_D_Classification0 = torch.nn.utils.clip_grad_norm(itertools.chain(self.netD_A_Classification0.parameters(), self.netD_B_Classification0.parameters()), self.opt.max_gnorm)
            self.optimizer_D_Classification0.step()
        elif self.sign == '1':
            self.optimizer_D_Classification1.zero_grad()
            self.backward_D_A_Classification1(1)
            self.backward_D_B_Classification1(1)
            gnorm_D_Classification1 = torch.nn.utils.clip_grad_norm(itertools.chain(self.netD_A_Classification1.parameters(), self.netD_B_Classification1.parameters()), self.opt.max_gnorm)
            self.optimizer_D_Classification1.step()
        elif self.sign == '2':
            self.optimizer_D_Classification2.zero_grad()
            self.backward_D_A_Classification2(1)
            self.backward_D_B_Classification2(1)
            gnorm_D_Classification2 = torch.nn.utils.clip_grad_norm(itertools.chain(self.netD_A_Classification2.parameters(), self.netD_B_Classification2.parameters()), self.opt.max_gnorm)
            self.optimizer_D_Classification2.step()
        elif self.sign == '3':
            self.optimizer_D_Classification3.zero_grad()
            self.backward_D_A_Classification3(1)
            self.backward_D_B_Classification3(1)
            gnorm_D_Classification3 = torch.nn.utils.clip_grad_norm(itertools.chain(self.netD_A_Classification3.parameters(), self.netD_B_Classification3.parameters()), self.opt.max_gnorm)
            self.optimizer_D_Classification3.step()
        elif self.sign == '4':
            self.optimizer_D_Classification0.zero_grad()
            self.backward_D_A_Classification0(self.random_number[0,0,0,0])
            self.backward_D_B_Classification0(self.random_number[0,0,0,0])
            gnorm_D_Classification0 = torch.nn.utils.clip_grad_norm(itertools.chain(self.netD_A_Classification0.parameters(), self.netD_B_Classification0.parameters()), self.opt.max_gnorm)
            self.optimizer_D_Classification0.step()

            self.optimizer_D_Classification1.zero_grad()
            self.backward_D_A_Classification1(self.random_number[0,1,0,0])
            self.backward_D_B_Classification1(self.random_number[0,1,0,0])
            gnorm_D_Classification1 = torch.nn.utils.clip_grad_norm(itertools.chain(self.netD_A_Classification1.parameters(), self.netD_B_Classification1.parameters()), self.opt.max_gnorm)
            self.optimizer_D_Classification1.step()

            self.optimizer_D_Classification2.zero_grad()
            self.backward_D_A_Classification2(self.random_number[0,2,0,0])
            self.backward_D_B_Classification2(self.random_number[0,2,0,0])
            gnorm_D_Classification2 = torch.nn.utils.clip_grad_norm(itertools.chain(self.netD_A_Classification2.parameters(), self.netD_B_Classification2.parameters()), self.opt.max_gnorm)
            self.optimizer_D_Classification2.step()

            self.optimizer_D_Classification3.zero_grad()
            self.backward_D_A_Classification3(self.random_number[0,3,0,0])
            self.backward_D_B_Classification3(self.random_number[0,3,0,0])
            gnorm_D_Classification3 = torch.nn.utils.clip_grad_norm(itertools.chain(self.netD_A_Classification3.parameters(), self.netD_B_Classification3.parameters()), self.opt.max_gnorm)
            self.optimizer_D_Classification3.step()

        else:
            print("Error occurs when optimizing the parameters")
        gnorm_DeConv = torch.nn.utils.clip_grad_norm(self.netDeConv.parameters(), self.opt.max_gnorm)
        self.optimizer_DeConv.step()
