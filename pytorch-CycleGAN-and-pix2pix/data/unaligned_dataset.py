import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_D = os.path.join(opt.dataroot, opt.phase + 'D')
        self.dir_E = os.path.join(opt.dataroot, opt.phase + 'E')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.C_paths = make_dataset(self.dir_C)
        self.D_paths = make_dataset(self.dir_D)
        self.E_paths = make_dataset(self.dir_E)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths)
        self.D_paths = sorted(self.D_paths)
        self.E_paths = sorted(self.E_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        self.E_size = len(self.E_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
            index_C = index % self.C_size
            index_D = index % self.D_size
            index_E = index % self.E_size
        else:
            index_B = random.randint(0, self.B_size - 1)
            index_C = random.randint(0, self.C_size - 1)
            index_D = random.randint(0, self.D_size - 1)
            index_E = random.randint(0, self.E_size - 1)
        B_path = self.B_paths[index_B]
        C_path = self.C_paths[index_C]
        D_path = self.D_paths[index_D]
        E_path = self.E_paths[index_E]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')
        D_img = Image.open(D_path).convert('RGB')
        E_img = Image.open(E_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        C = self.transform(C_img)
        D = self.transform(D_img)
        E = self.transform(E_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
                'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'D_paths': D_path, 'E_paths': E_path}

    def __len__(self):
        return max(self.A_size, self.B_size, self.C_size, self.D_size, self.E_size)

    def name(self):
        return 'UnalignedDataset'
