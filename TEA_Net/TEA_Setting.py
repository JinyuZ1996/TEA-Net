# @Author: Jinyu Zhang
# @Time: 2022/8/24 14:27
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

class Settings:
    def __init__(self):
        '''
            block 1: the parameters for model training
        '''
        self.lr_A = 0.005
        self.lr_B = 0.004
        self.keep_prob = 0.9
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.epochs = 100   # 100 for DOUBAN 80 for AMAZON
        self.verbose = 10
        self.gpu_num = '0'

        '''
            block 2: the parameters for TEA_Model.py
        '''
        self.embedding_size = 16
        # self.n_fold = int(self.embedding_size / 2)
        self.n_fold = self.embedding_size
        self.layer_size = '['+str(self.embedding_size)+']'
        self.num_layers = 3
        self.padding_int = 0

        self.alpha = 0.2    # temperature——coff
        self.beta = 0.1
        # FOR EA-A
        self.num_heads = 2      # 2， 4， 8， 16
        self.dim_coefficient = 2 * self.num_heads
        self.regular_rate_att = 1e-7
        self.l2_regular_rate = 1e-7

        '''
            block 3: the parameters for file paths
            Datasets could be download from https://bitbucket.org/jinyuz1996/tea-net-data/src/main/
        '''
        self.dataset = 'Douban'  # Douban or Amazon
        self.path_train = 'data/' + self.dataset + '/train_data.txt'
        self.path_test = 'data/' + self.dataset + '/test_data.txt'
        self.path_dict_A = 'data/' + self.dataset + '/A_dict.txt'
        self.path_dict_B = 'data/' + self.dataset + '/B_dict.txt'
        self.path_dict_U = 'data/' + self.dataset + '/U_dict.txt'


        self.checkpoint = 'checkpoint/trained_model.ckpt'

        self.fast_running = False
        self.fast_ratio = 0.8