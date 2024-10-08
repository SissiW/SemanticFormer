import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch.nn.functional as F
import torch
import LDA_SLIC
import CEGCN
import pdb

# import spectral

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# FLAG =1, indian
# FLAG =2, paviaU
# FLAG =3, salinas
samples_type = ['ratio', 'same_num'][0]

# for (FLAG, curr_train_ratio) in [(1,5),(1,10),(1,15),(1,20),(1,25),
# (2,5),(2,10),(2,15),(2,20),(2,25),
# (3,5),(3,10),(3,15),(3,20),(3,25)]:
# for (FLAG, curr_train_ratio, Scale) in [(1, 0.1, 100)]:
for (FLAG, curr_train_ratio, Scale, learning_rate) in [(1, 0.1, 100, 5e-4)]:  # ,(3,0.01,100)]:
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []

    Seed_List = [0,1,2,3,4]
    # Seed_List = [1, 2, 3, 4, 5]

    if FLAG == 1:
        data_mat = sio.loadmat('/home/wxx/Work/Working2/HSI_code/MSSG-UNet-main/HyperImage_data/indian/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('/home/wxx/Work/Working2/HSI_code/MSSG-UNet-main/HyperImage_data/indian/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']

        # train_ratio = 0.05 
        val_ratio = 0.01
        class_count = 16
        learning_rate = 5e-4
        max_epoch = 600
        dataset_name = "indian_"
        alpha, beta, garma, lamb = 0, 0, 0, 4
        # superpixel_scale=100
        pass
    if FLAG == 2:
        data_mat = sio.loadmat('.\\HyperImage_data\\paviaU\\PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('.\\HyperImage_data\\paviaU\\PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']
        alpha, beta, garma, lamb = 0, 0, 0, 4

        # train_ratio = 0.01 
        val_ratio = 0.01
        class_count = 9
        # learning_rate = 5e-4 
        max_epoch = 600
        dataset_name = "paviaU_"
        # superpixel_scale = 100
        pass
    '''if FLAG == 3:
        data_mat = sio.loadmat('/home/wxx/Work/Working2/HSI_code/MSSG-UNet-main/HyperImage_data/Salinas/Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('/home/wxx/Work/Working2/HSI_code/MSSG-UNet-main/HyperImage_data/Salinas/Salinas_gt.mat')
        gt = gt_mat['salinas_gt']

        # train_ratio = 0.01
        val_ratio = 0.01
        class_count = 16
        learning_rate = 5e-4
        max_epoch = 600
        alpha, beta, garma, lamb = 0, 0, 0, 2
        dataset_name = "salinas_"
        # superpixel_scale = 100
        pass'''
    if FLAG == 3:
        data_mat = sio.loadmat('/home/wxx/Work/Working2/HSI_code/MSSG-UNet-main/HyperImage_data/houston/Houston.mat')
        data = data_mat['Houston']
        gt_mat = sio.loadmat('/home/wxx/Work/Working2/HSI_code/MSSG-UNet-main/HyperImage_data/houston/Houston_gt.mat')
        gt = gt_mat['Houston_gt']


        # train_ratio = 0.01
        val_ratio = 0.1
        class_count = 15
        learning_rate = 1e-4
        max_epoch = 800
        alpha, beta, garma, lamb = 0, 0, 0, 1
        dataset_name = "houston_"
        # superpixel_scale = 100
        pass
    if FLAG == 4:
        data_mat = sio.loadmat('.\\HyperImage_data\\KSC\\KSC.mat')
        data = data_mat['KSC']
        gt_mat = sio.loadmat('.\\HyperImage_data\\KSC\\KSC_gt.mat')
        gt = gt_mat['KSC_gt']

        # train_ratio = 0.05
        val_ratio = 0.01
        class_count = 13
        learning_rate = 5e-4
        max_epoch = 1000
        dataset_name = "KSC_"
        # superpixel_·scale = 200
        pass
    ###########
    superpixel_scale = Scale  #########################
    train_samples_per_class = curr_train_ratio
    val_samples = class_count
    train_ratio = curr_train_ratio
    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data.shape

    orig_data = data
    height, width, bands = data.shape  # (145, 145, 200)
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])  # (145, 145, 200)
    # pdb.set_trace()

    # gt_reshape=np.reshape(gt, [-1])
    # for i in range(class_count):
    #     idx = np.where(gt_reshape == i + 1)[-1]
    #     samplesCount = len(idx)
    #     print(samplesCount)

    def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
        '''
        get classification map , then save to given path
        :param label: classification label, 2D
        :param name: saving path and file's name
        :param scale: scale of image. If equals to 1, then saving-size is just the label-size
        :param dpi: default is OK
        :return: null
        '''
        fig, ax = plt.subplots()
        numlabel = np.array(label)
        # v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        foo_fig = plt.gcf()  # 'get current figure'
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
        pass
    def GT_To_One_Hot(gt, class_count):
        '''
        Convet Gt to one-hot labels
        :param gt:
        :param class_count:
        :return:
        '''
        GT_One_Hot = []
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                temp = np.zeros(class_count, dtype=np.float32)
                if gt[i, j] != 0:
                    temp[int(gt[i, j]) - 1] = 1
                GT_One_Hot.append(temp)
        GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
        return GT_One_Hot


    for curr_seed in Seed_List:

        random.seed(curr_seed)
        torch.manual_seed(curr_seed)
        gt_reshape = np.reshape(gt, [-1])
        train_rand_idx = []
        val_rand_idx = []
        if samples_type == 'ratio':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                rand_list = [i for i in range(samplesCount)]
                rand_idx = random.sample(rand_list,
                                         np.ceil(samplesCount * train_ratio).astype('int32'))
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)
            train_rand_idx = np.array(train_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)

            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)

            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx

            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index

            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)

        if samples_type == 'same_num':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                real_train_samples_per_class = train_samples_per_class
                rand_list = [i for i in range(samplesCount)]
                if real_train_samples_per_class > samplesCount:
                    real_train_samples_per_class = samplesCount
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)
            train_rand_idx = np.array(train_rand_idx)
            val_rand_idx = np.array(val_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)

            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)

            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx

            val_data_count = int(val_samples)
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)

            test_data_index = test_data_index - val_data_index

            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)

        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass

        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass

        Test_GT = np.reshape(test_samples_gt, [m, n])

        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass

        train_samples_gt = np.reshape(train_samples_gt, [height, width])
        test_samples_gt = np.reshape(test_samples_gt, [height, width])
        val_samples_gt = np.reshape(val_samples_gt, [height, width])


        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
        val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m * n, class_count])

        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m * n, class_count])

        ls = LDA_SLIC.LDA_SLIC(data=data, labels=gt, n_component=class_count - 1, class_count=class_count, gt=gt)
        tic0 = time.time()
        # Q, S, A, Seg = ls.simple_superpixel(scale=superpixel_scale)
        Q, S, A = ls.simple_superpixel(scale=superpixel_scale)
        # Q, S, A = ls.simple_superpixel_no_LDA(scale=superpixel_scale)
        toc0 = time.time()
        # LDA_SLIC_Time = toc0 - tic0
        create_graph_time = toc0 - tic0
        # np.save(dataset_name+'Seg',Seg)
        # print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
        print("create_graph costs time: {}".format(create_graph_time))
        Q = torch.from_numpy(Q).to(device)
        A = torch.from_numpy(A).to(device)

        train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        net_input = np.array(data, np.float32)
        net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)  # torch.Size([145, 145, 200])

        if dataset_name == "indian_":
            net = CEGCN.CEGCN(height, width, bands, class_count, Q, A, model='smoothed')
        else:
            net = CEGCN.CEGCN(height, width, bands, class_count, Q, A)

        print("parameters", net.parameters(), len(list(net.parameters())))
        # paras=list(net.parameters())
        # k=0
        # for i in paras:
        #     l=1
        #     for j in i.size():
        #         l*=j
        #     k=k+l
        # print('参数总量：' + str(k))
        # print(aa)
        net.to(device)


        def consis_loss(logps, temp=0.5, temp_p=1.2):
            ps = [torch.exp(p) for p in logps]
            ps_sharp = [(torch.pow(p, 1. / temp_p) / torch.sum(torch.pow(p, 1. / temp_p), dim=1, keepdim=True)).detach() for p in logps]
            sum_p = 0.
            for p in ps_sharp:
                sum_p = sum_p + p
            avg_p = sum_p / len(ps)
            # p2 = torch.exp(logp2)

            # sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1,
            #                                                    keepdim=True)).detach()
            loss = 0.
            for p in ps:
                loss += torch.mean((p - avg_p).pow(2).sum(1))
            loss = loss / len(ps)
            return alpha * loss


        # def consis_loss(CNN_result, GCN_result):
        #     return torch.norm(CNN_result - GCN_result)


        def softmax_kl_loss(input_logits, target_logits):
            """Takes softmax on both sides and returns KL divergence
            Note:
            - Returns the sum over all examples. Divide by the batch size afterwards
              if you want the mean.
            - Sends gradients to inputs but not the targets.
            """
            assert input_logits.size() == target_logits.size()
            input_log_softmax = F.log_softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            return F.kl_div(input_log_softmax, target_softmax, size_average=False)


        def softmax_mse_loss(input_logits, target_logits):
            """Takes softmax on both sides and returns MSE loss
            Note:
            - Returns the sum over all examples. Divide by the batch size afterwards
              if you want the mean.
            - Sends gradients to inputs but not the targets.
            """
            assert input_logits.size() == target_logits.size()
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            num_classes = input_logits.size()[1]
            # pdb.set_trace()
            return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


        def symmetric_mse_loss(input1, input2):
            """Like F.mse_loss but sends gradients to both directions
            Note:
            - Returns the sum over all examples. Divide by the batch size afterwards
              if you want the mean.
            - Sends gradients to both input1 and input2.
            """
            assert input1.size() == input2.size()
            num_classes = input1.size()[1]
            return torch.sum((input1 - input2) ** 2) / num_classes


        def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
            real_labels = reallabel_onehot
            we = -torch.mul(real_labels, torch.log(predict))
            we = torch.mul(we, reallabel_mask)
            pool_cross_entropy = torch.sum(we)
            return pool_cross_entropy


        def getweith(input_data):
            input_data = input_data.data.cpu().numpy()
            Mask = input_data > np.mean(input_data)
            WeightMat = np.ones(Mask.shape)
            NegativeRatio = np.mean(Mask == 1)
            PostiveRatio = np.mean(Mask == 0)
            WeightMat[Mask == 0] = NegativeRatio
            WeightMat[Mask] = PostiveRatio
            WeightMat = torch.FloatTensor(WeightMat)
            WeightMat = WeightMat.cuda()
            return WeightMat


        zeros = torch.zeros([m * n]).to(device).float()


        def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                                 printFlag=True):
            if False == require_AA_KPP:
                with torch.no_grad():
                    available_label_idx = (train_samples_gt != 0).float()
                    available_label_count = available_label_idx.sum()
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                        available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count

                    return OA
            else:
                with torch.no_grad():

                    available_label_idx = (train_samples_gt != 0).float()
                    available_label_count = available_label_idx.sum()
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                        available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count
                    OA = OA.cpu().numpy()

                    zero_vector = np.zeros([class_count])
                    output_data = network_output.cpu().numpy()
                    train_samples_gt = train_samples_gt.cpu().numpy()
                    train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    for z in range(output_data.shape[0]):
                        if ~(zero_vector == output_data[z]).all():
                            idx[z] += 1
                    # idx = idx + train_samples_gt
                    count_perclass = np.zeros([class_count])
                    correct_perclass = np.zeros([class_count])
                    for x in range(len(train_samples_gt)):
                        if train_samples_gt[x] != 0:
                            count_perclass[int(train_samples_gt[x] - 1)] += 1
                            if train_samples_gt[x] == idx[x]:
                                correct_perclass[int(train_samples_gt[x] - 1)] += 1
                    test_AC_list = correct_perclass / count_perclass
                    test_AA = np.average(test_AC_list)

                    test_pre_label_list = []
                    test_real_label_list = []
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    idx = np.reshape(idx, [m, n])
                    for ii in range(m):
                        for jj in range(n):
                            if Test_GT[ii][jj] != 0:
                                test_pre_label_list.append(idx[ii][jj] + 1)
                                test_real_label_list.append(Test_GT[ii][jj])
                    test_pre_label_list = np.array(test_pre_label_list)
                    test_real_label_list = np.array(test_real_label_list)
                    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                      test_real_label_list.astype(np.int16))
                    test_kpp = kappa

                    if printFlag:
                        print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                        print('acc per class:')
                        print(test_AC_list)

                    OA_ALL.append(OA)
                    AA_ALL.append(test_AA)
                    KPP_ALL.append(test_kpp)
                    AVG_ALL.append(test_AC_list)

                    # f = open('results\\' + dataset_name + '_results.txt', 'a+')这是自动喊的第3句话
                    # 
                    # str_results = '\n======================' \
                    #               + " learning rate=" + str(learning_rate) \
                    #               + " epochs=" + str(max_epoch) \
                    #               + " train ratio=" + str(train_ratio) \
                    #               + " val ratio=" + str(val_ratio) \
                    #               + " ======================" \
                    #               + "\nOA=" + str(OA) \
                    #               + "\nAA=" + str(test_AA) \
                    #               + '\nkpp=' + str(test_kpp) \
                    #               + '\nacc per class:' + str(test_AC_list) + "\n"
                    # # + '\ntrain time:' + str(time_train_end - time_train_start) \
                    # # + '\ntest time:' + str(time_test_end - time_test_start) \
                    # f.write(str_results)
                    # f.close()
                    return OA


        # epoch = [i for i in range(1, 61)]这是自动喊的第2句话
        # 

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # ,weight_decay=0.0001
        best_loss = 99999
        net.train()
        tic1 = time.perf_counter()
        trainloss_list, valloss_list = [], []
        for i in range(max_epoch + 1):
            optimizer.zero_grad()  # zero the gradient buffers
            CNN_result, GCN_result, output = net(net_input, train=True)
            result_list = [CNN_result, GCN_result]
            # entropy_tar = -sum(
            #     sum(torch.mul(F.softmax(CNN_result, dim=1), F.log_softmax(GCN_result, dim=1)))) / \
            #               CNN_result.shape[1]
            # pdb.set_trace()
            # train_CNN, train_GCN = CNN_result * train_label_mask, GCN_result * train_label_mask
            # pdb.set_trace()
            # t = consis_loss(CNN_result, GCN_result)
            # print(softmax_mse_loss(CNN_result, GCN_result))
            # pdb.set_trace()
            loss = compute_loss(output, train_samples_gt_onehot, train_label_mask) \
                   + lamb * softmax_mse_loss(CNN_result, GCN_result) \
                   + garma * compute_loss(CNN_result, GCN_result, train_label_mask) \
                   + consis_loss(result_list) \
                   # + beta * entropy_tar
            # loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
            # pdb.set_trace()
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update
            if i % 10 == 0:
                with torch.no_grad():
                    net.eval()
                    CNN_result, GCN_result, output = net(net_input, train=True)
                    result_list = [CNN_result, GCN_result]
                    # entropy_tar = -sum(
                    #     sum(torch.mul(F.softmax(CNN_result, dim=1), F.log_softmax(GCN_result, dim=1)))) / \
                    #               CNN_result.shape[1]
                    # train_CNN, train_GCN = CNN_result * train_label_mask, GCN_result * train_label_mask
                    # print(softmax_mse_loss(CNN_result, GCN_result))
                    # l = torch.sum(torch.pow(CNN_result - GCN_result, 2) * getweith(GCN_result))
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask) \
                                + lamb * softmax_mse_loss(CNN_result, GCN_result) \
                                + garma * compute_loss(CNN_result, GCN_result, train_label_mask) \
                                + consis_loss(result_list) \
                                # + beta * entropy_tar
                    # trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainloss_list.append(trainloss)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    # val_CNN, val_GCN = CNN_result * val_label_mask, GCN_result * val_label_mask
                    # print(softmax_mse_loss(CNN_result, GCN_result))
                    l = torch.sum(torch.pow(CNN_result - GCN_result, 2) * getweith(GCN_result))
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask) \
                              # + lamb * softmax_mse_loss(CNN_result, GCN_result) \
                              # + garma * compute_loss(CNN_result, GCN_result, val_label_mask) \
                              # + consis_loss(result_list) \
                              # + beta * entropy_tar
                    # valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valloss_list.append(valloss)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print(
                        "{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                         valloss, valOA))
                    # print(
                    #     "{}\t consis_loss={}\t entropy_tar={}\t".format(str(i + 1), consis_loss(result_list), entropy_tar))
                    if valloss < best_loss:
                        best_loss = valloss
                        torch.save(net.state_dict(), "model\\best_model.pt")
                        print('save model...')
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.perf_counter()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time = toc1 - tic1 + create_graph_time  # + LDA_SLIC_Time
        Train_Time_ALL.append(training_time)
        testloss_list = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model\\best_model.pt"))
            net.eval()
            tic2 = time.perf_counter()
            CNN_result, GCN_result, output = net(net_input)
            result_list = [CNN_result, GCN_result]
            # entropy_tar = -sum(
            #     sum(torch.mul(F.softmax(CNN_result, dim=1), F.log_softmax(GCN_result, dim=1)))) / \
            #               CNN_result.shape[1]
            # test_CNN, test_GCN = CNN_result * test_label_mask, GCN_result * test_label_mask
            toc2 = time.perf_counter()
            # print(softmax_mse_loss(CNN_result, GCN_result))
            l = torch.sum(torch.pow(CNN_result - GCN_result, 2) * getweith(GCN_result))
            # testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask) \
            #            # + lamb * softmax_mse_loss(CNN_result, GCN_result) \
            #            # + garma * compute_loss(CNN_result, GCN_result, test_label_mask) \
            #            # + consis_loss(result_list) \
            #            # + beta * entropy_tar
            testloss = compute_loss(output, test_samples_gt_onehot, train_label_mask)
            testloss_list.append(testloss)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True,
                                          printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            # print(
            #     "{}\t loss={}\t entropy_tar={}".format(str(i + 1), consis_loss(result_list), entropy_tar))
            # 计算
            classification_map = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
            classification_map_ndarray = classification_map.numpy()
            Draw_Classification_Map(classification_map, "results\\" + dataset_name + str(testOA))
            testing_time = toc2 - tic2 + create_graph_time  # + LDA_SLIC_Time  # 分割耗时需要算进去
            Test_Time_ALL.append(testing_time)
            ## Saving data
            # sio.savemat(dataset_name+"softmax",{'softmax':output.reshape([height,width,-1]).cpu().numpy()})
            # np.save(dataset_name+"A_1", A_1.cpu().numpy())
            # np.save(dataset_name+"A_2", A_2.cpu().numpy())

        torch.cuda.empty_cache()
        del net
        # plt.plot(trainloss_list)
        # plt.show()
        # plt.plot(valloss_list)
        # plt.show()
        # plt.plot(testloss_list)
        # plt.show()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)
    # OA.append(OA_ALL)

    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

    # 保存数据信息
    f = open('results\\' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
                  + "\ntrain_ratio={}".format(curr_train_ratio) \
                  + "\ncurr_train_ratio={}".format(curr_train_ratio) \
                  + "\nlearning_rate={}".format(learning_rate) \
                  + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                  + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                  + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                  + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
                  + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
                  + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()

# plt.plot(lr, OA)
