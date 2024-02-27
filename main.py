from __future__ import print_function
import time
from args_fusion import args
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from net_stage1 import Stage1
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.
img_names = ['01','02','03','04','05','06','07','08','09','10','11','12']
times = []
psrn_maxs = []
epoch_list = []
for i in range(1):
    for img_name in img_names:

        fname = 'data/Set12_sig25_stage1/{}.png'.format(img_name)

        print("the image is set12_%s" % img_name)
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)


        img_np_torch = np_to_torch(img_np).type(dtype)

        fname1 = 'data/Set12/{}.png'.format(img_name)

        img_pil1 = crop_image(get_image(fname1, imsize)[0], d=32)
        img_np1 = pil_to_np(img_pil1)

        img_noisy_pil, img_noisy_np = get_noisy_image(img_np1, sigma_)
        img_set12_np = img_np1.astype(np.float32)
        img_set12_pil = np_to_pil(img_set12_np)
        img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
        img_noisy_pil= np_to_pil(img_noisy_np)

        if PLOT:

            plot_image_grid([img_np, img_noisy_np], 4, 6);

        workplace = 'result_paper/set12_sig25/{}'.format(img_name)
        if not os.path.exists(workplace):
            os.makedirs(workplace)

        img_noisy_pil.save(os.path.join(workplace, "noisy.bmp"))
        gpu = 0
        batch_size = args.batch_size
        ################## Using GPU when it is available ##################
        device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

        print(torch.cuda.is_available())
        print('device info:')
        print(device)
        print('#_______________________')

        INPUT = 'noise'  # 'meshgrid'
        pad = 'reflection'
        OPT_OVER = 'net'  # 'net,input'

        reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
        LR = 0.005

        OPTIMIZER = 'adam'  # 'LBFGS'
        show_every = 100
        exp_weight = 0.99


        num_iter = 6000
        input_depth = 1
        figsize = 4

        net = Stage1()
        net = net.cuda()
        net_input = img_np_torch

        s = sum([np.prod(list(p.size())) for p in net.parameters()]);
        print('Number of params: %d' % s)

        mse = torch.nn.MSELoss().type(dtype)

        img_set12_torch = np_to_torch(img_set12_np).type(dtype)

        file = open(os.path.join(workplace, "logs.txt"), 'w')
        file.truncate(0)

        net_input_saved = net_input.detach().clone()
        orignal = net_input.detach().clone()
        out_avg = None
        last_net = None
        psrn_noisy_last = 0
        psrn_max = 0
        epoch_max = 0

        i = 0

        strat = time.time()
        def closure():
            global i, out_avg, psrn_noisy_last, last_net, net_input, psrn_max,  epoch_max, total_loss ,psrn_gt_sm

            if reg_noise_std > 0:
                net_input = net_input_saved

            out = net(net_input)

            if out_avg is None:

                out_avg = out.detach()
            else:
                out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

            total_loss = mse(out, img_np_torch) + mse(out, img_noisy_torch) * 0.4
            total_loss.requires_grad_(True)
            total_loss.backward()

            psrn_gt = compare_psnr(img_set12_np, out.detach().cpu().numpy()[0])
            psrn_gt_sm = compare_psnr(img_set12_np, out_avg.detach().cpu().numpy()[0])

            if psrn_gt > psrn_max:
                psrn_max = psrn_gt_sm
                epoch_max = i + 1
                pil_max = np_to_pil(out.detach().cpu().numpy()[0])
                pil_max.save(os.path.join(workplace, "best_channel0.bmp"))
            #print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f  Time %.2f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm,_t['im_detect'].total_time), '\r')

            if PLOT and i % show_every == 0:
                print('Iteration %05d    Loss %f    PSRN_gt: %f PSNR_gt_sm: %f ' % (
                i, total_loss.item(), psrn_gt, psrn_gt_sm), '\r')
                out_np = torch_to_np(out)

            i += 1
            return total_loss

        p = get_params(OPT_OVER, net, net_input)
        optimize(OPTIMIZER, p, closure, LR, num_iter)
        end = time.time()
        times.append(end - strat)

        psrn_maxs.append(psrn_max)
        epoch_list.append(epoch_max)

        file.write('Iteration_max %05d    Loss %f   psrn_max: %f  psrn_sm:%f\n' % (
            epoch_max, total_loss.item(), psrn_max, psrn_gt_sm))
        print('psrn_maxs: ', psrn_maxs, "   avg_max: ", np.mean(np.array(psrn_maxs)))
        print('epoch_list: ', epoch_list, "   avg_max: ", np.mean(np.array(psrn_maxs)))
    print("times_avg: %f" % np.mean(np.array(times)))
    print(times)
    file.write("times_avg: " % (np.mean(np.array(times))))




