import sys
import torch
from collections import OrderedDict

# alpha = float(sys.argv[1])
alphalist = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
alpha = alphalist[5]
net_PSNR_path = 'K:\\DeNoisedrop\\weights\\AG\\AG_MSE_2586087.pkl'
net_EnGAN_path = 'K:\\DeNoisedrop\\weights\\AG\\AG_FT1_721136-1.pkl'
net_interp_path = 'K:\\DeNoisedrop\\weights\\AG\\new_interp_{:02d}.pkl'.format(
    int(alpha * 10))

net_PSNR = torch.load(net_PSNR_path)
net_ENGAN = torch.load(net_EnGAN_path)
net_interp = OrderedDict()

print('Interpolating with alpha = ', alpha)

for k, v_PSNR in net_PSNR.items():
    v_ESRGAN = net_ENGAN[k]
    net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

torch.save(net_interp, net_interp_path)