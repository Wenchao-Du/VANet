from easydict import EasyDict as edict

__C = edict()

config = __C

# train part options

__C.TRAIN = edict()
__C.TRAIN.istrain = True
__C.TRAIN.epochs = 20
__C.TRAIN.g_lr = 0.0005
__C.TRAIN.d_lr = 0.0005
__C.TRAIN.batch_size = 32
__C.TRAIN.log_step = 10
__C.TRAIN.sample_step = 5000
__C.TRAIN.sample_size = 64
__C.TRAIN.sample_path = ''
__C.TRAIN.model_path = ''
__C.TRAIN.root = ''
__C.TRAIN.beta1 = 0.5
__C.TRAIN.beta2 = 0.9
__C.TRAIN.weights = None
__C.TRAIN.test_step = 200
__C.TRAIN.test_batchsize = 4

__C.TEST = edict
