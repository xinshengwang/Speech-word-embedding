from easydict import EasyDict as edict

__C = edict()
cfg = __C


__C.manualSeed = 100
__C.CUDA = True
__C.workers = 0

__C.WBDNet = edict()
__C.WBDNet.start_epoch = 0
__C.WBDNet.batch_size = 64
__C.WBDNet.input_size = 80
__C.WBDNet.hidden_size = 20
__C.WBDNet.num_layers = 2


__C.WD = edict()
__C.WD.imgsize = 256
__C.WD.input_size = 80
__C.WD.hidden_size = 256

__C.WD.smooth1 = 5
__C.WD.smooth2 = 5
__C.WD.smooth3 = 5