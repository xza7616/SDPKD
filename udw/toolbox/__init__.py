from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, adjust_lr
from .ranger.ranger import Ranger
from .ranger.ranger913A import RangerVA
from .ranger.rangerqh import RangerQH

def get_dataset(cfg):
    assert cfg['dataset'] in ['SUIM','WE3Ds']

    if cfg['dataset'] == 'SUIM':
        from .datasets.suim import SUIM
        return SUIM(cfg, mode='train'), SUIM(cfg, mode='test')
    if cfg['dataset'] == 'WE3Ds':
        from .datasets.WE3Ds import WE3Ds
        return WE3Ds(cfg, mode='train'), WE3Ds(cfg, mode='test')


def get_model(cfg):
    if cfg['model_name'] == 'SPRNet':
        from SPRNet import EnDecoderModel
        return EnDecoderModel()
    if cfg['model_name'] == 'kd':
        from PANet import EncoderDecoder
        return EncoderDecoder()


def get_teacher_model(cfg):
    if cfg['model_name'] == 'kd':
        from SPRNet import EnDecoderModel
        return EnDecoderModel()




