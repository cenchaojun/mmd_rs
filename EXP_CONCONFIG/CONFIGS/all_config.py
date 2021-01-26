from EXP_CONCONFIG.CONFIGS.model_DOTA_obb_tv_config import obb_cfgs
from EXP_CONCONFIG.CONFIGS.model_DIOR_full_config import DIOR_cfgs
from EXP_CONCONFIG.CONFIGS.model_DIOR_full_voc_test_config import DIOR_voc_cfgs
from EXP_CONCONFIG.CONFIGS.model_DOTA_hbb_tv_config import cfgs

all_cfgs = {}
all_cfgs.update(obb_cfgs)
all_cfgs.update(DIOR_cfgs)
all_cfgs.update(DIOR_voc_cfgs)
all_cfgs.update(cfgs)
