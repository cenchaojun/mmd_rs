import os
from EXP_CONCONFIG.CONFIGS.model_DIOR_full_config import DIOR_cfgs
from EXP_CONCONFIG.CONFIGS.model_DIOR_full_ms_test_config import DIOR_ms_test_cfgs

cfgs = DIOR_ms_test_cfgs
cfgs.update(DIOR_cfgs)

for model_name, cfg in cfgs.items():
    os.system('rm %s' % cfg['dota_eval_results'])
    print('rm %s' % cfg['dota_eval_results'])