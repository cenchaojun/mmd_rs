import os
os.chdir('../../')
dota_val_root = './data/dota_1_1024_824/val'
print(len(os.listdir(dota_val_root + '/images')))
print(len(os.listdir(dota_val_root + '/labelTxt')))
print(os.path.exists(dota_val_root + '/images/P0246__1__0___0.png'))
print(os.path.exists(dota_val_root + '/labelTxt/P0422__1__0___0.txt'))