import os
exp_name = 'exp-MNET_V2_224_BASE_LATEST'
for epoch in range(1,25):
    os.system(f'python3 ./mAP.py {epoch} 224 {exp_name} pred')
for epoch in range(1,25):
    os.system(f'python3 ./mAP.py {epoch} 224 {exp_name} calc')

exp_name = 'exp-MNET_V2_224_MUL_SIG_5110'
for epoch in range(1,25):
    os.system(f'python3 ./mAP.py {epoch} 224 {exp_name} pred')
for epoch in range(1,25):
    os.system(f'python3 ./mAP.py {epoch} 224 {exp_name} calc')
    
#exp_name = 'exp-MNET_V2_224_BASE_LATEST'
#for epoch in range(10,25):
#    os.system(f'python3 ./mAP.py {epoch} 224 {exp_name} pred')
#for epoch in range(10,25):
#    os.system(f'python3 ./mAP.py {epoch} 224 {exp_name} calc')
