import torch
from huggingface_hub import hf_hub_download

p = hf_hub_download(repo_id='MahmoodLab/UNI2-h', repo_type='model', filename='pytorch_model.bin')
print('checkpoint path:', p)
ckpt = torch.load(p, map_location='cpu')
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    sd = ckpt['state_dict']
else:
    sd = ckpt
print('state_dict keys:', len(sd))
keys = list(sd.keys())
for k in keys[:40]:
    v = sd[k]
    try:
        print(k, tuple(v.shape))
    except Exception:
        print(k, type(v))

# print a few tail keys
print('\n--- tail keys ---')
for k in keys[-40:]:
    v = sd[k]
    try:
        print(k, tuple(v.shape))
    except Exception:
        print(k, type(v))
