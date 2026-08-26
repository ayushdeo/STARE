import h5py
import numpy as np
from stare import data
from stare.encoders import embed_patches, _get_model

sample_id = 'TENX174'
p = data._sample_hf_path(sample_id, 'patches/{sample_id}.h5')
with h5py.File(p, 'r') as f:
    imgs = f['img'][:8]

print('loaded imgs', imgs.shape, imgs.dtype)
# ensure model instantiated
m = _get_model()
print('model:', type(m))
emb = embed_patches(imgs, batch_size=4)
print('emb shape', emb.shape)
print('emb std', emb.std())
print('emb min/max', emb.min(), emb.max())

assert emb.shape == (8, 1536), f'emb shape mismatch: {emb.shape}'
assert emb.std() > 0, 'embeddings appear degenerate (std==0)'
print('sanity check passed')
