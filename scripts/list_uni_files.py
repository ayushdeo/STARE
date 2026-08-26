from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('MahmoodLab/UNI2-h', repo_type='model')
print('\n'.join(files))
print('COUNT', len(files))
