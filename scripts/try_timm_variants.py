import timm

candidates = [
    'vit_giant_patch14_224',
    'eva_giant_patch14_224',
    'beit3_giant_patch14_224',
    'davit_giant',
    'vit_large_patch14_224',
    'vit_base_patch14_224',
]

for name in candidates:
    try:
        m = timm.create_model(name, pretrained=False, num_classes=0, global_pool='token')
        nf = getattr(m, 'num_features', None)
        print(name, 'num_features=', nf)
    except Exception as e:
        print(name, 'ERROR', e)
