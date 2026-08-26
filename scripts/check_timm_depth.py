import timm

name='vit_giant_patch14_224'
for depth in [24, 32, 40]:
    try:
        m = timm.create_model(name, pretrained=False, num_classes=0, global_pool='token', embed_dim=1536, depth=depth)
        print('depth', depth, 'num_features=', getattr(m,'num_features',None), 'blocks=', len(getattr(m,'blocks',[])))
    except Exception as e:
        print('depth', depth, 'ERROR', e)
