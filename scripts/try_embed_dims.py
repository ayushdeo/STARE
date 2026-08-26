import timm

name='vit_giant_patch14_224'
for kw in [{'embed_dim':1536}, {'num_features':1536}, {'features':1536}, {'embed_dim':1536,'num_heads':48}]:
    try:
        m = timm.create_model(name, pretrained=False, num_classes=0, global_pool='token', **kw)
        print('kw',kw,'num_features',getattr(m,'num_features',None))
    except Exception as e:
        print('kw',kw,'ERROR',e)
