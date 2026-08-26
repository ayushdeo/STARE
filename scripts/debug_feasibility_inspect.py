import traceback
print('PYTHON:', __import__('sys').executable)
try:
    from stare import config, data
    print('FEASIBILITY_TISSUE', config.FEASIBILITY_TISSUE)
    slides = data.load_tissue(config.FEASIBILITY_TISSUE)
    print('loaded slides count:', len(slides))
    if slides:
        s = slides[0]
        print('first slide regions:', len(s.regions))
        print('first slide genes:', len(s.genes))
except Exception:
    traceback.print_exc()
