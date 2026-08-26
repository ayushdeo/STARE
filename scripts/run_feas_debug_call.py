import traceback
print('starting feasibility debug call')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('feas', 'scripts/02_run_feasibility.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print('module loaded; calling main()')
    if hasattr(mod, 'main'):
        mod.main()
    print('main() finished')
except Exception:
    traceback.print_exc()
