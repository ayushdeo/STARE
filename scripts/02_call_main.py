import traceback
print('calling main() from 02_run_feasibility')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('feas', 'scripts/02_run_feasibility.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, 'main'):
        mod.main()
    else:
        print('no main() found')
except Exception:
    traceback.print_exc()
