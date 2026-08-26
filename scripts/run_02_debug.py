import traceback

try:
    from scripts import _run
except Exception:
    pass

try:
    import importlib, runpy
    print('running 02_run_feasibility.py')
    runpy.run_path('scripts/02_run_feasibility.py', run_name='__main__')
except Exception as e:
    print('EXC:')
    traceback.print_exc()
