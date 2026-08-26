import sys
import runpy
import traceback

print('python exe:', sys.executable)
print('python version:', sys.version.replace('\n', ' '))

try:
    print('running 02_run_feasibility.py')
    runpy.run_path('scripts/02_run_feasibility.py', run_name='__main__')
    print('finished run')
except Exception:
    print('EXCEPTION during feasibility run:')
    traceback.print_exc()
