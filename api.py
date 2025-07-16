import pkgutil
import importlib
import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import scipy
except ImportError:
    print("Error: The 'numpy' package is not installed. Please install it with 'pip install numpy'.")
    exit(1)

def get_public_api_with_docstrings(package):
    public_api = []
    package_name = package.__name__
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package_name + '.'):
        # Skip internal modules, f2py, and tests
        if (modname.startswith(package_name + '._core') or
            'f2py' in modname or
            'tests' in modname):
            continue
        if ispkg:
            continue
        try:
            module = importlib.import_module(modname)
        except Exception as e:
            print(f"Warning: Could not import {modname}: {e}")
            continue
        if hasattr(module, '__all__'):
            public_names = module.__all__
        else:
            public_names = [name for name in dir(module) if not name.startswith('_')]
        for name in public_names:
            try:
                obj = getattr(module, name)
            except AttributeError as e:
                print(f"Warning: Module {modname} has no attribute {name}: {e}")
                continue
            docstring = getattr(obj, '__doc__', None)
            if docstring:
                docstring = docstring.strip().split('\n')
            public_api.append( (modname, name, docstring) )
    return public_api

api = get_public_api_with_docstrings(scipy)
for modname, name, docstring in sorted(api, key=lambda x: f"{x[0]}.{x[1]}"):
    qualified_name = f"{modname}.{name}"
    if docstring:
        print(f"# {docstring}")
    print(qualified_name)
    print("")
