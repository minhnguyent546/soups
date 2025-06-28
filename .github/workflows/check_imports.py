#!/usr/bin/env python3

import pathlib
import sys
import importlib.util


def check_imports(path: str = 'src'):
    src_path = pathlib.Path(path)
    num_files = 0
    for pyfile in src_path.rglob('*.py'):
        print('Trying to import', pyfile)
        module_name = pyfile.with_suffix('').as_posix().replace('/', '.')
        spec = importlib.util.spec_from_file_location(
            name=module_name,
            location=pyfile,
        )
        if spec is None:
            print(f'Failed to load {pyfile}', file=sys.stderr)
            sys.exit(1)

        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            num_files += 1
        except Exception as e:
            print(f'Error importing {module_name}: {e}', file=sys.stderr)
            sys.exit(1)

    print(f'All {num_files} imports checked successfully.')


if __name__ == '__main__':
    check_imports('soups')
