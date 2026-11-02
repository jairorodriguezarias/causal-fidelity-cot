"""The portable tutorial must contain the current, inspectable source snapshot."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path

from scripts.build_colab_notebook import BUNDLE_FILES


def test_notebook_embedded_sources_match_repository_and_cells_compile():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/causal_fidelity_colab.ipynb').read_text())
    constants = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        compile(source, cell['id'], 'exec')
        assert cell['execution_count'] is None
        assert cell['outputs'] == []
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
    packed = base64.b64decode(constants['BUNDLE_B64'], validate=True)
    assert hashlib.sha256(packed).hexdigest() == constants['BUNDLE_SHA256']
    embedded = json.loads(gzip.decompress(packed))
    assert set(embedded) == set(BUNDLE_FILES)
    for name, content in embedded.items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert content == (root / relative).read_text(), f'Rebuild notebook: {name} differs'
