import sys
import os

# Ajoute le dossier parent (~/maps/MAPS) au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
def test_imports():
    import mesh_topology
    import priority_queue
    import dk
    import obja
    import geometry_utils
    import conformal_mapping
    assert True
