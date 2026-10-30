"""Import aliases for checkpoints saved before the packages were renamed.

Refiner and Stage 1 checkpoints are loaded with ``torch.load(weights_only=False)``
and may contain objects pickled under the original development package names.
This finder maps those names to the renamed modules, so that existing
checkpoints load without being re-saved.
"""

import importlib
import importlib.abc
import importlib.util
import sys

_PACKAGES = {
    "agh_former_v2": "facial_landmarks.pipeline",
    "agh_former_vnext_orthodontic_comparison": "facial_landmarks.stage2",
    "all23_mvsc_refinement": "facial_landmarks.refiner",
    "all23_rgb_geodesic_cascade": "facial_landmarks.core",
    "core20_mvsc_refinement": "facial_landmarks.refiner_base",
    "hard3_anatomical_context_refinement": "facial_landmarks.optional.context_refiner",
    "curve_supervised_hard3_refinement": "facial_landmarks.optional.curve_refiner",
}
_MODULES = {
    "agh_former_v2.run_agh_v2": "facial_landmarks.pipeline.run_cv",
    "agh_former_v2.colab_run_agh_v2": "facial_landmarks.pipeline.colab_run",
    "agh_former_vnext_orthodontic_comparison.run_aghformer_vnext": "facial_landmarks.stage2.runner",
    "agh_former_vnext_orthodontic_comparison.colab_run_aghformer_vnext": "facial_landmarks.stage2.colab",
    "agh_former_vnext_orthodontic_comparison.hard3_structured": "facial_landmarks.stage2.hairline_mandible",
    "all23_rgb_geodesic_cascade.run_all23_rgb_geodesic": "facial_landmarks.core.runner",
}


def _target(name):
    if name in _MODULES:
        return _MODULES[name]
    head, _, tail = name.partition(".")
    if head in _PACKAGES:
        return _PACKAGES[head] + ("." + tail if tail else "")
    return None


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        new = _target(name)
        if new is None:
            return None
        return importlib.util.spec_from_loader(name, _AliasLoader(new))


def install():
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        # First, so that submodules of an aliased package are not loaded a second time
        # by the path finder under their old name.
        sys.meta_path.insert(0, _AliasFinder())
