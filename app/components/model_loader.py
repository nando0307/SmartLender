"""Load saved models from disk."""
import os
import glob
import joblib


def load_module_models(module_name: str, models_dir: str = None) -> dict:
    """
    Load all saved .joblib models for a given module.

    Args:
        module_name: e.g. 'module_a'
        models_dir: Base models directory. Defaults to project models/ dir.

    Returns:
        Dict of {algorithm_name: model_or_pipeline}
    """
    if models_dir is None:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'models'
        )

    module_dir = os.path.join(models_dir, module_name)
    if not os.path.exists(module_dir):
        return {}

    models = {}
    for filepath in sorted(glob.glob(os.path.join(module_dir, '*.joblib'))):
        name = os.path.basename(filepath).replace('.joblib', '').replace('_', ' ').title()
        models[name] = joblib.load(filepath)

    return models


def get_available_modules(models_dir: str = None) -> list:
    """Return list of module names that have saved models."""
    if models_dir is None:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'models'
        )

    if not os.path.exists(models_dir):
        return []

    modules = []
    for name in sorted(os.listdir(models_dir)):
        module_path = os.path.join(models_dir, name)
        if os.path.isdir(module_path) and glob.glob(os.path.join(module_path, '*.joblib')):
            modules.append(name)
    return modules
