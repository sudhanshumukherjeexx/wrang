#!/usr/bin/env python3
"""
RIDE Core Module
Data processing and analysis core functionality
"""

# Import all core classes for easy access
try:
    from .loader import FastDataLoader, DataSaver, load_data, save_data
    from .inspector import DataInspector, inspect_data
    from .explorer import DataExplorer, explore_data
    from .cleaner import DataCleaner, BatchCleaner, clean_data, quick_clean
    from .transformer import DataTransformer, TransformationPipeline, transform_data, create_pipeline, quick_transform
    
    __all__ = [
        # Loader classes and functions
        'FastDataLoader', 'DataSaver', 'load_data', 'save_data',
        
        # Inspector classes and functions
        'DataInspector', 'inspect_data',
        
        # Explorer classes and functions  
        'DataExplorer', 'explore_data',
        
        # Cleaner classes and functions
        'DataCleaner', 'BatchCleaner', 'clean_data', 'quick_clean',
        
        # Transformer classes and functions
        'DataTransformer', 'TransformationPipeline', 'transform_data', 'create_pipeline', 'quick_transform'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import some core modules: {e}")
    __all__ = []