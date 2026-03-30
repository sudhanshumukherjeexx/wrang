"""Smoke tests — package import and version sanity."""

import wrang


def test_import():
    """Package imports without error."""
    assert hasattr(ride, "__version__")


def test_version_format():
    """Version is a non-empty string."""
    v = ride.__version__
    assert isinstance(v, str)
    assert len(v) > 0


def test_lazy_import_no_side_effects():
    """Accessing an unknown attribute raises AttributeError (not ImportError)."""
    import pytest
    with pytest.raises(AttributeError):
        _ = ride._totally_undefined_name_xyz
