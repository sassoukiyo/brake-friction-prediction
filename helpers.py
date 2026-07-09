"""
helpers.py  --  student-facing utilities (handed out with each assignment)
==========================================================================
Provides everything students need to load data, save/load their model,
inspect model complexity, and read the PDF submission sheet.

Do not add grading logic here. This file must stay lean and task-agnostic
so it can be reused without modification in future terms.
"""

import os
import pickle

import numpy as np


# -- Data ---------------------------------------------------------------------

def load_dataset(filepath):
    """Load a CSV dataset (with header row) and return (X, y).

    Convention: all columns except the last are features (X);
    the last column is the regression target (y).

    Returns:
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data[:, :-1], data[:, -1]


# -- Model --------------------------------------------------------------------

def save_model(model, filepath):
    """Serialize a fitted model to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath):
    """Load a model from a pickle file and validate it has predict().

    Raises:
        FileNotFoundError: if the file does not exist.
        TypeError:         if the loaded object lacks a callable predict().
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "predict") or not callable(model.predict):
        raise TypeError(
            f"Loaded object has no callable predict(): {type(model)}"
        )
    return model


# -- PDF ----------------------------------------------------------------------

def read_pdf_fields(filepath, fields):
    """Extract and type-cast AcroForm fields from a filled submission PDF.

    Args:
        filepath : path to the PDF.
        fields   : dict mapping AcroForm field name -> Python type,
                   e.g. {"mat_num": int, "expected_r2": float}.

    Returns:
        dict with the same keys; values cast to the requested type,
        or None when a field is absent or cannot be cast.

    Raises:
        FileNotFoundError: if the PDF does not exist.
        ImportError:       if pypdf is not installed (pip install pypdf).
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    raw = PdfReader(filepath).get_fields() or {}
    return {
        key: _cast(raw[key].get("/V") if key in raw else None, dtype)
        for key, dtype in fields.items()
    }


def _cast(raw, dtype):
    """Cast a raw PDF string to dtype; return None on failure."""
    if raw is None:
        return None
    try:
        s = str(raw).strip()
        if dtype is float:
            s = s.replace(",", ".")   # accept comma as decimal separator
        return dtype(s)
    except (ValueError, TypeError):
        return None
