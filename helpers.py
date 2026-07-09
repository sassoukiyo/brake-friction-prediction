

import os
import pickle

import numpy as np



def load_dataset(filepath):
    
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data[:, :-1], data[:, -1]



def save_model(model, filepath):
    """Serialize a fitted model to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath):
    
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "predict") or not callable(model.predict):
        raise TypeError(
            f"Loaded object has no callable predict(): {type(model)}"
        )
    return model



def read_pdf_fields(filepath, fields):
   
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
