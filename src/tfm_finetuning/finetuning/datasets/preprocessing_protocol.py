"""The two preprocessing protocols evaluated in the thesis."""

PREPROCESSING_NATIVE = "native"
PREPROCESSING_SHARED = "shared"
SUPPORTED_PREPROCESSING_PROTOCOLS = {PREPROCESSING_NATIVE, PREPROCESSING_SHARED}


def normalize_preprocessing_protocol(value: object) -> str:
    """Normalize spelling; callers validate membership in the supported protocols."""
    return str(value or "").strip().lower()
