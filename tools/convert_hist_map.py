"""One-shot converter for `hist_map.pickle`.

The original pickle (shipped with the repo) stores per-bucket histogram
probabilities as numpy float128 scalars in `collections.Counter` instances.
float128 is x86 80-bit extended precision and is not available on platforms
that lack the hardware (e.g. Apple Silicon) or on numpy 2.x, where the
legacy `'f16'` dtype string is rejected outright. As a result the pickle
cannot be loaded today on most machines.

This script decodes the 16-byte payloads by hand, rewriting each float128
scalar as a plain Python float (which numpy promotes to float64). The
result is written back to `hist_map.pickle`; nothing in `support_functions`
needs to change once the conversion has run.

Run once:
    python tools/convert_hist_map.py
"""
import io
import os
import pickle
import struct

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PICKLE_PATH = os.path.join(os.path.dirname(HERE), "hist_map.pickle")


def decode_x86_extended(raw16: bytes) -> float:
    """Decode 16 bytes of x86 80-bit extended precision into a Python float.

    Layout (little-endian, 10 bytes of value + 6 bytes of padding):
      bytes  0..7 : 64-bit significand (integer bit at MSB, then 63 fraction bits)
      bytes  8..9 : 16-bit field with 1 sign bit then 15 exponent bits
      bytes 10..15: padding
    """
    if len(raw16) != 16:
        raise ValueError(f"expected 16 bytes, got {len(raw16)}")
    significand = int.from_bytes(raw16[:8], "little")
    sign_exp = int.from_bytes(raw16[8:10], "little")
    sign = (sign_exp >> 15) & 0x1
    exponent = sign_exp & 0x7FFF
    integer_bit = (significand >> 63) & 0x1
    fraction = significand & 0x7FFFFFFFFFFFFFFF
    bias = 16383

    if exponent == 0:
        if significand == 0:
            value = 0.0
        else:  # subnormal
            value = (fraction / 2 ** 63) * 2.0 ** (1 - bias)
    elif exponent == 0x7FFF:
        if fraction == 0 and integer_bit == 1:
            value = float("inf")
        else:
            value = float("nan")
    else:
        mantissa = integer_bit + fraction / 2 ** 63
        value = mantissa * 2.0 ** (exponent - bias)

    return -value if sign else value


class _F16DtypeStub:
    """Placeholder returned in place of np.dtype('f16') during unpickling.

    Pickle calls __setstate__ on the dtype it just created; we accept and
    ignore that so the unpickler can move on.
    """

    def __setstate__(self, state):  # pragma: no cover - trivial
        pass


def _safe_dtype(*args, **kwargs):
    if args and args[0] == "f16":
        return _F16DtypeStub()
    return np.dtype(*args, **kwargs)


def _safe_scalar(dtype, raw_bytes):
    """Replacement for `numpy.core.multiarray.scalar`."""
    if isinstance(dtype, _F16DtypeStub):
        return decode_x86_extended(raw_bytes)
    return np.core.multiarray.scalar(dtype, raw_bytes)


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy" and name == "dtype":
            return _safe_dtype
        if module == "numpy.core.multiarray" and name == "scalar":
            return _safe_scalar
        return super().find_class(module, name)


def main():
    # The unpickled objects need their original classes; make sure the
    # sequencing.calling package is importable.
    import sys
    sys.path.insert(0, os.path.dirname(HERE))

    with open(PICKLE_PATH, "rb") as f:
        hist_map = CompatUnpickler(f).load()

    # Sanity check: every histogram should now contain plain Python floats.
    sample = hist_map[0]._hist
    if sample:
        any_value = next(iter(sample.values()))
        assert isinstance(any_value, float), f"unexpected value type {type(any_value)}"

    # Re-save using a standard pickle (everything is now float64-compatible).
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(hist_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Converted {len(hist_map)} histograms; "
        f"sample value type now {type(any_value).__name__}."
    )


if __name__ == "__main__":
    main()
