"""CPU-only ABI preflight for the compiled source adapter inside the ROCm image."""

import argparse
import ctypes


def check(path: str) -> None:
    library = ctypes.CDLL(path)
    validate = library.aiter_rs_opus_a16w16_validate
    validate.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p]
    validate.restype = ctypes.c_int
    for args in ((8192, 4096, 256, 1300), (12287, 4096, 256, 300), (16384, 2192, 512, 300)):
        if validate(*args, None) != 0:
            raise AssertionError(f"valid null/default HIP stream rejected for {args}")
    if validate(16384, 2177, 512, 300, None) != 1:
        raise AssertionError("unaligned N was not rejected")
    if validate(8192, 4096, 256, 7, None) != 1:
        raise AssertionError("unknown kid was not rejected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    args = parser.parse_args()
    check(args.library)
    print("adapter ABI preflight passed (default stream 0 and shape/kid guards)")


if __name__ == "__main__":
    main()
