from __future__ import annotations

import unittest.mock

import numpy as np
import pytest

import mrms


def grib_file() -> bytes:
    with open("data/MRMS_VIL_00.50_20240509-082840.grib2.gz", "rb") as f:
        return f.read()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@unittest.mock.patch(
    "requests.get",
    return_value=unittest.mock.MagicMock(
        __enter__=lambda *_: unittest.mock.MagicMock(
            status_code=200, iter_content=lambda chunk_size: iter([grib_file()])
        )
    ),
)
def test_get(mock, dtype):
    arr = mrms.get(file="MRMS_VIL_00.50_20240509-082840.grib2.gz", dtype=dtype)
    assert isinstance(arr, mrms.Array) and isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape == mrms.longitude.shape == mrms.latitude.shape
    assert arr.dtype == dtype
