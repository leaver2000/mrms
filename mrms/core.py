from __future__ import annotations

import datetime
import gzip
import os
import re
import tempfile
from http import HTTPStatus
from typing import TYPE_CHECKING, Final

import numpy as np
import pygrib._pygrib
import requests
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from mrms import Vars

type N = int
type X = int
type Y = int
type Nd[*Ts] = tuple[*Ts]


__x = np.load(os.path.join(os.path.dirname(__file__), "MRMS_latlon.npz"), allow_pickle=False)
latitude: Final[Array[Nd[Y, X], np.float_]] = __x["latitude"]
longitude: Final[Array[Nd[Y, X], np.float_]] = __x["longitude"]
proj = {"a": 6378160.0, "b": 6356775.0, "proj": "longlat"}
del __x


_datetime_format = "%Y%m%d-%H%M%S"
_datetime_pattern = re.compile(r"(\d{8}-\d{6})")
_filename_pattern = re.compile(r"<a href=\"(.+\d{8}-\d{6}.grib2.gz)\">")


def _search_time(s: str) -> datetime.datetime:
    assert (m := _datetime_pattern.search(s)), f"Failed to parse datetime from {s}"
    return datetime.datetime.strptime(m.group(0), _datetime_format)


def file_directory(
    name: Vars = "VIL",
) -> tuple[np.ndarray[Nd[N], np.dtype[np.datetime64]], np.ndarray[Nd[N], np.dtype[np.str_]]]:
    """
    Retrieves the file directory for the specified name from the MRMS server.

    Parameters:
    ----------
    name : Vars, optional
        The name of the file directory to retrieve. Defaults to "VIL".

    Returns:
    --------
    array : datetime64 array containing the corresponding datetime values.
    array : str array containing the corresponding file names.

    Examples:
    ---------
    >>> fd = mrms.file_directory(mrms.VIL)
    >>> s = pd.Series(dict(zip(*fd)), dtype='string')
    >>> s
    2024-05-08 02:00:39    MRMS_VIL_00.50_20240508-020039.grib2.gz
    2024-05-08 02:02:39    MRMS_VIL_00.50_20240508-020239.grib2.gz
    2024-05-08 02:04:40    MRMS_VIL_00.50_20240508-020440.grib2.gz
    2024-05-08 02:06:40    MRMS_VIL_00.50_20240508-020640.grib2.gz
    2024-05-08 02:08:38    MRMS_VIL_00.50_20240508-020838.grib2.gz
                                            ...
    2024-05-09 03:38:41    MRMS_VIL_00.50_20240509-033841.grib2.gz
    2024-05-09 03:40:38    MRMS_VIL_00.50_20240509-034038.grib2.gz
    2024-05-09 03:42:37    MRMS_VIL_00.50_20240509-034237.grib2.gz
    2024-05-09 03:44:40    MRMS_VIL_00.50_20240509-034440.grib2.gz
    2024-05-09 03:46:41    MRMS_VIL_00.50_20240509-034641.grib2.gz
    Length: 774, dtype: string
    """

    if (r := requests.get(f"https://mrms.ncep.noaa.gov/data/2D/{name}")).status_code != HTTPStatus.OK:
        code = HTTPStatus(r.status_code)
        raise requests.HTTPError(f"Failed to access MRMS server: {code.value} {code.phrase}")

    fn = _filename_pattern.findall(r.text)
    dt = [_search_time(x) for x in fn]
    return np.array(dt, dtype=np.datetime64), np.array(fn, dtype=np.str_)


# -------------------------------------------------------------------------------------------------
class Array[Nd: Nd, T: np.generic](np.ndarray[Nd, np.dtype[T]]):  # type: ignore
    datetime: datetime.datetime
    name: Vars | None

    def __new__(
        cls,
        values: ArrayLike,
        *,
        dtype: type[T] | None = None,
        datetime: datetime.datetime,
        name: Vars | None = None,
    ) -> Array[Nd, T]:
        x = np.asarray(values, dtype=dtype).view(cls)
        x.name = name
        x.datetime = datetime
        return x


def load[T: np.float_](f: str, name: Vars | None = None, *, dtype: type[T] = np.float64) -> Array[Nd[Y, X], T]:
    with pygrib._pygrib.open(f) as grbs:
        grb = grbs[1]
        x = Array(grb.values, datetime=grb.validDate, name=name, dtype=dtype)

    return x


def get[
    T: np.float_
](
    name: Vars = "VIL", *, file: str = "latest", datetime: datetime.datetime | None = None, dtype: type[T] = np.float64
) -> Array[Nd[Y, X], T]:
    if file == "latest":
        dt, files = file_directory(name)
        if datetime is None:  # latest
            idx = np.argmax(dt)
        else:
            idx = np.argmin(np.abs(dt - np.datetime64(datetime)))

        file = files[idx]

    assert file.endswith(".grib2.gz"), f"Invalid file extension: {file}"

    url = f"https://mrms.ncep.noaa.gov/data/2D/{name}/{file}"

    with requests.get(url, stream=True) as r:
        if r.status_code != HTTPStatus.OK:
            code = HTTPStatus(r.status_code)
            raise requests.HTTPError(f"Failed to access MRMS server: {code.value} {code.phrase}")

        with tempfile.NamedTemporaryFile("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                f.flush()

            with gzip.open(f.name, "rb") as f:
                with tempfile.NamedTemporaryFile("wb") as tmp:
                    tmp.write(f.read())
                    tmp.flush()

                    return load(tmp.name, name, dtype=dtype)
