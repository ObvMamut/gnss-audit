"""IQ sources (docs/8 section 6, step 1): the SpoofRepo L5 files store Q before I (P2b task 6: GNSS-SDR tracks real L5
only with the halves swapped), so a source can declare it and is read as I + jQ either way."""

from pathlib import Path

import numpy as np
import pytest

from gnssaudit.inject.source import IQSource, read_iq


def test_a_source_storing_q_first_is_read_as_i_plus_jq(tmp_path: Path) -> None:
    p = tmp_path / "s.bin"
    np.array([1, -3, 5, 7, -1, 3], dtype=np.int8).tofile(p)  # stored pairs (Q, I)
    plain = IQSource(path=p, item_type="ibyte", fs_hz=1.0, sample_start=0, sample_end=3)
    swapped = IQSource(path=p, item_type="ibyte", fs_hz=1.0, sample_start=0, sample_end=3, q_first=True)
    np.testing.assert_array_equal(read_iq(plain, 0, 3), [1 - 3j, 5 + 7j, -1 + 3j])
    np.testing.assert_array_equal(read_iq(swapped, 0, 3), [-3 + 1j, 7 + 5j, 3 - 1j])


def test_a_read_outside_the_source_range_is_refused(tmp_path: Path) -> None:
    """Final review R2, I1: a read that starts inside [sample_start, sample_end) and runs past its end is refused (on
    FGI segment B the bytes after sample_end hold the C/N0 step and the spoofer), as are a read that starts before the
    range and a negative count. A read that ends at sample_end is the range's last."""
    p = tmp_path / "s.bin"
    np.arange(200, dtype=np.int8).tofile(p)  # 100 samples
    src = IQSource(path=p, item_type="ibyte", fs_hz=1.0, sample_start=10, sample_end=50)
    assert read_iq(src, 40, 10).size == 10 and read_iq(src, 10, 40).size == 40
    for start, n in ((40, 30), (49, 2), (5, 10), (5, 50), (20, -1)):
        with pytest.raises(ValueError, match="outside the source range"):
            read_iq(src, start, n)
