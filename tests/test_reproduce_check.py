"""Checks on the comparisons used by the reproduction workflow."""
import io

import numpy as np
import pandas as pd
from PIL import Image

from reproduce_check import compare_images, compare_page, compare_tables


def png(pixels: np.ndarray, **save_options) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buf, format="PNG", **save_options)
    return buf.getvalue()


def test_images_with_the_same_pixels_but_other_bytes_count_as_same():
    pixels = np.random.default_rng(0).integers(0, 255, size=(40, 60, 3))
    a, b = png(pixels, compress_level=1), png(pixels, compress_level=9)
    assert a != b
    assert compare_images(a, b)[0] == "same pixels"


def test_image_changes_are_graded_by_share_of_pixels():
    base = np.full((100, 100, 3), 255)
    small = base.copy()
    small[0, :10] = 0                      # 0.1% of pixels
    large = base.copy()
    large[:20] = 0                         # 20% of pixels
    assert compare_images(png(small), png(base))[0] == "pixels differ"
    assert compare_images(png(large), png(base))[0] == "differs"
    assert compare_images(png(base[:50]), png(base))[0] == "differs"   # other size
    assert compare_images(b"not an image", png(base))[0] == "differs"


def test_pages_must_have_equal_text_and_close_numbers():
    old = "<p>MASE 1.010 over 77 origins</p>"
    assert compare_page("<p>MASE 1.011 over 77 origins</p>", old)[0] == "within tolerance"
    assert compare_page("<p>MASE 1.050 over 77 origins</p>", old)[0] == "differs"
    assert compare_page("<p>MAE 1.010 over 77 origins</p>", old)[0] == "differs"


def test_tables_allow_one_unit_in_the_last_decimal_only():
    old = pd.DataFrame({"model": ["ETS", "ARIMA"], "MASE": [1.010, 1.015]})
    near = old.assign(MASE=[1.011, 1.015])
    far = old.assign(MASE=[1.030, 1.015])
    assert compare_tables(near, old, rounded=True)[0] == "within tolerance"
    assert compare_tables(far, old, rounded=True)[0] == "differs"
    assert compare_tables(old.iloc[:1], old, rounded=True)[0] == "differs"
