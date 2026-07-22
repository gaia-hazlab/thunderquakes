import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thunderquakes.plotting import LETTER_WIDTH_IN, MIN_FONT, letter_figsize, set_paper_style


def test_min_font_floor():
    assert MIN_FONT >= 10


def test_letter_width_fits_page_with_margins():
    assert LETTER_WIDTH_IN <= 6.5


def test_letter_figsize_respects_width():
    w, h = letter_figsize(aspect=0.5)
    assert w == LETTER_WIDTH_IN
    assert h == LETTER_WIDTH_IN * 0.5


def test_set_paper_style_sets_readable_rcparams():
    set_paper_style()
    assert plt.rcParams["font.size"] >= MIN_FONT
    assert plt.rcParams["xtick.labelsize"] >= MIN_FONT
    assert plt.rcParams["ytick.labelsize"] >= MIN_FONT
    assert plt.rcParams["legend.fontsize"] >= MIN_FONT
