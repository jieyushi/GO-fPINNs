"""Shared explicit font setup. Change sizes in FONT_SIZES; no silent substitution."""
from pathlib import Path

FONT_FAMILY = 'Times New Roman'

MATH_FONTSET = 'stix'

FONT_SIZES = {
    'base': 22,
    'axis_label': 24,
    'axis_tick': 22,
    'column_title': 24,
    'row_label': 24,
    'colorbar_label': 22,
    'colorbar_tick': 22,
    'verification': 14,
}

def configure_typography(plt, font_manager, font_dir=None, font_family=None):
    """Use local Times files, including the parent project's fonts folder."""
    global FONT_FAMILY
    if font_family is not None:FONT_FAMILY=font_family
    package = Path(__file__).resolve().parent
    if font_dir is not None:
        directory = Path(font_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError('Font directory does not exist: ' + str(directory))
        directories = [directory]
    else:
        directories = list(dict.fromkeys([
            package / 'fonts',
            package.parent / 'fonts',
            Path.cwd() / 'fonts',
            Path.home() / '.local' / 'share' / 'fonts',
            Path.home() / '.fonts',
        ]))
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob('*')):
            if path.suffix.lower() not in {'.ttf', '.otf'}:
                continue
            properties = font_manager.FontProperties(fname=str(path))
            if properties.get_name().casefold() == FONT_FAMILY.casefold():
                font_manager.fontManager.addfont(str(path))
    try:
        font_path = font_manager.findfont(
            font_manager.FontProperties(family=FONT_FAMILY),
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise RuntimeError(
            FONT_FAMILY+' is unavailable. Install it or put its .ttf/.otf files '
            'in fonts/ and use --font-dir. No substitute font was used.'
        ) from exc
    resolved_family = font_manager.FontProperties(fname=font_path).get_name()
    if resolved_family.casefold() != FONT_FAMILY.casefold():
        raise RuntimeError('Unexpected font substitution: ' + resolved_family)
    plt.rcParams.update({
        'font.family': [FONT_FAMILY],
        'font.serif': [FONT_FAMILY],
        'font.size': FONT_SIZES['base'],
        'axes.labelsize': FONT_SIZES['axis_label'],
        'axes.titlesize': FONT_SIZES['column_title'],
        'xtick.labelsize': FONT_SIZES['axis_tick'],
        'ytick.labelsize': FONT_SIZES['axis_tick'],
        'mathtext.fontset': MATH_FONTSET,
        'text.usetex': False,
        'axes.linewidth': .8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })
    print('Ordinary text font: ' + resolved_family, flush=True)
    print('Resolved font file: ' + str(font_path), flush=True)
    return font_path

def enforce_figure_fonts(fig, Text):
    """Clear explicit old font-file overrides while retaining each text size."""
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        properties = text.get_fontproperties().copy()
        properties.set_file(None)
        properties.set_family(FONT_FAMILY)
        text.set_fontproperties(properties)
        text.set_math_fontfamily(MATH_FONTSET)
