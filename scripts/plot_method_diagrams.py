"""Draw editable vector versions of the conceptual and method schematics.

These deterministic schematics reconstruct the methodological relationships;
their geometry is not an extraction of the manuscript's original artwork.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

INK = '#28333d'
ACCENT = '#435c72'
GRAY = '#f4f5f6'


def panel(ax, xy, width, height, text, fontsize=10, fill='white', bold=False):
    ax.add_patch(Rectangle(xy, width, height, edgecolor=ACCENT,
                           facecolor=fill, linewidth=0.9))
    ax.text(xy[0]+width/2, xy[1]+height/2, text, ha='center', va='center',
            fontsize=fontsize, color=INK, fontweight='bold' if bold else 'normal',
            linespacing=1.5)


def arrow(ax, start, end, label=None, offset=(0, .025)):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=10, color=ACCENT, linewidth=0.9))
    if label:
        ax.text((start[0]+end[0])/2+offset[0], (start[1]+end[1])/2+offset[1],
                label, ha='center', va='bottom', fontsize=8.5, color=INK)


def new_figure(width=9.2, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.975)
    return fig, ax


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/(name+'.pdf'), bbox_inches='tight',
                metadata={'Author':'', 'Creator':'Matplotlib'})
    fig.savefig(output/(name+'.svg'), bbox_inches='tight',
                metadata={'Creator':'Matplotlib'})
    plt.close(fig)


def conceptual(output, name='map'):
    fig, ax = new_figure()
    panel(ax, (.02,.55), .24,.29,
          'NTFS problem family\nForward / inverse / operator', fill=GRAY)
    panel(ax, (.02,.13), .24,.27,
          'Coordinates and parameters\nEquation and source\nInitial and boundary data', fontsize=9.4)
    panel(ax, (.36,.60), .31,.25,
          'Task-configured branch selector\nCoordinate or parameter-conditioned', fill=GRAY)
    panel(ax, (.36,.12), .31,.32,
          'Physics-informed training and transfer\nFast Caputo residual\nRepresentation and loss adaptation\nProgressive enrichment', fontsize=9.5)
    panel(ax, (.78,.56), .20,.26,
          'Solution fields\nCoefficients and potentials\nSolution operator', fontsize=9.0, fill=GRAY)
    panel(ax, (.78,.13), .20,.27,
          'Held-out errors\nReference convergence\nTraining / inference cost', fontsize=9.0)
    arrow(ax, (.14,.55), (.14,.40))
    arrow(ax, (.26,.70), (.36,.72))
    arrow(ax, (.26,.27), (.36,.27))
    arrow(ax, (.515,.60), (.515,.44))
    arrow(ax, (.67,.72), (.78,.69))
    arrow(ax, (.67,.27), (.78,.27))
    arrow(ax, (.88,.56), (.88,.40))
    save(fig, output, name)


def architecture(output):
    fig, ax = new_figure(height=4.7)
    panel(ax, (.02,.62), .21,.21, 'Coordinates\n'+r'$(\boldsymbol{x},t)\in\mathbb{R}^{d+1}$', fill=GRAY)
    panel(ax, (.02,.19), .21,.21, 'Physical parameters\n'+r'$\boldsymbol{\mu}\in\mathbb{R}^{d_\mu}$', fill=GRAY)
    panel(ax, (.31,.14), .25,.29,
          'Normalize + Fourier lift\nParameter encoder: 2 x 64\nLatent features: 16', fontsize=9.5)
    panel(ax, (.31,.60), .25,.26,
          'Prescribed branch selection\n'+r'$g=0:\ d+1$'+'\n'+r'$g=1:\ d+1+d_z$', fontsize=9.5)
    panel(ax, (.67,.69), .29,.16, 'Real-part solution network\n'+r'$u(\boldsymbol{x},t;\boldsymbol{\mu})$', fontsize=9.5)
    panel(ax, (.67,.44), .29,.16, 'Imaginary-part solution network\n'+r'$v(\boldsymbol{x},t;\boldsymbol{\mu})$', fontsize=9.5)
    panel(ax, (.67,.09), .29,.22,
          'Physics-informed objective\nEquation + initial + boundary\nObservations when required', fontsize=9.5, fill=GRAY)
    arrow(ax, (.23,.725), (.31,.725))
    arrow(ax, (.23,.295), (.31,.295))
    arrow(ax, (.435,.43), (.435,.60), label=r'$g=1$')
    arrow(ax, (.56,.77), (.67,.77))
    arrow(ax, (.56,.67), (.67,.52))
    arrow(ax, (.815,.44), (.815,.31))
    ax.plot([.96,.985,.985],[.77,.77,.20],color=ACCENT,linewidth=.9)
    arrow(ax,(.985,.20),(.96,.20))
    ax.text(.435,.035, 'Gate fixed before training; task-specific network input dimension',
            ha='center', va='center', fontsize=8.3, color=INK)
    save(fig, output, 'Figure_1_revise')


def enrichment(output):
    fig, ax = new_figure(height=4.6)
    widths = [.21, .21, .21]
    xs = [.24, .49, .74]
    titles = ['Coarse learning space', 'Enriched learning space', 'Final learning space']
    for x, title in zip(xs, titles):
        ax.text(x+.105, .91, title, ha='center', va='center', fontsize=10, color=INK)
    labels = [('Forward', 'Coarse space-time grid', 'Refined grid', 'Refinement / decomposition'),
              ('Inverse', 'Initial basis', 'Expanded basis', 'Enriched potential\nrepresentation'),
              ('Operator', 'Initial parameter samples', 'Nested sample enrichment', 'Broader parameter coverage')]
    for row, (title, *texts) in enumerate(labels):
        y = .65-row*.24
        ax.text(.03,y+.06,title,ha='left',va='center',fontsize=10,color=INK)
        for x, width, text in zip(xs, widths, texts):
            panel(ax,(x,y),width,.13,text,fontsize=8.5,fill=GRAY if row==0 else 'white')
        for start, end in [(.45,.49),(.70,.74)]:arrow(ax,(start,y+.065),(end,y+.065))
    ax.text(.60,.055, 'Transfer inherited information; temporarily constrain prediction changes',
            ha='center',va='center',fontsize=9,color=INK)
    save(fig,output,'Figure_2_revise')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/method_diagrams'))
    args = parser.parse_args(argv)
    plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42,
                         'svg.fonttype':'none','mathtext.fontset':'dejavuserif'})
    conceptual(args.output)
    conceptual(args.output, 'graphical_abstract')
    architecture(args.output)
    enrichment(args.output)
    print('Wrote four PDF/SVG method schematics to '+str(args.output))


if __name__ == '__main__':
    main()
