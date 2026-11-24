"""Figure 6/7: actual results, complete axes on every subplot, adjacent colorbars."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot_style import configure_typography,enforce_figure_fonts
from protocol import LABELS


FIG6_SIZE=(15.,7.3)

FIG7_COLUMN_WIDTH=4.8

FIG7_ROW_HEIGHT=3.1

FIG7_WSPACE=.65

COLORBAR_PAD=.035

COLORBAR_SIZE='3.5%'
COLORS={'go':'#263f57','fpinn':'#9b654d','fl1':'#657b65'}


def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def save(fig,path,dpi):
    enforce_figure_fonts(fig,Text)
    for ext in ['pdf','png']:
        tmp=path.parent/(path.name+'_%d.%s.tmp'%(os.getpid(),ext))
        fig.savefig(tmp,format=ext,dpi=dpi,bbox_inches='tight',pad_inches=.12)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path.with_suffix('.'+ext))
    plt.close(fig)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--root',type=Path,default=Path('runs/C02_gradient_v2'))
    q.add_argument('--seed',type=int,default=1234)
    q.add_argument('--font-dir',type=Path)
    q.add_argument('--font-family',default='Times New Roman')
    q.add_argument('--dpi',type=int,default=300)
    a=q.parse_args(argv);configure_typography(plt,font_manager,a.font_dir,a.font_family)
    folder=a.root/'evaluated';meta=json.loads((folder/'evaluation.json').read_text())
    from problem import SOURCE_MODEL
    if meta.get('source_model')!=SOURCE_MODEL:raise ValueError('Re-evaluate the new manufactured-source runs before plotting.')
    if meta['plot_seed']!=a.seed:raise ValueError('Regenerate fields for this plot seed before plotting.')
    curves=[r for r in read(folder/'training_curves.csv') if int(r['seed'])==a.seed]
    resources=[r for r in read(folder/'resources.csv') if int(r['seed'])==a.seed]
    methods=meta['methods'];out=a.root/'figures'/('seed_'+str(a.seed));out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'legend.fontsize':14})
    fig=plt.figure(figsize=FIG6_SIZE)
    gs=fig.add_gridspec(2,2,width_ratios=[1,1],height_ratios=[1,.65],wspace=.60,hspace=.65)
    loss_ax=fig.add_subplot(gs[:,0]);error_ax=fig.add_subplot(gs[0,1]);time_ax=fig.add_subplot(gs[1,1])
    for method in methods:
        rows=sorted([r for r in curves if r['method']==method],key=lambda r:int(r['global_step']))
        updates=[int(r['global_step']) for r in rows]
        loss_ax.plot(updates,[float(r['training_loss']) for r in rows],color=COLORS[method],lw=1.5,label=LABELS[method])
        error_ax.plot(updates,[float(r['validation_relative_l2']) for r in rows],color=COLORS[method],lw=1.5,label=LABELS[method])
    loss_ax.set(xlabel='Optimizer updates',ylabel='Training loss',yscale='log',title='(a) Training loss')
    ylabel=r'Relative $L_2$ error'
    error_ax.set(xlabel='Optimizer updates',ylabel=ylabel,yscale='log',title='(b) Error and training time')
    values=[next(float(r['optimization_seconds'])/3600 for r in resources if r['method']==m) for m in methods]
    time_ax.bar(range(len(methods)),values,color=[COLORS[m] for m in methods],width=.55)
    time_ax.set_xticks(range(len(methods)));time_ax.set_xticklabels([LABELS[m] for m in methods],fontsize=13)
    time_ax.set(xlabel='Method',ylabel='Training time (h)')
    for ax in [loss_ax,error_ax,time_ax]:
        ax.tick_params(labelleft=True,labelbottom=True,direction='out');ax.grid(axis='y',alpha=.15,linewidth=.6)
    for ax in [loss_ax,error_ax]:
        lo,hi=ax.get_ylim()
        if hi/lo<20:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            formatter=ticker.ScalarFormatter(useMathText=True)
            formatter.set_useOffset(False);formatter.set_powerlimits((-3,3))
            ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    loss_ax.legend(frameon=False);error_ax.legend(frameon=False,fontsize=11)
    if meta['smoke_only']:fig.text(.5,-.025,'SMOKE CHECK — not paper accuracy or timing',ha='center',fontsize=12,color='.4')
    save(fig,out/'Figure_6',a.dpi)
    with np.load(folder/'plot_fields.npz',allow_pickle=False) as a_fields:
        fields={k:a_fields[k].copy() for k in a_fields.files}
    displayed=[m for m in ['fpinn','fl1','go'] if m in methods]
    ncols=1+len(displayed)
    fig,axes=plt.subplots(2,ncols,squeeze=False,figsize=(FIG7_COLUMN_WIDTH*ncols,FIG7_ROW_HEIGHT*2))
    fig.subplots_adjust(left=.07,right=.98,bottom=.14,top=.90,wspace=FIG7_WSPACE,hspace=.53)
    ref=fields['reference'];amp=max(float(abs(ref.real).max()),float(abs(ref.imag).max()),1e-15)
    # Use componentwise errors; choose ONE maximum over both components/methods.
    maximum=max([float(np.abs((fields['prediction_'+m]-ref).real).max()) for m in displayed]+
                [float(np.abs((fields['prediction_'+m]-ref).imag).max()) for m in displayed]+[1e-15])
    for row in [0,1]:
        for col in range(ncols):
            ax=axes[row,col]
            if col==0:
                array=ref.real if row==0 else ref.imag;vmin=-amp;vmax=amp;cmap='viridis'
                title='Numerical reference' if meta['reference_role'].startswith('NUMERICAL') else 'Exact reference'
            else:
                method=displayed[col-1];diff=fields['prediction_'+method]-ref
                array=np.abs(diff.real if row==0 else diff.imag);vmin=0;vmax=maximum;cmap='magma'
                title=LABELS[method]+' error'
            image=ax.pcolormesh(fields['x'],fields['t'],array.T,shading='auto',cmap=cmap,
                                vmin=vmin,vmax=vmax,rasterized=True)
            ax.set(xlabel=r'$x$',ylabel=r'$t$',ylim=(0,1))
            ax.set_xticks(np.array([-3,0,3])*np.pi);ax.set_xticklabels([r'$-3\pi$',r'$0$',r'$3\pi$'])
            ax.set_yticks([0,.5,1]);ax.tick_params(labelleft=True,labelbottom=True,direction='out')
            if row==0:ax.set_title(title,pad=12,fontsize=19)
            divider=make_axes_locatable(ax);cax=divider.append_axes('right',size=COLORBAR_SIZE,pad=COLORBAR_PAD)
            bar=fig.colorbar(image,cax=cax);bar.locator=ticker.MaxNLocator(3);bar.update_ticks()
            bar.ax.tick_params(labelsize=16)
            formatter=ticker.ScalarFormatter(useMathText=True);formatter.set_powerlimits((-2,3));bar.formatter=formatter;bar.update_ticks()
    fig.canvas.draw()
    for row,label in enumerate(['Real part','Imaginary part']):
        pos=axes[row,0].get_position();fig.text(.009,(pos.y0+pos.y1)/2,label,rotation=90,va='center',fontsize=21)
    if meta['smoke_only']:fig.text(.5,-.045,'SMOKE CHECK — not a trained paper result',ha='center',fontsize=12,color='.4')
    save(fig,out/'Figure_7',a.dpi)
    atomic=dict(reference_role=meta['reference_role'],reference_is_analytic=meta['reference_is_analytic'],source_model=SOURCE_MODEL,
                every_subplot_has_labels_and_ticks=True,colorbar_gap_inches=COLORBAR_PAD,
                common_component_error_maximum=maximum,font_family=a.font_family,smoke_only=meta['smoke_only'])
    (out/'figure_settings.json').write_text(json.dumps(atomic,indent=2)+'\n')
    print('Figure 6/7 saved in '+str(out))


if __name__=='__main__':main()
