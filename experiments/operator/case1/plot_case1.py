"""Paper plots: complete x/y labels and ticks in EVERY panel; adjacent colorbars.

Figure 13: predicted real/imaginary fields and their componentwise errors.
Figure 14: actually available clean test curves only, mean/sample SD over seeds.
No synthetic baseline curves and no invented 1% noise results are inserted.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot_style import configure_typography,enforce_figure_fonts,FONT_SIZES


FIELD_COLUMN_WIDTH=5.4

FIELD_ROW_HEIGHT=4.25

FIELD_WSPACE=.50

FIELD_HSPACE=.42

COLORBAR_PAD=.035

COLORBAR_SIZE='4%'

CURVE_COLUMN_WIDTH=5.5

CURVE_HEIGHT=4.9

SYMBOLS={'alpha':r'\alpha','beta':r'\beta','gamma':r'\gamma'}
ROW_LABELS=[r'Predicted $u$',r'Predicted $v$',r'$|u-u^*|$',r'$|v-v^*|$']


def save_figure(fig,path,dpi):
    """Atomic writes avoid publishing a partly written PNG/PDF."""
    enforce_figure_fonts(fig,Text)
    for ext in ['pdf','png']:
        target=path.with_suffix('.'+ext)
        temporary=path.parent/(path.name+'_%d.%s.tmp'%(os.getpid(),ext))
        fig.savefig(temporary,format=ext,dpi=dpi,bbox_inches='tight',pad_inches=.10)
        with temporary.open('rb') as f:os.fsync(f.fileno())
        os.replace(temporary,target)
    plt.close(fig)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--root',type=Path,required=True)
    q.add_argument('--operators',default='alpha,beta,gamma')
    q.add_argument('--seed',type=int,default=1234)
    q.add_argument('--output',type=Path)
    q.add_argument('--font-family',default='Times New Roman')
    q.add_argument('--font-dir',type=Path)
    q.add_argument('--dpi',type=int,default=300)
    a=q.parse_args(argv);operators=a.operators.split(',')
    font_path=configure_typography(plt,font_manager,a.font_dir,a.font_family)
    plt.rcParams.update({'legend.fontsize':16})
    data=a.root/'paper_data'/('seed_'+str(a.seed))
    out=a.output if a.output else a.root/'figures';out.mkdir(parents=True,exist_ok=True)
    fields=[];metadata=[]
    for operator in operators:
        with np.load(data/('fields_'+operator+'.npz'),allow_pickle=False) as values:
            fields.append({key:values[key].copy() for key in values.files})
        metadata.append(json.loads((data/('fields_'+operator+'.json')).read_text()))
    low=min(float(f['prediction'].min()) for f in fields)
    high=max(float(f['prediction'].max()) for f in fields)
    error_high=max(float(np.abs(f['prediction']-f['exact']).max()) for f in fields)
    if high<=low:high=low+1e-12
    error_high=max(error_high,1e-15)
    fig,axes=plt.subplots(4,len(operators),squeeze=False,
                          figsize=(FIELD_COLUMN_WIDTH*len(operators),FIELD_ROW_HEIGHT*4))
    fig.subplots_adjust(left=.09,right=.96,bottom=.070,top=.96,
                        wspace=FIELD_WSPACE,hspace=FIELD_HSPACE)
    for col,(operator,field,meta) in enumerate(zip(operators,fields,metadata)):
        errors=np.abs(field['prediction']-field['exact'])
        arrays=[field['prediction'][...,0],field['prediction'][...,1],errors[...,0],errors[...,1]]
        for row,array in enumerate(arrays):
            ax=axes[row,col]
            image=ax.pcolormesh(field['x'],field['y'],array.T,shading='auto',rasterized=True,
                                cmap='viridis' if row<2 else 'magma',
                                vmin=low if row<2 else 0,vmax=high if row<2 else error_high)
            ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel=r'$x$',ylabel=r'$y$')
            ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
            ax.tick_params(labelbottom=True,labelleft=True,direction='out',width=.8,length=4)
            if row==0:
                ax.set_title('(%s) $%s=%.3f$'%(chr(97+col),SYMBOLS[operator],meta['parameter']['active_value']),pad=12)
            divider=make_axes_locatable(ax)
            cax=divider.append_axes('right',size=COLORBAR_SIZE,pad=COLORBAR_PAD)
            bar=fig.colorbar(image,cax=cax)
            bar.locator=ticker.MaxNLocator(nbins=3);bar.update_ticks()
            bar.ax.tick_params(labelsize=FONT_SIZES['colorbar_tick'],length=3)
            bar.formatter=ticker.ScalarFormatter(useMathText=True)
            bar.formatter.set_powerlimits((-2,3));bar.update_ticks()
    # Row names are kept separate from the y labels, which remain on ALL panels.
    fig.canvas.draw()
    for row,label in enumerate(ROW_LABELS):
        pos=axes[row,0].get_position()
        fig.text(.014,(pos.y0+pos.y1)/2,label,rotation=90,va='center',ha='center',fontsize=22)
    smoke=any(m['status']=='SMOKE_ONLY' for m in metadata)
    if smoke:fig.text(.5,.004,'SMOKE CHECK — not a trained paper result',ha='center',color='.35',fontsize=12)
    save_figure(fig,out/'Figure_13_revise',a.dpi)
    # Main curves contain only actual outputs. The optional reconstructed
    # baseline is labelled explicitly; clean runs never produce a noisy curve.
    with (a.root/'aggregated/error_curves.csv').open(newline='') as f:curves=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,len(operators),squeeze=False,
                          figsize=(CURVE_COLUMN_WIDTH*len(operators),CURVE_HEIGHT))
    fig.subplots_adjust(left=.075,right=.99,bottom=.20,top=.88,wspace=.62)
    styles={'go':('GO-fPINNs','#253c54','o','-'),
            'l1_reimplementation':('L1-DeepONet (reimpl.)','#915a43','s','--')}
    for col,operator in enumerate(operators):
        ax=axes[0,col]
        for method,(label,color,marker,line) in styles.items():
            rows=sorted([r for r in curves if r['operator']==operator and r['method']==method],
                         key=lambda r:float(r['active_value']))
            if not rows:continue
            x=np.array([float(r['active_value']) for r in rows])
            y=np.array([float(r['relative_l2_mean']) for r in rows])
            if np.any(y<=0):raise ValueError('Log error plots require positive relative errors.')
            sd=np.array([float(r['relative_l2_sample_sd']) if r['relative_l2_sample_sd'] else np.nan for r in rows])
            ax.plot(x,y,color=color,lw=1.5,marker=marker,ms=4,markevery=2,linestyle=line,label=label)
            if np.all(np.isfinite(sd)):
                # A nonpositive lower SD bound has no location on a log axis.
                # Leave that band segment blank; do not replace it by a tiny
                # number that would distort the y limits or the uncertainty.
                lower=np.where(y-sd>0,y-sd,np.nan)
                ax.fill_between(x,lower,y+sd,color=color,alpha=.12,linewidth=0)
        ax.set(xlabel='$'+SYMBOLS[operator]+'$',ylabel=r'Relative $L_2$ error',yscale='log')
        ax.set_title('(%s) $%s$-operator'%(chr(97+col),SYMBOLS[operator]),pad=10)
        ranges={'alpha':(.6,.9),'beta':(.5,1.5),'gamma':(1,2)}
        ax.set_xlim(*ranges[operator]);ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ymin,ymax=ax.get_ylim()
        if ymax/ymin<20:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=3))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
        else:
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10,numticks=5))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(axis='y',alpha=.16,linewidth=.6);ax.tick_params(labelleft=True,labelbottom=True,direction='out')
        ax.legend(frameon=False,loc='best',fontsize=12)
    if smoke:fig.text(.5,.015,'SMOKE CHECK — not a trained paper result',ha='center',color='.35',fontsize=12)
    save_figure(fig,out/'Figure_14_revise',a.dpi)
    record={'font_family':a.font_family,'resolved_font_path':font_path,'operators':operators,
            'figure13_seed':a.seed,'every_subplot_has_xy_labels_and_ticks':True,
            'colorbar_gap_inches':COLORBAR_PAD,'common_prediction_scale':[low,high],
            'common_component_error_scale':[0,error_high],
            'baseline_original_reproduction_claimed':False,'noise_curves_generated':False,'smoke_only':smoke}
    (out/'figure_settings.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Saved Figure_13_revise and Figure_14_revise in '+str(out))


if __name__=='__main__':main()
