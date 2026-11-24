"Plot the long-time GO-fPINNs training curves, fields, and errors."
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm,Normalize
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot_style import configure_typography,enforce_figure_fonts,FONT_SIZES


TRAIN_FIGSIZE=(16.,5.4)       
FIELD_FIGSIZE=(14.,7.2)       
FIELD_WSPACE=.48             
FIELD_HSPACE=.38             
COLORBAR_SIZE='4%'           
COLORBAR_PAD=.035            
ERROR_DISPLAY_MIN=1e-4       
OUTPUT_DPI=300              
LINE_WIDTH=1.7              



def save(fig,path,smoke):
    if smoke:fig.text(.5,-.15,'SMOKE CHECK: NOT A TRAINED RESULT',ha='center',fontsize=FONT_SIZES['verification'])
    enforce_figure_fonts(fig,Text)
    for ext in ['pdf','png']:fig.savefig(path.with_suffix('.'+ext),dpi=OUTPUT_DPI,bbox_inches='tight')
    plt.close(fig)


def complete_axes(ax):
    ax.tick_params(axis='both',direction='in',top=True,right=True,labelbottom=True,labelleft=True)


def plot(run,font_dir=None,font_family=None):
    run=Path(run);evaluated=run/'evaluated';out=run/'figures';out.mkdir(exist_ok=True)
    info=json.loads((evaluated/'metrics.json').read_text())
    font=configure_typography(plt,font_manager,font_dir,font_family)
    with (run/'training.csv').open(newline='') as f:training=list(csv.DictReader(f))
    with (evaluated/'stage_errors.csv').open(newline='') as f:stages=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=TRAIN_FIGSIZE);fig.subplots_adjust(wspace=.48)
    ax=axes[0];steps=np.array([int(r['global_step']) for r in training]);loss=np.array([float(r['loss']) for r in training])
    ax.semilogy(steps,np.maximum(loss,1e-30),color='#174c78',lw=LINE_WIDTH,label='GO-fPINNs')
    previous=0
    for row in stages:
        sid=int(row['stage']);end=int(row['global_step'])
        if sid%2==0:ax.axvspan(previous,end,color='#e9edf0',zorder=-1)
        if sid<8:ax.axvline(end,color='#bbc4ca',lw=.55,ls=':',zorder=-1)
        previous=end
    ax.text(0,1.02,'S0–S4: refinement; S5–S8: time windows',transform=ax.transAxes,fontsize=14,va='bottom')
    ax.set(xlabel='Optimizer update',ylabel='Training loss (scaled)');ax.legend(loc='upper right',fontsize=17)
    ax.grid(alpha=.2,ls=':');complete_axes(ax)
    ax=axes[1];second=ax.twinx();sid=np.array([int(r['stage']) for r in stages])
    times=[float(r['cumulative_optimization_seconds']) for r in stages]
    errors=[max(float(r['relative_l2']),1e-30) for r in stages]
    bar=ax.bar(sid,times,width=.46,color='#d6e1e9',edgecolor='#174c78',label='Training time')
    line=second.semilogy(sid,errors,'o-',color='#174c78',lw=LINE_WIDTH,markersize=7,label=r'Relative $L_2$ error')
    ax.set(xlabel='Training stage',ylabel='Training time (s)');ax.set_xticks(sid,['S%d'%v for v in sid])
    ax.tick_params(axis='x',labelsize=17);second.set_ylabel(r'Relative $L_2$ error')
    ax.legend([bar,line[0]],['Training time',r'Relative $L_2$ error'],loc='upper right',fontsize=15)
    complete_axes(ax);second.tick_params(direction='in');ax.grid(axis='y',alpha=.2,ls=':')
    for ax,label in zip(axes,['(a)','(b)']):ax.text(.5,-.25,label,transform=ax.transAxes,ha='center',fontsize=24)
    save(fig,out/'Figure_8_GO',info['smoke'])
    with np.load(evaluated/'full_interval.npz',allow_pickle=False) as d:
        x=d['x'];t=d['t'];pred=d['prediction'];true=d['reference']
    component_error=[np.abs(pred.real-true.real),np.abs(pred.imag-true.imag)]
    vmax=max(float(z.max()) for z in component_error)
    vmax=max(vmax,1e-14);vmin=min(ERROR_DISPLAY_MIN,vmax*1e-3)
    err_norm=LogNorm(vmin=vmin,vmax=vmax)
    exact_norm=Normalize(-100,100)
    fig,axes=plt.subplots(2,2,figsize=FIELD_FIGSIZE,sharex=True,sharey=True)
    fig.subplots_adjust(wspace=FIELD_WSPACE,hspace=FIELD_HSPACE)
    for row in range(2):
        exact=true.real if row==0 else true.imag
        for col,values in enumerate([exact,component_error[row]]):
            ax=axes[row,col];shown=values if col==0 else np.maximum(values,vmin)
            artist=ax.pcolormesh(t,x,shown,shading='auto',rasterized=True,
                                 cmap='RdBu_r' if col==0 else 'Blues',norm=exact_norm if col==0 else err_norm)
            ax.set(xlabel=r'$t$',ylabel=r'$x$',xlim=(0,100),ylim=(0,1));ax.set_xticks([0,25,50,75,100]);ax.set_yticks([0,.5,1])
            complete_axes(ax)
            label=(r'$u$' if row==0 else r'$v$') if col==0 else (r'$|u-\widetilde{u}|$' if row==0 else r'$|v-\widetilde{v}|$')
            ax.text(.035,.86,label,transform=ax.transAxes,fontsize=24,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=2))
            cax=make_axes_locatable(ax).append_axes('right',size=COLORBAR_SIZE,pad=COLORBAR_PAD)
            cb=fig.colorbar(artist,cax=cax);cb.ax.tick_params(labelsize=FONT_SIZES['colorbar_tick'],direction='in')
            if row==0:ax.set_title(['Exact solution','GO-fPINNs error'][col],pad=12)
    save(fig,out/'Figure_9_GO',info['smoke'])
    (out/'plot_metadata.json').write_text(json.dumps(dict(font_file=font,metrics=info,
                time_axis='horizontal',space_axis='vertical',baseline_included=False),indent=2)+'\n')
    print('Figures saved to '+str(out),flush=True)
    return out


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--run',default='runs/C03/seed_1234')
    q.add_argument('--font-dir');q.add_argument('--font-family',help='Explicit alternative font; default is strict Times New Roman.')
    a=q.parse_args(argv);return plot(a.run,a.font_dir,a.font_family)


if __name__=='__main__':main()
