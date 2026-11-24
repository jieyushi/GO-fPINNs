"Plot coefficient-identification trajectories and stage-wise errors."
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter,MaxNLocator
from plot_style import configure_typography,enforce_figure_fonts,FONT_SIZES
from protocol import TRUE,METHODS,LABELS,CASES,FIGURE_CASES,unknown_names


FIGURE_WIDTH=22.             
FIGURE_HEIGHT=8.2            
HORIZONTAL_SPACE=.62         
VERTICAL_SPACE=.38           
AXIS_LABEL_SIZE=26           
AXIS_TICK_SIZE=22            
TITLE_SIZE=26                
LEGEND_SIZE=20               
LINE_WIDTH=1.6               
STAGE_MARKER_SIZE=7          
ERROR_BOUND=.01             
PNG_DPI=300
STYLE={'fl1':dict(color='#d95f02',ls='--',marker='^'),
       'go_single':dict(color='#b23b73',ls='-.',marker='s'),
       'go_multi':dict(color='#1976b9',ls='-',marker='o')}
SYMBOLS={'alpha':r'\alpha','beta':r'\beta','gamma':r'\gamma'}


def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def complete(ax):
    ax.tick_params(direction='in',top=True,right=True,labelbottom=True,labelleft=True)


def plot(root,seed=1234,cases=None,font_dir=None,font_family=None):
    root=Path(root);cases=list(cases or FIGURE_CASES);records={};present=set();smoke_flags=set()
    for case in cases:
        if case not in CASES:raise ValueError('Unknown case '+case)
        common_observations=None
        for method in METHODS:
            run=root/case/method/('seed_'+str(seed))
            if not (run/'summary.json').exists():continue
            config=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
            if config['args']['unknown']!=CASES[case]['unknown'] or config['args']['initial_values']!=CASES[case]['initial']:
                raise ValueError('Case directory does not match its recorded unknowns/initial values.')
            with np.load(run/'observations.npz',allow_pickle=False) as o:
                h=hashlib.sha256(o['tx'].tobytes()+o['target'].tobytes()).hexdigest()
            if common_observations is not None and common_observations!=h:raise ValueError('Methods do not share the same observations.')
            common_observations=h
            records[case,method]=(config,summary,read_csv(run/'trajectory.csv'))
            present.add(method);smoke_flags.add(config['args']['smoke'])
    if not records:raise FileNotFoundError('No completed runs for the selected cases/seed.')
    if len(smoke_flags)>1:raise ValueError('Cannot combine smoke and formal runs.')
    smoke=next(iter(smoke_flags))
    panels=[(case,name) for case in cases for name in unknown_names(CASES[case]['unknown'])]
    FONT_SIZES.update(base=24,axis_label=AXIS_LABEL_SIZE,axis_tick=AXIS_TICK_SIZE,column_title=TITLE_SIZE)
    font=configure_typography(plt,font_manager,font_dir,font_family)
    width=FIGURE_WIDTH if len(panels)==6 else max(8.,len(panels)*3.7)
    fig,axes=plt.subplots(2,len(panels),figsize=(width,FIGURE_HEIGHT),squeeze=False)
    fig.subplots_adjust(left=.055,right=.99,top=.84,bottom=.18,wspace=HORIZONTAL_SPACE,hspace=VERTICAL_SPACE)
    for column,(case,name) in enumerate(panels):
        initial=CASES[case]['initial'][unknown_names(CASES[case]['unknown']).index(name)]
        top,bottom=axes[:,column];truth=TRUE[name]
        staged=records.get((case,'go_multi'))
        if staged:
            previous=0
            for item in staged[1]['stages']:
                end=item['global_step'];sid=item['stage']
                if sid%2==0:
                    top.axvspan(previous,end,color='#eceff1',zorder=-2)
                    bottom.axvspan(sid-.5,sid+.5,color='#eceff1',zorder=-2)
                top.text((previous+end)/2,1.015,'S%d'%sid,transform=top.get_xaxis_transform(),
                         ha='center',va='bottom',fontsize=13)
                previous=end
        for method in METHODS:
            record=records.get((case,method))
            if record is None:continue
            config,summary,trajectory=record;style=STYLE[method]
            step=[0]+[int(r['global_step']) for r in trajectory]
            value=[initial]+[float(r[name]) for r in trajectory]
            top.plot(step,value,color=style['color'],ls=style['ls'],lw=LINE_WIDTH)
            if method=='go_multi':
                stage=summary['stages']
                bottom.semilogy([r['stage'] for r in stage],
                                [max(r['best_parameter_errors'][name+'_relative_error'],1e-16) for r in stage],
                                color=style['color'],ls=style['ls'],marker=style['marker'],
                                markersize=STAGE_MARKER_SIZE,lw=LINE_WIDTH)
            else:
                
                error=max(summary['parameter_errors'][name+'_relative_error'],1e-16)
                bottom.axhline(error,color=style['color'],ls=style['ls'],lw=LINE_WIDTH)
        top.axhline(truth,color='.3',ls='--',lw=1.2)
        top.axhline(truth*(1-ERROR_BOUND),color='#bb3030',ls=':',lw=1.)
        top.axhline(truth*(1+ERROR_BOUND),color='#bb3030',ls=':',lw=1.)
        top.plot([0],[initial],marker='*',ms=13,mfc='white',mec='.4',ls='none',clip_on=False)
        top.set_title('$'+SYMBOLS[name]+'_0=%g$'%initial,pad=21)
        top.set(xlabel='Epoch',ylabel='Estimate')
        top.xaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
        if top.get_xlim()[1]>=1000:
            top.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:'%g'%(v/1000)))
            top.set_xlabel(r'Epoch ($\times10^3$)')
        top.grid(alpha=.18,ls=':');complete(top)
        bottom.set_yscale('log');bottom.axhline(ERROR_BOUND,color='#bb3030',ls=':',lw=1.2)
        stages=range(len(staged[1]['stages'])) if staged else range(1)
        bottom.set_xticks(list(stages),['S%d'%v for v in stages])
        bottom.tick_params(axis='x',labelsize=20)
        bottom.set(xlabel='Stage',ylabel='Relative error');bottom.grid(alpha=.18,ls=':');complete(bottom)
    handles=[Line2D([],[],color=STYLE[m]['color'],ls=STYLE[m]['ls'],marker=STYLE[m]['marker'],
                    label=LABELS[m],lw=LINE_WIDTH) for m in METHODS if m in present]
    handles += [Line2D([],[],color='.3',ls='--',label='True value'),
                Line2D([],[],color='.4',ls='none',marker='*',mfc='white',markersize=12,label='Initial value'),
                Line2D([],[],color='#bb3030',ls=':',label='1% error bound')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.01),
               ncol=min(len(handles),6 if len(panels)>=6 else 3),fontsize=LEGEND_SIZE,frameon=True)
    fig.text(.5,.055,'Bottom row: stage-wise best training-loss estimates; horizontal references: final single-stage estimates.',
             ha='center',fontsize=18)
    if smoke:fig.text(.5,.01,'SMOKE CHECK: NOT A TRAINED RESULT',ha='center',fontsize=16)
    enforce_figure_fonts(fig,Text)
    out=root/'figures'/('seed_'+str(seed));out.mkdir(parents=True,exist_ok=True)
    filename='Figure_10_revise' if cases==list(FIGURE_CASES) else 'inverse_identification'
    for ext in ['pdf','png']:fig.savefig(out/(filename+'.'+ext),dpi=PNG_DPI,bbox_inches='tight')
    plt.close(fig)
    (out/'plot_metadata.json').write_text(json.dumps(dict(cases=cases,seed=seed,methods=sorted(present),font_file=font,
       smoke=smoke,baseline_stage_values_invented=False,selection='finest-stage minimum pre-update training loss'),indent=2)+'\n')
    print('Saved figures to '+str(out),flush=True)
    return out


def main(argv=None):
    from protocol import RUN_ROOT
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--root',default=RUN_ROOT)
    q.add_argument('--seed',type=int,default=1234);q.add_argument('--cases',default=','.join(FIGURE_CASES))
    q.add_argument('--font-dir');q.add_argument('--font-family');a=q.parse_args(argv)
    return plot(a.root,a.seed,a.cases.split(','),a.font_dir,a.font_family)


if __name__=='__main__':main()
