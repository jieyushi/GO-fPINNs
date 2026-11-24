"Plot one-dimensional potential reconstructions and sparse observations."
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from plot_style import configure_typography,enforce_figure_fonts,FONT_SIZES
from problem import eta_truth
from protocol import METHODS,LABELS
from train_potential import FORMAT,load_for_prediction,potential_numpy


FIGURE_WIDTH=14.2       
FIGURE_HEIGHT=6.6       
FINAL_FIGURE_HEIGHT=4.8 
AXIS_LABEL_SIZE=20     
AXIS_TICK_SIZE=16      
LEGEND_SIZE=16         
LINE_WIDTH=1.8         
SUPERVISED_SIZE=40     
VIEW_ELEV=23           
VIEW_AZIM=-62          
PNG_DPI=300            
COLORS={'fl1':'#ba3545','go_multi':'#194c78','truth':'#444444','supervised':'#f4c445'}


def save(fig,path):
    enforce_figure_fonts(fig,Text)
    extras=list(fig.texts)+list(fig.legends)+[ax.zaxis.label for ax in fig.axes if hasattr(ax,'zaxis')]
    for ext in ['pdf','png']:fig.savefig(path.with_suffix('.'+ext),dpi=PNG_DPI,bbox_inches='tight',bbox_extra_artists=extras or None,pad_inches=.15)
    plt.close(fig)


def plot(root,problem='manufactured_series',counts=(3,2),seed=1234,font_dir=None,font_family=None):
    torch.set_num_threads(1);root=Path(root);counts=list(counts);records={};smoke=set()
    for count in counts:
        common=None
        for method in METHODS:
            run=root/problem/('eta_'+str(count))/method/('seed_'+str(seed))
            if not (run/'summary.json').exists():continue
            config=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
            if config.get('format')!=FORMAT or summary.get('format')!=FORMAT:
                raise ValueError('Use results produced by the current source/residual version.')
            if config['args']['problem']!=problem or config['args']['supervised_points']!=count:raise ValueError('Directory/settings mismatch.')
            with np.load(run/'observations.npz',allow_pickle=False) as a:
                data={k:a[k].copy() for k in ['eta_x','eta_target','obs_tx','obs_target']}
            key=hashlib.sha256(b''.join(data[k].tobytes() for k in sorted(data))).hexdigest()
            if common is not None and common!=key:raise ValueError('Methods do not share supervision and field data.')
            common=key;records[count,method]=(run,summary,data);smoke.add(config['args']['smoke'])
    if not records:raise FileNotFoundError('No completed selected runs.')
    if len(smoke)!=1:raise ValueError('Cannot mix formal and smoke curves.')
    FONT_SIZES.update(base=18,axis_label=AXIS_LABEL_SIZE,axis_tick=AXIS_TICK_SIZE,column_title=20)
    font=configure_typography(plt,font_manager,font_dir,font_family)
    out=root/'figures'/problem/('seed_'+str(seed));out.mkdir(parents=True,exist_ok=True)
    x=np.linspace(0,np.pi,501);truth=eta_truth(x)
    handles=[Line2D([],[],color=COLORS['truth'],ls=':',lw=2,label=r'True $\eta(x)$')]
    for method in METHODS:
        if any(m==method for n,m in records):handles.append(Line2D([],[],color=COLORS[method],lw=LINE_WIDTH,label=LABELS[method]))
    handles.append(Line2D([],[],ls='none',marker='o',color='black',markerfacecolor=COLORS['supervised'],label='Supervised points'))
    fig=plt.figure(figsize=(FIGURE_WIDTH if len(counts)==2 else 8,FIGURE_HEIGHT))
    for col,count in enumerate(counts):
        ax=fig.add_subplot(1,len(counts),col+1,projection='3d');curves=[]
        go=records.get((count,'go_multi'));base=records.get((count,'fl1'))
        stage_ids=range(len(go[1]['stages'])) if go else range(1)
        for sid in stage_ids:
            ax.plot(x,np.full_like(x,sid),truth,color=COLORS['truth'],ls=':',lw=2)
            if go:
                state,_=load_for_prediction(go[0]/('stage_%02d_best_loss.pt'%sid))
                pred=potential_numpy(state,x);curves.append(pred)
                ax.plot(x,np.full_like(x,sid),pred,color=COLORS['go_multi'],lw=LINE_WIDTH)
            data=(go or base)[2]
            ax.scatter(data['eta_x'].ravel(),np.full(count,sid),data['eta_target'].ravel(),
                       c=COLORS['supervised'],edgecolors='black',s=SUPERVISED_SIZE,depthshade=False,zorder=5)
        if base:
            
            state,_=load_for_prediction(base[0]/'final.pt');pred=potential_numpy(state,x);curves.append(pred)
            final_sid=len(go[1]['stages'])-1 if go else 0
            ax.plot(x,np.full_like(x,final_sid),pred,color=COLORS['fl1'],lw=LINE_WIDTH)
        ax.set(xlabel='$x$',ylabel='Stage',zlabel=r'$\eta(x)$',xlim=(0,np.pi))
        ax.set_xticks([0,np.pi/2,np.pi],['0',r'$\pi/2$',r'$\pi$'])
        ax.set_yticks(list(stage_ids),['S%d'%v for v in stage_ids]);ax.invert_yaxis()
        ax.set_title('(%s) %d supervised points'%(chr(97+col),count),pad=16)
        ax.xaxis.labelpad=8;ax.yaxis.labelpad=12;ax.zaxis.labelpad=9
        all_values=np.concatenate([truth]+curves) if curves else truth
        lo,hi=float(all_values.min()),float(all_values.max());pad=max(.08,.08*(hi-lo))
        ax.set_zlim(min(-1.05,lo-pad),max(.05,hi+pad));ax.view_init(elev=VIEW_ELEV,azim=VIEW_AZIM)
        ax.set_box_aspect((1.3,1.,1.))
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
    fig.legend(handles=handles,loc='upper center',ncol=len(handles),fontsize=LEGEND_SIZE,bbox_to_anchor=(.5,1.02))
    fig.subplots_adjust(left=.03,right=.91,top=.85,bottom=.13,wspace=.15)
    fig.text(.5,.025,'FL1-fPINN is single-stage; its final reconstruction is shown only on the final-stage plane.',ha='center',fontsize=12)
    if next(iter(smoke)):fig.text(.5,-.02,'SMOKE CHECK: NOT A TRAINED RESULT',ha='center',fontsize=13)
    save(fig,out/'Figure_11_revise')
    
    fig,axes=plt.subplots(1,len(counts),figsize=(FIGURE_WIDTH,FINAL_FIGURE_HEIGHT),squeeze=False)
    for col,count in enumerate(counts):
        ax=axes[0,col];ax.plot(x,truth,color=COLORS['truth'],ls=':',lw=2)
        data=None
        for method in METHODS:
            rec=records.get((count,method))
            if rec is None:continue
            state,_=load_for_prediction(rec[0]/'final.pt');ax.plot(x,potential_numpy(state,x),color=COLORS[method],lw=LINE_WIDTH);data=rec[2]
        if data:ax.scatter(data['eta_x'],data['eta_target'],c=COLORS['supervised'],edgecolors='black',s=SUPERVISED_SIZE,zorder=5,clip_on=False)
        ax.set(xlabel='$x$',ylabel=r'$\eta(x)$',xlim=(0,np.pi));ax.set_title('(%s) %d supervised points'%(chr(97+col),count))
        ax.set_xticks([0,np.pi/2,np.pi],['0',r'$\pi/2$',r'$\pi$'])
        ax.tick_params(direction='in',top=True,right=True,labelleft=True,labelbottom=True);ax.grid(alpha=.16)
    fig.legend(handles=handles,loc='upper center',ncol=len(handles),fontsize=LEGEND_SIZE,bbox_to_anchor=(.5,1.08))
    fig.subplots_adjust(left=.08,right=.98,top=.81,bottom=.18,wspace=.3)
    if next(iter(smoke)):fig.text(.5,.01,'SMOKE CHECK: NOT A TRAINED RESULT',ha='center',fontsize=12)
    save(fig,out/'potential_final_comparison')
    (out/'plot_metadata.json').write_text(json.dumps(dict(problem=problem,seed=seed,supervision=counts,font_file=font,
             smoke=next(iter(smoke)),baseline_stage_curves_invented=False),indent=2)+'\n')
    print('Saved figures to '+str(out),flush=True);return out


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--root',default='runs/I02_gradient_v2');q.add_argument('--problem',choices=['zero_source','manufactured_series'],default='manufactured_series')
    q.add_argument('--supervision',default='3,2');q.add_argument('--seed',type=int,default=1234);q.add_argument('--font-dir');q.add_argument('--font-family')
    a=q.parse_args(argv);plot(a.root,a.problem,[int(n) for n in a.supervision.split(',')],a.seed,a.font_dir,a.font_family)


if __name__=='__main__':main()
