"Plot two-dimensional potential reconstructions and sparse observations."
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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text
from plot_style import configure_typography,enforce_figure_fonts,FONT_SIZES
from problem import eta_truth
from protocol import METHODS
from train_potential2d import load_for_prediction,potential_numpy


FIGURE_WIDTH=13.4       
FIGURE_HEIGHT=14.2      
FINAL_FIGURE_HEIGHT=7.2 
AXIS_LABEL_SIZE=20     
AXIS_TICK_SIZE=16      
TITLE_SIZE=19          
GROUP_TITLE_SIZE=21    
COLORBAR_LABEL_SIZE=20 
COLORBAR_TICK_SIZE=17  
HORIZONTAL_SPACE=.38   
VERTICAL_SPACE=.48     
GROUP_SPACE=.52        
COLORBAR_GAP=.012      
COLORBAR_WIDTH=.018    
SUPERVISED_SIZE=40     
GRID_NODES=101         
COLORMAP='RdBu_r'      
PNG_DPI=300            


def image_panel(ax,field,xy,title,norm):
    axis=np.linspace(0,1,GRID_NODES)
    artist=ax.pcolormesh(axis,axis,field.T,cmap=COLORMAP,norm=norm,shading='auto',rasterized=True)
    ax.scatter(xy[:,0],xy[:,1],s=SUPERVISED_SIZE,c='#e3b52c',edgecolors='white',linewidths=1.6,zorder=4,clip_on=False)
    ax.scatter(xy[:,0],xy[:,1],s=SUPERVISED_SIZE*.62,c='#e3b52c',edgecolors='#333333',linewidths=.5,zorder=5,clip_on=False)
    ax.set(xlabel='$x$',ylabel='$y$',xlim=(0,1),ylim=(0,1),aspect='equal')
    ax.set_xticks([0,.5,1],['0','0.5','1']);ax.set_yticks([0,.5,1],['0','0.5','1'])
    ax.tick_params(direction='in',top=True,right=True,labelleft=True,labelbottom=True)
    ax.set_title(title,fontsize=TITLE_SIZE,pad=8)
    return artist


def save(fig,path):
    enforce_figure_fonts(fig,Text)
    for ext in ['pdf','png']:fig.savefig(path.with_suffix('.'+ext),dpi=PNG_DPI,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def plot(root,counts=(9,4),seed=1234,font_dir=None,font_family=None):
    torch.set_num_threads(1);root=Path(root);counts=list(counts);records={};smoke=set();panels={}
    axis=np.linspace(0,1,GRID_NODES);xx,yy=np.meshgrid(axis,axis,indexing='ij');evaluation=np.column_stack([xx.ravel(),yy.ravel()])
    truth=eta_truth(evaluation).reshape(GRID_NODES,GRID_NODES);all_fields=[truth]
    for count in counts:
        identity=None;rows=[];supervised=None
        for method in METHODS:
            run=root/('eta_'+str(count))/method/('seed_'+str(seed))
            if not (run/'summary.json').exists():continue
            config=json.loads((run/'config.json').read_text());s=json.loads((run/'summary.json').read_text())
            if config['args']['supervised_points']!=count:raise ValueError('Directory/settings mismatch.')
            with np.load(run/'observations.npz',allow_pickle=False) as data:
                h=hashlib.sha256(b''.join(data[k].tobytes() for k in ['obs_tx','obs_target','eta_xy','eta_target'])).hexdigest()
                xy=data['eta_xy'].copy()
            if identity is not None and identity!=h:raise ValueError('Methods do not share the same observations.')
            identity=h;supervised=xy;records[count,method]=(run,s);smoke.add(config['args']['smoke'])
        go=records.get((count,'go_multi'));baseline=records.get((count,'fl1'))
        if go:
            for stage in go[1]['stages']:
                sid=stage['stage'];state,_=load_for_prediction(go[0]/('stage_%02d_best_loss.pt'%sid))
                field=potential_numpy(state,evaluation).reshape(GRID_NODES,GRID_NODES)
                rows.append(('GO-fPINNs S%d'%sid,field));all_fields.append(field)
        if baseline:
            state,_=load_for_prediction(baseline[0]/'final.pt');field=potential_numpy(state,evaluation).reshape(GRID_NODES,GRID_NODES)
            rows.append(('FL1-fPINN (final)',field));all_fields.append(field)
        if supervised is not None:rows.append((r'True $\eta(x,y)$',truth));panels[count]=(rows,supervised)
    if not records:raise FileNotFoundError('No completed selected runs.')
    if len(smoke)!=1:raise ValueError('Cannot mix formal/smoke curves.')
    if any(n not in panels for n in counts):raise FileNotFoundError('Missing all selected methods for a supervision setting.')
    if not all(np.isfinite(f).all() for f in all_fields):raise ValueError('Nonfinite potential field.')
    low=min(float(f.min()) for f in all_fields);high=max(float(f.max()) for f in all_fields)
    if high-low<1e-12:high=low+1e-12
    norm=Normalize(low,high)
    FONT_SIZES.update(base=18,axis_label=AXIS_LABEL_SIZE,axis_tick=AXIS_TICK_SIZE,column_title=TITLE_SIZE)
    font=configure_typography(plt,font_manager,font_dir,font_family)
    out=root/'figures'/('seed_'+str(seed));out.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(FIGURE_WIDTH,FIGURE_HEIGHT if len(counts)==2 else FIGURE_HEIGHT/2))
    outer=fig.add_gridspec(len(counts),2,width_ratios=[1,.035],left=.07,right=.93,top=.88,bottom=.065,
                          hspace=GROUP_SPACE,wspace=.04)
    for group,count in enumerate(counts):
        rows,xy=panels[count];ncols=3 if len(rows)>=3 else 2;nrows=int(np.ceil(len(rows)/ncols))
        sub=outer[group,0].subgridspec(nrows,ncols,wspace=HORIZONTAL_SPACE,hspace=VERTICAL_SPACE)
        axes=[]
        for j,(title,field) in enumerate(rows):
            ax=fig.add_subplot(sub[j//ncols,j%ncols]);image_panel(ax,field,xy,title,norm);axes.append(ax)
        positions=[ax.get_position() for ax in axes]
        cax=fig.add_subplot(outer[group,1]);bottom=min(p.y0 for p in positions);top=max(p.y1 for p in positions)
        cax.set_position([max(p.x1 for p in positions)+COLORBAR_GAP,bottom,COLORBAR_WIDTH,top-bottom])
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=COLORMAP),cax=cax)
        bar.set_label(r'$\eta(x,y)$',size=COLORBAR_LABEL_SIZE);bar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
        pos=outer[group,0].get_position(fig)
        fig.text(pos.x0,pos.y1+.048,'(%s) %d supervised potential points'%(chr(97+group),count),fontsize=GROUP_TITLE_SIZE)
    handle=Line2D([],[],ls='none',marker='o',mfc='#e3b52c',mec='#333333',label='Supervised points')
    fig.legend(handles=[handle],loc='upper center',bbox_to_anchor=(.5,1.005),fontsize=17)
    if next(iter(smoke)):fig.text(.5,-.028,'SMOKE CHECK: NOT A TRAINED RESULT',ha='center',fontsize=13)
    save(fig,out/'Figure_12_revise')
    
    fig=plt.figure(figsize=(FIGURE_WIDTH,FINAL_FIGURE_HEIGHT if len(counts)==2 else FINAL_FIGURE_HEIGHT/2+.8))
    grid=fig.add_gridspec(len(counts),4,width_ratios=[1,1,1,.035],left=.07,right=.93,top=.87,bottom=.13,
                         wspace=HORIZONTAL_SPACE,hspace=.7)
    for group,count in enumerate(counts):
        rows,xy=panels[count];lookup=dict(rows)
        final_titles=[t for t in lookup if t.startswith('GO-fPINNs')]
        specs=[]
        if 'FL1-fPINN (final)' in lookup:specs.append(('FL1-fPINN (final)',lookup['FL1-fPINN (final)']))
        if final_titles:specs.append(('GO-fPINNs (final)',lookup[final_titles[-1]]))
        specs.append((r'True $\eta(x,y)$',truth))
        axes=[]
        for col,(title,field) in enumerate(specs):
            ax=fig.add_subplot(grid[group,col]);image_panel(ax,field,xy,title,norm);axes.append(ax)
        positions=[ax.get_position() for ax in axes];bottom=min(p.y0 for p in positions);top=max(p.y1 for p in positions)
        cax=fig.add_subplot(grid[group,3]);cax.set_position([max(p.x1 for p in positions)+COLORBAR_GAP,bottom,COLORBAR_WIDTH,top-bottom])
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=COLORMAP),cax=cax);bar.set_label(r'$\eta(x,y)$',size=COLORBAR_LABEL_SIZE)
        bar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    fig.legend(handles=[handle],loc='upper center',bbox_to_anchor=(.5,1.015),fontsize=17)
    fig.text(.5,.015,'Rows: '+', '.join('%d supervised points'%n for n in counts),ha='center',fontsize=13)
    if next(iter(smoke)):fig.text(.5,-.02,'SMOKE CHECK: NOT A TRAINED RESULT',ha='center',fontsize=13)
    save(fig,out/'potential2d_final_comparison')
    (out/'plot_metadata.json').write_text(json.dumps(dict(seed=seed,supervision=counts,font_file=font,
          smoke=next(iter(smoke)),common_color_limits=[low,high],color_scale='all displayed predictions and truth; no clipping',
          invented_FL1_stage_checkpoints=False,complete_axes_labels_and_ticks=True),indent=2)+'\n')
    print('Saved figures to '+str(out),flush=True);return out


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--root',default='runs/I03_gradient_v2');q.add_argument('--supervision',default='9,4')
    q.add_argument('--seed',type=int,default=1234);q.add_argument('--font-dir');q.add_argument('--font-family')
    a=q.parse_args(argv);plot(a.root,[int(n) for n in a.supervision.split(',')],a.seed,a.font_dir,a.font_family)


if __name__=='__main__':main()
