"""EMF self-control proof of concept: five-unit recurrent IAC network.

One fixed, rounded synaptic weight matrix reproduces (a) slower healthy
choices at low versus high restraint, and (b) smooth mouse paths initially
pulled toward temptation. The RT conversion is *calibrated* to approximate
means reported in Suri & Paap (2024) from Masterson et al. (2019), not a
prediction or independent fit. Stillman mouse paths are model-generated
qualitative analogues, NOT recovered trial-level observations.

Run: python EMF_SelfControl_Simulation.py
Dependencies: numpy, pandas, matplotlib.
"""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
NAMES=['Tasty Option','Healthy Option','Health Goal','Approach Tasty','Approach Healthy']
TI,HI,G,AT,AH=range(5)

@dataclass(frozen=True)
class Settings:
    tasty_to_approach: float=1.0
    healthy_to_approach: float=0.9
    healthy_goal_reciprocal: float=0.9
    approach_inhibition: float=-0.7
    leak: float=0.1
    dt: float=0.1
    cycles: int=1200
    criterion: float=0.10  # absolute advantage of one Approach unit over the other
    mouse_gain: float=0.15 # readout: horizontal displacement per model cycle per activation unit
    tasty_input: float=0.25
    healthy_input: float=0.20
    high_goal_input: float=0.25
    low_goal_input: float=0.00


def weights(p):
    W=np.zeros((5,5))   # W[to,from]
    W[AT,TI]=p.tasty_to_approach
    W[AH,HI]=p.healthy_to_approach
    W[HI,G]=W[G,HI]=p.healthy_goal_reciprocal
    W[AT,AH]=W[AH,AT]=p.approach_inhibition
    return W


def run(p,external,initial=None):
    W=weights(p)
    a=np.zeros(5) if initial is None else np.array(initial,dtype=float)
    hist=[]
    for cycle in range(p.cycles):
        net=W@a+np.array(external,dtype=float)
        # Same bounded, leaky IAC rule as sodium need Simulation 1.
        da=(1-a)*np.maximum(net,0)+a*np.minimum(net,0)-p.leak*a
        a=np.clip(a+p.dt*da,0,1)
        hist.append([cycle+1,*a,a[AH]-a[AT]])
    cols=['cycle',*['a_'+n.replace(' ','_') for n in NAMES],'D_healthy_minus_tasty']
    return pd.DataFrame(hist,columns=cols)


def decision(df,criterion):
    d=df['D_healthy_minus_tasty'].to_numpy()
    crossings=np.where(np.abs(d)>=criterion)[0]
    if not len(crossings): return None,None
    idx=int(crossings[0]);return idx+1,('Healthy' if d[idx]>0 else 'Tasty')


def fixed_condition(p,label,health_input,goal_input):
    external=[p.tasty_input,health_input,goal_input,0,0]
    df=run(p,external)
    cyc,choice=decision(df,p.criterion)
    if cyc is None: raise RuntimeError('Undecided: '+label)
    df.insert(0,'Condition',label)
    return df,dict(Condition=label,Health_input=health_input,Goal_input=goal_input,
                   Decision_cycles=cyc,Choice=choice,
                   First_differential=float(df['D_healthy_minus_tasty'].iloc[0]),
                   Most_tasty_biased_differential=float(df['D_healthy_minus_tasty'].min()),
                   Final_differential=float(df['D_healthy_minus_tasty'].iloc[-1]))


def movement(p,df):
    # Exactly the previously specified readout: dx/dcycle = k*(a_healthy-a_tasty).
    d=df['D_healthy_minus_tasty'].to_numpy()
    x=np.cumsum(p.mouse_gain*d)
    arrived=np.where(x>=1.0)[0]
    if not len(arrived): raise RuntimeError('Mouse never reached healthy choice')
    ix=int(arrived[0])
    # For comparing shapes, mouse trajectories conventionally normalized
    # to common endpoints; the *same* movement gain is used for both groups.
    y=np.linspace(0,1,ix+2)
    x=np.r_[0,x[:ix+1]]
    x[-1]=1.0
    return pd.DataFrame({'time_normalized':y,'mouse_x':x,
                         'mouse_y_normalized':y,
                         'cycle':np.arange(ix+2)})


def main():
    p=Settings()
    scenarios=[('Tasty choice',0,0),
               ('Healthy / higher restraint',p.healthy_input,p.high_goal_input),
               ('Healthy / lower restraint',p.healthy_input,p.low_goal_input)]
    trs=[];records=[]
    for condition,h,g in scenarios:
        df,row=fixed_condition(p,condition,h,g)
        trs.append(df);records.append(row)
    res=pd.DataFrame(records)
    assert res.Choice.tolist()==['Tasty','Healthy','Healthy'],res[['Condition','Choice']]
    assert res.Decision_cycles.iloc[0]<res.Decision_cycles.iloc[1]<res.Decision_cycles.iloc[2]
    # Approximate descriptive points reported by Suri & Paap (2024), citing
    # Masterson et al. (2019): Tasty 1300, healthy high 1330, healthy low 1400.
    targets=np.array([1300.,1330.,1400.])
    X=np.column_stack((np.ones(len(res)),res.Decision_cycles.to_numpy()))
    intercept,ms_per_cycle=np.linalg.lstsq(X,targets,rcond=None)[0]
    res['RT_empirical_approx_ms']=targets
    res['RT_model_calibrated_ms']=X@np.array([intercept,ms_per_cycle])
    res['RT_fit_residual_ms']=res['RT_model_calibrated_ms']-targets
    res.to_csv(HERE/'EMF_SelfControl_Results.csv',index=False,float_format='%.6f')
    pd.concat(trs,ignore_index=True).to_csv(HERE/'EMF_SelfControl_Activation_Traces.csv',index=False,float_format='%.7f')
    rows=[]
    W=weights(p)
    for i,recipient in enumerate(NAMES):
        for j,source in enumerate(NAMES):
            if W[i,j]!=0:
                rows.append({'From':source,'To':recipient,'Weight':W[i,j]})
    pd.DataFrame(rows).to_csv(HERE/'EMF_SelfControl_Weights.csv',index=False)
    pd.DataFrame([{'Parameter':k,'Value':v} for k,v in vars(p).items()]+[
        {'Parameter':'RT_readout_intercept_ms','Value':intercept},
        {'Parameter':'RT_readout_ms_per_cycle','Value':ms_per_cycle}
    ]).to_csv(HERE/'EMF_SelfControl_Settings.csv',index=False)

    paths={}
    for idx,label in [(1,'Higher restraint'),(2,'Lower restraint')]:
        paths[label]=movement(p,trs[idx])
        paths[label].assign(Condition=label).to_csv(HERE/(label.replace(' ','_')+'_Mouse_Path.csv'),index=False)

    plt.rcParams.update({'font.size':11, 'font.family':'DejaVu Sans'})
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.1,4.7),gridspec_kw={'width_ratios':[1,1.05]})
    colors=['#4a4a4a','#12807c','#cd7344']
    xx=np.arange(3)
    ax1.scatter(xx-.09,res['RT_empirical_approx_ms'],marker='o',s=60,c='#282828',label='Approx. empirical',zorder=3)
    ax1.scatter(xx+.09,res['RT_model_calibrated_ms'],marker='s',s=55,c='#ca693a',label='EMF (calibrated)',zorder=3)
    ax1.set_xticks(xx,['Tasty','Healthy\nhigh restraint','Healthy\nlow restraint'])
    ax1.set_ylabel('Choice latency (ms)')
    ax1.set_ylim(1280,1420);ax1.set_yticks(np.arange(1300,1420,25))
    ax1.set_title('Masterson: Choice latency',weight='bold',pad=13)
    ax1.spines[['top','right']].set_visible(False)
    ax1.legend(frameon=False,loc='upper left',fontsize=9)
    for name,color in [('Higher restraint','#12807c'),('Lower restraint','#cd7344')]:
        path=paths[name]
        ax2.plot(path.mouse_x,path.mouse_y_normalized,color=color,linewidth=2.4,label=name)
    ax2.scatter([0,1,-1],[0,1,1],color=['#222','#12807c','#aaa'],s=[18,35,28],zorder=4)
    ax2.text(-1,1.05,'Tasty',ha='center',fontsize=9)
    ax2.text(1,1.05,'Healthy',ha='center',fontsize=9)
    ax2.text(0,-.08,'Start',ha='center',fontsize=9)
    ax2.set_ylim(-.09,1.13);ax2.set_xlim(-1.15,1.25);ax2.set_aspect(1.35)
    ax2.set_xlabel('Horizontal mouse displacement (normalized)')
    ax2.set_ylabel('Normalized movement progress')
    ax2.set_title('Stillman: Model-generated mouse paths',weight='bold',pad=13)
    ax2.spines[['top','right']].set_visible(False)
    ax2.legend(frameon=False,loc='lower right',fontsize=9)
    fig.suptitle('Self-control: A shared five-unit IAC network',fontweight='bold',y=1.04,fontsize=15)
    fig.tight_layout()
    fig.savefig(HERE/'EMF_SelfControl_Panel_D.png',dpi=240,bbox_inches='tight')
    fig.savefig(HERE/'EMF_SelfControl_Panel_D.svg',bbox_inches='tight')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.2,4.5))
    for name,df,color in [('Healthy / higher restraint',trs[1],'#12807c'),('Healthy / lower restraint',trs[2],'#cd7344')]:
        cut=180
        ax.plot(df.cycle.iloc[:cut],df.D_healthy_minus_tasty.iloc[:cut],label=name,color=color,lw=2.5)
    ax.axhline(0,color='#777',lw=.85)
    ax.axhline(p.criterion,color='#aaa',ls='--',lw=.9,label='Healthy decision threshold')
    ax.set_xlabel('Network update cycle');ax.set_ylabel('Healthy Approach − Tasty Approach')
    ax.set_title('Goal feedback gradually reverses the response preference',weight='bold')
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE/'EMF_SelfControl_Differentials.png',dpi=240)
    plt.close(fig)
    print('\n',res.to_string(index=False))
    print('\nMouse dynamics:')
    for label,path in paths.items(): print(label,'n cycles',path.cycle.max(),'minimum x',round(path.mouse_x.min(),3))
    print('Fit:',round(intercept,3),'+',round(ms_per_cycle,4),'ms/network cycle')

if __name__=='__main__': main()
