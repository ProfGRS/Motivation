"""EMF habit simulation, revised against Tricomi et al. (2009) Figure 2.

Empirical means are VISUAL APPROXIMATIONS from user-supplied published plot,
NOT published numeric tables nor trial-level observations.

Network: 7 units, M&M/Frito cue, Approach, Press; Frito-specific Enough detector.
One Hebbian episode abstracts one 8-minute training session. Two vs 12 episodes.
One fixed weight matrix and satiation strength for all tests.

Single behavioral scale K (responses/second per unit Press activation) is
calibrated only to the average of FOUR PRE-satiation empirical figure values.
No post-satiation data are used in calibrating this readout. Network parameters
were chosen previously to reproduce a qualitative devaluation interaction;
thus this remains a mechanistic demonstration, not an independent validation.

Run from anywhere: python EMF_Habit_Revised_Simulation.py
Dependencies: numpy, pandas, matplotlib.
"""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
NAMES = ('M&M Cue','Frito Cue','Approach M&M','Approach Frito','Press M&M','Press Frito','Enough (Frito)')
MC, FC, MA, FA, MP, FP, SAT = range(7)

@dataclass(frozen=True)
class Parameters:
    cue_to_approach: float = 1.0
    approach_to_press: float = 1.0
    enough_to_approach_frito: float = -1.0
    learning_rate: float = 0.05
    learned_ceiling: float = 1.0
    leak: float = 0.1
    dt: float = 0.1
    settling_updates: int = 400
    limited_episodes: int = 2
    extended_episodes: int = 12

# Figure 2 visual approximations; all rates are responses/second.
# 'Frito' is simply the illustrative food subsequently consumed to satiety;
# the experiment counterbalanced which of M&Ms/Fritos was satiated.
EMPIRICAL = [
    (2,'Before satiation','Unsatiated food',2.85),
    (2,'Before satiation','Food subsequently satiated',2.40),
    (12,'Before satiation','Unsatiated food',2.30),
    (12,'Before satiation','Food subsequently satiated',2.05),
    (2,'After satiation','Unsatiated food',3.05),
    (2,'After satiation','Satiated food',1.20),
    (12,'After satiation','Unsatiated food',2.60),
    (12,'After satiation','Satiated food',2.40),
]

def weights_initial(p):
    W=np.zeros((7,7)) # W[receiver, sender]
    for c,a,pr in [(MC,MA,MP),(FC,FA,FP)]:
        W[a,c]=p.cue_to_approach
        W[pr,a]=p.approach_to_press
    W[FA,SAT]=p.enough_to_approach_frito
    return W

def settle(p,W,cue,enough=False,trace=False):
    a=np.zeros(7)
    a[cue]=1.0
    a[SAT]=float(enough)
    free=[MA,FA,MP,FP]
    trajectory=[]
    for t in range(p.settling_updates):
        net=W@a
        delta=(1-a)*np.maximum(net,0) + a*np.minimum(net,0) - p.leak*a
        a[free]=np.clip(a[free]+p.dt*delta[free],0,1)
        if trace:
            trajectory.append({'update':t+1,'cue':NAMES[cue], 'enough_active':int(enough),
                               'Approach_M&M':a[MA], 'Approach_Frito':a[FA],
                               'Press_M&M':a[MP], 'Press_Frito':a[FP]})
    return (a,trajectory) if trace else a

def train(p):
    W=weights_initial(p)
    matrices={0:W.copy()}
    history=[]
    for episode in range(1,p.extended_episodes+1):
        for cue,press in [(MC,MP),(FC,FP)]:
            a=settle(p,W,cue,enough=False)
            w=W[press,cue]
            W[press,cue]=w+p.learning_rate*a[cue]*a[press]*(p.learned_ceiling-w)
        matrices[episode]=W.copy()
        history.append({'episode':episode,'cue_to_press_weight':W[FP,FC],
                        'mean_valued_press_activation':settle(p,W,FC,False)[FP],
                        'satiated_press_activation':settle(p,W,FC,True)[FP]})
    return matrices,pd.DataFrame(history)

def run():
    p=Parameters()
    mats, history=train(p)
    # Calibrate ONLY from the four empirical BEFORE-satiation means.
    # Pre-satiation model predicts same activation for both foods, within a group.
    pre_emp=np.array([r[3] for r in EMPIRICAL if r[1]=='Before satiation'])
    pre_model=np.array([settle(p,mats[2],MC,False)[MP]]*2 + [settle(p,mats[12],MC,False)[MP]]*2)
    scale=pre_emp.mean()/pre_model.mean()  # 2.40 / ~0.92
    data=[]; traces=[]
    for sessions,phase,food,emp in EMPIRICAL:
        cue,press,approach = ((MC,MP,MA) if food=='Unsatiated food' else (FC,FP,FA))
        enough=(phase=='After satiation' and cue==FC)
        a,trace=settle(p,mats[sessions],cue,enough,True)
        for row in trace: row.update({'sessions':sessions,'phase':phase,'food':food})
        traces+=trace
        data.append({'training_sessions':sessions,'training':'Limited (1 day)' if sessions==2 else 'Extended (3 days)',
                     'phase':phase,'food':food,'empirical_estimate_responses_per_s':emp,
                     'predicted_responses_per_s':scale*a[press],
                     'empirical_estimate_responses_per_min':emp*60,
                     'predicted_responses_per_min':scale*a[press]*60,
                     'Press_activation':a[press], 'Approach_activation':a[approach],
                     'cue_to_press_weight':mats[sessions][press,cue],
                     'estimated_scale_responses_per_s_at_unit_activation':scale,
                     'enough_state_active':int(enough)})
    df=pd.DataFrame(data)
    df.to_csv(OUT/'EMF_Habit_Empirical_vs_Model.csv',index=False,float_format='%.8f')
    pd.DataFrame([{'training_sessions':n,'phase':phase,'food':food,'empirical_responses_per_s_approx':rate,
                   'source':'Tricomi et al. (2009), Fig 2 (visual estimates)'} for n,phase,food,rate in EMPIRICAL]).to_csv(
                   OUT/'Tricomi_Figure2_Approx_Empirical_Data.csv',index=False)
    pd.DataFrame(traces).to_csv(OUT/'EMF_Habit_Activation_Traces.csv',index=False,float_format='%.8f')
    history.to_csv(OUT/'EMF_Habit_Learning_Curve.csv',index=False,float_format='%.8f')
    w_rows=[]
    for ti,name in enumerate(NAMES):
        for sj,src in enumerate(NAMES):
            if any(abs(mats[t][ti,sj])>1e-10 for t in [2,12]):
                w_rows.append({'source':src,'target':name,'initial_weight':weights_initial(p)[ti,sj],
                               'limited_weight':mats[2][ti,sj], 'extended_weight':mats[12][ti,sj]})
    pd.DataFrame(w_rows).to_csv(OUT/'EMF_Habit_Network_Weights.csv',index=False,float_format='%.8f')
    settings=[{'parameter':k,'value':v} for k,v in vars(p).items()]
    settings += [{'parameter':'single_linear_scale_responses_s_at_unit_activation','value':scale},
                 {'parameter':'scale_calibration','value':'pooled pre-satiation empirical grand mean only'},
                 {'parameter':'empirical_values','value':'visual approximations from original Figure 2, not tabulated published values'}]
    pd.DataFrame(settings).to_csv(OUT/'EMF_Habit_Settings.csv',index=False)
    controls=[]
    for sessions in [2,12]:
        W=mats[sessions].copy()
        nohab=W.copy();nohab[FP,FC]=0.
        for label,matrix,det in [('normal',W,False),('satiated',W,True),('satiated, direct cue-to-press removed',nohab,True)]:
            aa=settle(p,matrix,FC,det)
            controls.append({'sessions':sessions,'condition':label,'Approach_Frito':aa[FA],
                             'Press_Frito':aa[FP],'predicted_responses_per_s':scale*aa[FP]})
    pd.DataFrame(controls).to_csv(OUT/'EMF_Habit_Diagnostic_Controls.csv',index=False,float_format='%.8f')
    # Tests: empirical phases separate; learned shortcuts symmetric; equal satiation effects on approach;
    # before satiation nearly equal training groups; reduced devaluation sensitivity after 12 sessions.
    get=lambda n,phase,food:df[(df.training_sessions==n)&(df.phase==phase)&(df.food==food)].iloc[0]
    lim_v=get(2,'After satiation','Unsatiated food').predicted_responses_per_s
    lim_d=get(2,'After satiation','Satiated food').predicted_responses_per_s
    ext_v=get(12,'After satiation','Unsatiated food').predicted_responses_per_s
    ext_d=get(12,'After satiation','Satiated food').predicted_responses_per_s
    assert 0.4<(lim_v-lim_d)/lim_v<0.6
    assert 0<(ext_v-ext_d)/ext_v<0.2
    assert abs(ext_v-lim_v)/lim_v<0.05
    assert get(2,'After satiation','Satiated food').Approach_activation==0
    assert get(12,'After satiation','Satiated food').Approach_activation==0
    assert np.max(np.abs(mats[12]))<=1.0+1e-8
    assert all(not np.any(mats[n][FP,MC]!=0) for n in [2,12])

    # Plot small-multiples with truly common empirical/model scale.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig, axs=plt.subplots(2,2,figsize=(11.3,7.2),sharey=True)
    colors={2:'#a1346b',12:'#3575aa'}
    for ri,(src,col) in enumerate([('empirical_estimate_responses_per_s','Empirical (approx.)'),
                                   ('predicted_responses_per_s','EMF simulation')]):
        for ci,phase in enumerate(['Before satiation','After satiation']):
            ax=axs[ri,ci]; d=df[df.phase==phase]
            for sessions in [2,12]:
                q=d[d.training_sessions==sessions]
                ax.plot([0,1],q[src],marker='D' if sessions==2 else 's',lw=2.3,ms=7,
                        color=colors[sessions],linestyle='--' if sessions==2 else '-',
                        label='1 day' if sessions==2 else '3 days')
            ax.set_xticks([0,1], ['Unsatiated\nfood','Satiated\nfood' if ci else 'Later-satiated\nfood'])
            ax.set_xlim(-.26,1.26);ax.set_ylim(0,4.1)
            ax.grid(axis='y',alpha=.12)
            ax.spines[['top','right']].set_visible(False)
            if ri==0:ax.set_title(phase,fontsize=13,weight='semibold')
            if ci==0:ax.set_ylabel(f'{col}\nResponses / second',fontsize=10.5)
            if ri==0 and ci==1:ax.legend(frameon=False,loc='upper right')
    fig.suptitle('Habit: published measurements and network predictions',fontsize=15,weight='semibold',y=.98)
    fig.text(.50,.015,'Empirical points visually estimated from Tricomi et al. (2009), Fig. 2; no individual data used.',ha='center',fontsize=9,color='#555')
    fig.tight_layout(rect=[0,.035,1,.95],h_pad=2.5,w_pad=2.)
    fig.savefig(OUT/'EMF_Habit_Figure5f_Comparison.png',dpi=220)
    plt.close(fig)
    print(f'Pre-satiation calibration: observed mean {pre_emp.mean():.3f} responses/s; '
          f'model mean {pre_model.mean():.4f}; K = {scale:.5f} responses/s')
    print(df[['training','phase','food','empirical_estimate_responses_per_s',
              'predicted_responses_per_s']].to_string(index=False))
    print('RMSE across 8 figure points:', np.sqrt(np.mean((df.empirical_estimate_responses_per_s-df.predicted_responses_per_s)**2)))
    print('Predicted post-satiation reduction: 1-day',100*(lim_v-lim_d)/lim_v,
          '%; 3-day',100*(ext_v-ext_d)/ext_v,'%')

if __name__=='__main__':run()
