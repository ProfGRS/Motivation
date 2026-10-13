"""EMF / IAC sodium-depletion simulation for the eight-node user-specified network.

This is a mechanistic proof-of-principle, not an independently validated fit.
Empirical anchors are approximately read from Robinson & Berridge (2013),
Figure 2 as supplied by the author. No animal-level data, invented SEMs,
or stochastic perturbations are used.

No novelty parameter is included: the same input and weights are used for the
first and subsequent depleted CS-only trials. The direct CS+UCS condition
clamps the existing UCS_salt unit, without adding a node.

Running this script creates outputs in the same directory.
Dependencies: numpy, scipy, pandas, matplotlib.
"""
from pathlib import Path
from dataclasses import dataclass, replace
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent
NAMES = ('CS_sucrose', 'UCS_sucrose', 'CS_salt', 'UCS_salt',
         'Sodium_need', 'NM', 'Approach', 'Avoid')
IDX = {v:i for i,v in enumerate(NAMES)}

@dataclass(frozen=True)
class Params:
    # Fixed arrow weights defined in the diagram. All in relative activation units.
    cs_ucs: float = 0.80                # both directions for both CS/UCS pairs
    ucs_strong: float = 1.00            # sucrose->Approach and salt->Avoid
    ucs_weak: float = 0.10              # sucrose->Avoid and salt->Approach
    mutual_inhibition: float = -1.00   # both directions between action units
    sodium_to_nm: float = 1.00         # Na-state detector->NM
    nm_gain_coefficient: float = 9.30  # modulation, NOT an additive synaptic weight
    theta_approach: float = 0.67       # action input thresholds, not connections
    theta_avoid: float = 1.325
    decay: float = 0.10                # all nonclamped units
    dt: float = 0.10                   # simulation integration step
    settling_cycles: int = 500
    observation_scale: float = 5.00    # assumed conversion from activation balance to action counts
    # separate learning option after CS-UCS reinforcement
    plasticity_rate: float = 0.10
    plasticity_ceiling: float = 1.00


def weight_matrix(p):
    """Matrix w_ij for incoming activity to i from j."""
    W = np.zeros((len(NAMES),len(NAMES)))
    def link(a,b,w,bidir=False):
        W[IDX[b],IDX[a]] = w
        if bidir: W[IDX[a],IDX[b]] = w
    # CS/UCS links begin at zero and are learned through cue-outcome pairings.
    link('Sodium_need','NM',p.sodium_to_nm)
    link('UCS_sucrose','Approach',p.ucs_strong)
    link('UCS_sucrose','Avoid',p.ucs_weak)
    link('UCS_salt','Approach',p.ucs_weak)
    link('UCS_salt','Avoid',p.ucs_strong)
    link('Approach','Avoid',p.mutual_inhibition,True)
    return W


def train_cue_associations(p,W,pairings=80,learning_rate=0.15):
    """Simple Hebbian coactivation learning of reciprocal CS-UCS associations.

    During each training trial, the cue and its outcome are both clamped on;
    w <- w + eta * CS * UCS * (w_max - w). This is the SAME rule for salt
    and sucrose and approaches the 0.8 medium-strength connection.
    """
    W=W.copy()
    records=[]
    for t in range(pairings):
        for cue,outcome in [('CS_sucrose','UCS_sucrose'),('CS_salt','UCS_salt')]:
            i=IDX[cue];j=IDX[outcome]
            W[i,j] += learning_rate*(p.cs_ucs-W[i,j])
            W[j,i] += learning_rate*(p.cs_ucs-W[j,i])
        records.append({'training_pairing':t+1,
                       'CSsalt_to_UCSsalt':W[IDX['UCS_salt'],IDX['CS_salt']],
                       'UCSsalt_to_CSsalt':W[IDX['CS_salt'],IDX['UCS_salt']],
                       'CSsucrose_to_UCSsucrose':W[IDX['UCS_sucrose'],IDX['CS_sucrose']],
                       'UCSsucrose_to_CSsucrose':W[IDX['CS_sucrose'],IDX['UCS_sucrose']]})
    return W,records

def settle(p,W,cue,need,reinforce_salt=False,modulation=True,save_trace=False):
    """Bounded, leaky interactive activation with mutual inhibition.

    Da = dt * [(1-a)*max(net,0) + a*min(net,0) - decay*a].
    Inputs CS and sodium need are externally clamped at 1.0.

    NM modulates only the salt-UCS contribution to the Approach unit:
      input_A = [W @ a]_A - theta_A
                + gain * a_NM * W[A, UCS_salt] * a_UCS_salt
    This is a synapse-/input-selective modulation of Approach gain, shown
    by the square-ended connection in the author's diagram.
    """
    n = len(NAMES)
    a = np.zeros(n)
    clamped = np.zeros(n,dtype=bool)
    if cue == 'salt':
        a[IDX['CS_salt']]=1; clamped[IDX['CS_salt']]=True
    if cue == 'sucrose':
        a[IDX['CS_sucrose']]=1; clamped[IDX['CS_sucrose']]=True
    if need:
        a[IDX['Sodium_need']]=1; clamped[IDX['Sodium_need']]=True
    if reinforce_salt:
        a[IDX['UCS_salt']]=1; clamped[IDX['UCS_salt']]=True
    trace = []
    for t in range(p.settling_cycles):
        net = W @ a
        net[IDX['Approach']]-=p.theta_approach
        net[IDX['Avoid']]-=p.theta_avoid
        if modulation:
            net[IDX['Approach']] += (p.nm_gain_coefficient * a[IDX['NM']] *
                 W[IDX['Approach'],IDX['UCS_salt']]*a[IDX['UCS_salt']])
        da = (1-a)*np.maximum(net,0) + a*np.minimum(net,0) - p.decay*a
        a[~clamped] += p.dt*da[~clamped]
        a=np.clip(a,0,1)
        if save_trace and t % 2 == 0:
            trace.append({'cycle':t+1,'Approach':a[IDX['Approach']],
                          'Avoid':a[IDX['Avoid']], 'UCS_salt':a[IDX['UCS_salt']],
                          'NM':a[IDX['NM']], 'balance':p.observation_scale*(a[IDX['Approach']]-a[IDX['Avoid']])})
    vals = {k:float(a[IDX[k]]) for k in NAMES}
    vals['balance'] = p.observation_scale * (vals['Approach']-vals['Avoid'])
    if save_trace: return vals,trace
    return vals


def calibrate(p,W):
    """Three free parameters, three approximate figure landmarks.
    Uses thresholds (a fixed property of two action units) plus one gain.
    Fixed diagram-arrow weights are NOT fitted or changed.
    """
    th_a=brentq(lambda t: settle(replace(p,theta_approach=t),W,'sucrose',0)['balance']-4.35,0.0,1.0)
    p=replace(p,theta_approach=float(th_a))
    th_v=brentq(lambda t: settle(replace(p,theta_avoid=t),W,'salt',0)['balance']+0.40,0.80,0.90)
    p=replace(p,theta_avoid=float(th_v))
    gain=brentq(lambda t: settle(replace(p,nm_gain_coefficient=t),W,'salt',1)['balance']-3.85,3.0,9.0)
    return replace(p,nm_gain_coefficient=float(gain))


def learn_during_reinforcement(p,W,rate,episodes=10):
    """Optional standard NM-gated Hebbian strengthening AFTER first CS-only tests.
    Uses only the currently existing UCS_salt->Approach connection (no new nodes).
    """
    W=W.copy()
    log=[]
    for e in range(episodes):
        v=settle(p,W,'salt',1,reinforce_salt=True)
        m=v['NM'];pre=v['UCS_salt'];post=v['Approach']
        old=W[IDX['Approach'],IDX['UCS_salt']]
        W[IDX['Approach'],IDX['UCS_salt']] += rate*m*pre*post*(p.plasticity_ceiling-old)
        log.append({'episode':e+1,'salt_to_approach_weight':W[IDX['Approach'],IDX['UCS_salt']]})
    return W,log


def calibrate_later_learning(p,W):
    def residual(r):
        Wr,_=learn_during_reinforcement(p,W,r)
        return settle(p,Wr,'salt',0)['balance']-.6
    rate=brentq(residual,0.001,1.0)
    return float(rate),learn_during_reinforcement(p,W,rate)


def main():
    p0=Params()
    W,training_log=train_cue_associations(p0,weight_matrix(p0))
    pd.DataFrame(training_log).to_csv(OUT/'EMF_IAC_Sodium_CS_Training.csv',index=False,float_format='%.7f')
    W0=W.copy()
    p=calibrate(p0,W)
    conditions = [
       ('Day 11: normal','salt',0,False),
       ('Day 11: normal','sucrose',0,False),
       ('Day 13: first CS','salt',1,False),
       ('Day 13: CS 2–10','salt',1,False),  # deliberately unchanged; no novelty/attenuation term
       ('Day 13: CS 2–10','sucrose',1,False),
       ('Day 13: CS + UCS','salt',1,True),
       ('Day 18: repleted (unchanged weights)','salt',0,False),
       ('Sodium depleted; NM gain off','salt',1,False),
       ('Sodium depleted; neutral cue','neutral',1,False),
    ]
    rows=[]
    for label,cue,need,reinforce in conditions:
        vals=settle(p,W,cue,need,reinforce_salt=reinforce,
                    modulation=('gain off' not in label))
        rows.append({'Condition':label, 'Cue':cue, 'Need':need,
                     'Direct_salt_UCS':reinforce,
                     'Approach_activation':vals['Approach'],
                     'Avoid_activation':vals['Avoid'],
                     'NM_activation':vals['NM'],
                     'Action_balance_model':vals['balance']})
    assert np.array_equal(W,W0), 'First-responding simulations must hold weights fixed'
    assert rows[0]['Action_balance_model']<0 < rows[2]['Action_balance_model']
    assert abs(rows[1]['Action_balance_model']-rows[4]['Action_balance_model'])<1e-8
    assert abs(rows[6]['Action_balance_model']-rows[0]['Action_balance_model'])<1e-8
    assert rows[7]['Action_balance_model']<0
    assert abs(rows[8]['Action_balance_model'])<1e-12
    # Optional model of post-reinforcement repletion result.
    lr,(Wafter,learning_log)=calibrate_later_learning(p,W)
    final=settle(p,Wafter,'salt',0)
    rows.append({'Condition':'Day 18: repleted (post-reinforcement learning)',
                 'Cue':'salt','Need':0,'Direct_salt_UCS':False,
                 'Approach_activation':final['Approach'],'Avoid_activation':final['Avoid'],
                 'NM_activation':final['NM'],'Action_balance_model':final['balance']})
    pd.DataFrame(rows).to_csv(OUT/'EMF_IAC_Sodium_Results.csv',index=False,float_format='%.6f')
    pd.DataFrame(learning_log).to_csv(OUT/'EMF_IAC_Sodium_Optional_Learning.csv',index=False,float_format='%.6f')

    # Exact network-arrow weight manifest (modulatory gain listed separately).
    arrows=[]
    for receiver in NAMES:
        for sender in NAMES:
            w=W[IDX[receiver],IDX[sender]]
            if w:
                arrows.append({'Source':sender,'Target':receiver,'Weight':w,
                               'Kind':'Inhibitory' if w<0 else 'Excitatory'})
    arrows.append({'Source':'NM','Target':'Approach (salt-UCS input)',
                   'Weight':p.nm_gain_coefficient,'Kind':'Multiplicative gain coefficient (not additive)'})
    pd.DataFrame(arrows).to_csv(OUT/'EMF_IAC_Sodium_Arrow_Weights.csv',index=False,float_format='%.6f')

    metrics=[{'Parameter':k,'Value':v} for k,v in vars(p).items()]
    metrics.append({'Parameter':'fitted_postreinforcement_plasticity_rate','Value':lr})
    metrics.append({'Parameter':'salt_to_approach_weight_after_reinforcement','Value':Wafter[IDX['Approach'],IDX['UCS_salt']]})
    pd.DataFrame(metrics).to_csv(OUT/'EMF_IAC_Sodium_Settings.csv',index=False)

    normal,tn=settle(p,W,'salt',0,save_trace=True)
    depleted,td=settle(p,W,'salt',1,save_trace=True)
    saltdirect,tu=settle(p,W,'salt',1,reinforce_salt=True,save_trace=True)
    for trace,label in [(tn,'Normal'),(td,'Depleted'),(tu,'Depleted_plus_UCS')]:
        for row in trace: row['Condition']=label
    pd.DataFrame(tn+td+tu).to_csv(OUT/'EMF_IAC_Sodium_Activation_Traces.csv',index=False,float_format='%.6f')

    # Plot: comparison against visually estimated means only; no fabricated SEMs.
    names=['Normal salt','First salt CS','Later salt CS','CS + salt UCS','Repleted salt','Sucrose normal','Sucrose depleted']
    empirical=[-.40,3.85,2.90,3.15,.60,4.35,4.75]
    model=[rows[0]['Action_balance_model'],rows[2]['Action_balance_model'],
           rows[3]['Action_balance_model'],rows[5]['Action_balance_model'],
           rows[-1]['Action_balance_model'],rows[1]['Action_balance_model'],
           rows[4]['Action_balance_model']]
    fixed_model=[rows[0]['Action_balance_model'],rows[2]['Action_balance_model'],
           rows[3]['Action_balance_model'],rows[5]['Action_balance_model'],
           rows[6]['Action_balance_model'],rows[1]['Action_balance_model'],
           rows[4]['Action_balance_model']]
    x=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(10.6,4.7))
    ax.plot(x,empirical,'o-',label='Empirical means (visual estimates)')
    ax.plot(x,model,'s--',label='EMF IAC (optional learning after reinforcement)')
    ax.plot(x[4],fixed_model[4],marker='x',markersize=12,linestyle='None',label='Repletion prediction if weights never change')
    ax.axhline(0,color='0.55',lw=.8)
    ax.set_xticks(x,names,rotation=25,ha='right')
    ax.set_ylabel('Appetitive minus aversive actions per cue\n(empirical); scaled IAC activation balance (model)')
    ax.set_title('Sodium-need EMF simulation: comparison with published figure')
    ax.legend(loc='lower center',fontsize=8,frameon=False)
    fig.tight_layout()
    fig.savefig(OUT/'EMF_IAC_Sodium_Comparison.png',dpi=230)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,trial_result,title in [(axes[0],tn,'Normal'),(axes[1],td,'Sodium depleted')]:
        ax.plot([r['cycle'] for r in trial_result],[r['Approach'] for r in trial_result],label='Approach')
        ax.plot([r['cycle'] for r in trial_result],[r['Avoid'] for r in trial_result],label='Avoid')
        ax.set_xlabel('IAC update cycle')
        ax.set_ylabel('Activation (0–1)')
        ax.set_ylim(-.02,1)
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.suptitle('Same weights and salt cue, different sodium-need state')
    fig.tight_layout()
    fig.savefig(OUT/'EMF_IAC_Sodium_Trajectories.png',dpi=230)
    plt.close(fig)

    metrics_out={
      'thetaApproach':p.theta_approach,'thetaAvoid':p.theta_avoid,
      'NM_gain':p.nm_gain_coefficient,'NM_activation_depleted':depleted['NM'],
      'normal_salt':rows[0]['Action_balance_model'],
      'depleted_first_salt':rows[2]['Action_balance_model'],
      'normal_sucrose':rows[1]['Action_balance_model'],
      'depleted_sucrose':rows[4]['Action_balance_model'],
      'depleted_reinforced':rows[5]['Action_balance_model'],
      'repleted_fixed':rows[6]['Action_balance_model'],
      'repleted_after_learning':final['balance'],
      'ablation':rows[7]['Action_balance_model'],
      'control':rows[8]['Action_balance_model'],
      'optional_learning_rate':lr,
      'postlearned_salt_to_approach':Wafter[IDX['Approach'],IDX['UCS_salt']]
    }
    print(pd.DataFrame(rows).to_string(index=False))
    print('METRICS:',metrics_out)
    return metrics_out

if __name__=='__main__': main()
