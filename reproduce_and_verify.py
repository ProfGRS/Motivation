"""Independent clean-directory checks for the three original EMF simulation scripts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess, sys, shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SIMS=[('Simulation_1_Sodium','EMF_Sodium_IAC_Simulation.py'),('Simulation_2_SelfControl','EMF_SelfControl_Simulation.py'),('Simulation_3_Habits','EMF_Habit_Revised_Simulation.py')]

def near(got,want,atol=1e-6):
    if not np.isclose(float(got),float(want),rtol=0,atol=atol):
        raise AssertionError(f'{got} differs from expected {want} by > {atol}')

def verify_csvs(directory,domain):
    expected=ROOT/'Reference_CSVs'/domain
    refs=sorted(expected.glob('*.csv'))
    assert refs, f'No reference CSVs present for {domain}'
    for ref in refs:
        new=directory/ref.name
        assert new.exists(),f'Missing: {new.name}'
        if new.read_bytes()==ref.read_bytes():
            print('    exact CSV:',ref.name)
            continue
        a,b=pd.read_csv(ref),pd.read_csv(new)
        try:pd.testing.assert_frame_equal(a,b,atol=2e-8,rtol=0,check_exact=False)
        except AssertionError as e:raise AssertionError(f'CSV discrepancy: {domain}/{ref.name}\n{e}') from e
        print('    numeric CSV:',ref.name)

with TemporaryDirectory(prefix='EMF_clean_verify_') as tmp:
    outputs={}
    for domain,script in SIMS:
        work=Path(tmp)/domain;work.mkdir()
        shutil.copyfile(ROOT/domain/script,work/script)
        print(f'Running {domain}: {script}',flush=True)
        p=subprocess.run([sys.executable,str(work/script)],cwd=work,capture_output=True,text=True)
        if p.returncode: raise RuntimeError(f'Execution failed: {p.stderr}\n{p.stdout}')
        print('   ',p.stdout.splitlines()[-1] if p.stdout else 'completed')
        verify_csvs(work,domain)
        outputs[domain]=work
    # Sodium: exact calibration, no-learning state reversal, ablations, optional later learning
    s=pd.read_csv(outputs['Simulation_1_Sodium']/'EMF_IAC_Sodium_Results.csv')
    def salt_row(label):
        row=s[(s.Condition==label)&(s.Cue=='salt')]
        if label=='Sodium depleted; neutral cue':row=s[s.Condition==label]
        assert len(row)==1,(label,len(row))
        return float(row.iloc[0].Action_balance_model)
    for label,v in [('Day 11: normal',-.4),('Day 13: first CS',3.85),('Day 13: CS 2–10',3.85),
                    ('Day 18: repleted (unchanged weights)',-.4),('Sodium depleted; NM gain off',-.4),
                    ('Sodium depleted; neutral cue',0.),('Day 18: repleted (post-reinforcement learning)',.6)]:
        near(salt_row(label),v,1e-6)
    near(salt_row('Day 13: CS + UCS'),4.008109359,1e-5)
    cfg=pd.read_csv(outputs['Simulation_1_Sodium']/'EMF_IAC_Sodium_Settings.csv').set_index('Parameter')
    near(cfg.loc['nm_gain_coefficient','Value'],5.761202985771073,1e-9)
    near(cfg.loc['fitted_postreinforcement_plasticity_rate','Value'],.02452288956527288,1e-9)
    # Self-control
    c=pd.read_csv(outputs['Simulation_2_SelfControl']/'EMF_SelfControl_Results.csv')
    assert c.Decision_cycles.to_list()==[11,24,53]
    assert c.Choice.to_list()==['Tasty','Healthy','Healthy']
    for got,ref in zip(c.RT_model_calibrated_ms,[1299.581831,1330.605624,1399.812545]):near(got,ref,1e-5)
    for name,target in [('Higher_restraint_Mouse_Path.csv',-.009),('Lower_restraint_Mouse_Path.csv',-.242)]:
        near(pd.read_csv(outputs['Simulation_2_SelfControl']/name).mouse_x.min(),target,.002)
    # Habit: archived original output schema
    hroot=outputs['Simulation_3_Habits']
    h=pd.read_csv(hroot/'EMF_Habit_Empirical_vs_Model.csv')
    curve=pd.read_csv(hroot/'EMF_Habit_Learning_Curve.csv')
    cont=pd.read_csv(hroot/'EMF_Habit_Diagnostic_Controls.csv')
    setting=pd.read_csv(hroot/'EMF_Habit_Settings.csv').set_index('parameter')
    near(curve.loc[1,'cue_to_press_weight'],.088263,1e-6)
    near(curve.loc[11,'cue_to_press_weight'],.430797,1e-6)
    near(setting.loc['single_linear_scale_responses_s_at_unit_activation','value'],2.609515552,1e-8)
    def get(episodes,phase,food):
        match=h[(h.training_sessions==episodes)&(h.phase==phase)&(h.food==food)]
        assert len(match)==1
        return float(match.iloc[0].predicted_responses_per_s)
    for args,want in [((2,'After satiation','Unsatiated food'),2.371715),
                      ((2,'After satiation','Satiated food'),1.222805),
                      ((12,'After satiation','Unsatiated food'),2.428285),
                      ((12,'After satiation','Satiated food'),2.117894)]:near(get(*args),want,1e-5)
    assert (cont.query("condition == 'satiated'")['Approach_Frito'].abs()<1e-9).all()
    assert (cont.query("condition == 'satiated, direct cue-to-press removed'")['Press_Frito'].abs()<1e-9).all()
    assert (cont.query("condition == 'satiated'")['Press_Frito']>0).all()
    assert (hroot/'EMF_Habit_Figure5f_Comparison.png').exists()
    print(f'PASS: all three source scripts ran; every archived CSV reproduced; major model outputs and ablations match.')
