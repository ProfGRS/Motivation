# Emergent Motivation Framework — reproducibility package (Section 3)

Three complete executable interactive-activation-and-competition (IAC) simulations accompany the manuscript **The Emergence of Motivated Behavior: A Neural Network Perspective**.

## Run and verify

In the package directory:

```bash
python -m pip install -r requirements.txt
python reproduce_and_verify.py
```

The verification script launches each simulation from its source in a separate temporary directory, regenerates its result tables, verifies them against frozen reference CSVs, and tests key reported outputs and controls. All scripts are deterministic. Source scripts are preserved under their original filenames.

| Simulation | Executable | Central claim |
|---|---|---|
| 1. Sodium need | `Simulation_1_Sodium/EMF_Sodium_IAC_Simulation.py` | Salt-cue attraction reverses with acute sodium need despite fixed initial learned weights; control removal of selective NM gain eliminates reversal. |
| 2. Self-control | `Simulation_2_SelfControl/EMF_SelfControl_Simulation.py` | A single five-unit network generates delayed goal-consistent choices and characteristic mouse-trajectory curvature. |
| 3. Habits | `Simulation_3_Habits/EMF_Habit_Revised_Simulation.py` | Two or twelve training episodes gradually strengthen direct cue-to-press links and reproduce reduced devaluation sensitivity. |

The Simulation 3 script was recovered from the prior project file library (October 10, 2026 original working package). Its executable specification and results were tested against the supplementary methods. Each simulation's original script creates a set of CSV files and plot images in its own folder. Reference copies of the CSVs are in `Reference_CSVs/`.

## Interpreting comparisons

Figure readings from Robinson and Berridge (2013) and Tricomi et al. (2009) are **approximate group-mean estimates from published plots**; no individual-level raw observations are included. In Simulation 1 three parameters are calibrated to figure-estimated anchors, with an additional optional, separately fitted post-reinforcement learning rate. In Simulation 2 the network cycle-to-millisecond conversion is fitted to three approximate food-choice latencies and mouse trajectories are qualitative model outputs. Simulation 3 calibrates one common scale from predicted Press-unit activation to group-average responses/second based on the four pre-satiation empirical estimates. Exact agreement at these calibration points should not be confused with independent predictive validation. These simulations demonstrate particular mechanistic sufficiency, not uniqueness of the EMF.

See `Supplementary_Methods/EMF_Section3_Verified_Supplementary_Methods.docx` for equations, all relevant weights, experimental-condition settings, calibrated parameters, and numerical comparisons. No artificial subject-level observations, bootstrap samples or empirical p-values were created.
