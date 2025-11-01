# Experimental Analysis Scripts

Scripts for analyzing different scenarios and parameter sensitivities in the AI risk model.

## Files

- **`scenario_analysis_revised.py`** - Compares arms race, cooperation, and treaty scenarios. Analyzes outcomes from both race framing (who wins) and cooperation framing (collective safety metrics).

- **`scenario_comparison.py`** - Compares six scenarios including baseline arms race, constrained China, harder alignment problem, treaty cooperation, slowed progress, and high safety spillover. Focuses on race winners and safety states at win time.

- **`sensitivity_analysis_v2.py`** - Tests sensitivity of race outcomes to key parameters (capability growth rate, safety growth rate, AGI threshold). Reports winners and safety ratios at win conditions.

- **`sensitivity_analysis.py`** - Tests sensitivity across multiple parameters (alpha, gamma, eta, theta, K_threshold). Analyzes final states and total safety debt.

- **`verify_scenarios.py`** - Trajectory verification tool that plots detailed capability and safety trajectories for baseline, hard alignment, and treaty scenarios.

## Usage

All scripts can be run directly and will generate plots in the `plots/` directory. They require `CST_ag.py` in the parent directory.

