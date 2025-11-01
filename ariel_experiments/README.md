# Experimental Analysis Scripts

Analysis tools for the AI race dynamic model, implementing scenario comparisons and parameter sensitivity studies using the CST (Capability-Safety-Trust) framework.

## Files

### `scenario_analysis_revised.py`
Compares three scenarios with different policy modes and framings:
- **Arms Race**: `alpha=1.0`, `eta=0.2`, `beta=0.3`, mode `arms_race`, framing `race`
- **International Cooperation**: `eta=0.6`, `beta=0.5`, `delta_T=0.05`, `T0=0.3`, mode `treaty`, framing `cooperation`
- **Limited Treaty**: mode `treaty`, framing `cooperation`

**Configuration**: `t_span=(0.0, 30.0)`, 301 evaluation points, initial state `K0=[5.0, 4.0, 3.0]`, `S0=[0.5, 0.4, 0.3]`

**Metrics**:
- Race framing: winner identification (WIN_THRESHOLD=50.0), win time, safety ratio at win
- Cooperation framing: system-wide safety ratios, total safety debt over time, maximum risk exposure

**Output**: Multi-panel visualization comparing trajectories and system-wide safety metrics; prints comparison tables.

### `scenario_comparison.py`
Six-scenario comparative analysis with win condition detection:
1. Baseline arms race
2. China compute-constrained (`K0_CN=2.5`)
3. Harder alignment (`K_threshold=12.0`)
4. International treaty (`mode='treaty'`)
5. Slowed progress (`alpha=0.6`)
6. High safety spillover (`eta=0.6`, `beta=0.5`, `T0=0.3`)

**Configuration**: `t_span=(0.0, 50.0)`, 501 evaluation points

**Analysis function**: `analyze_trajectory()` computes winner (first to `K>=50.0`), win time, safety debt `max(0, K - theta*S)`, and total system debt trajectories.

**Output**: Comparative visualizations of winners, safety ratios, win times, and debt evolution; summary statistics table.

### `sensitivity_analysis_v2.py`
Parameter sensitivity focused on race outcomes (winner identification and safety state at win).

**Parameter ranges**:
- `alpha`: `np.linspace(0.6, 1.4, 8)` - capability growth efficiency
- `gamma`: `np.linspace(0.2, 0.8, 8)` - safety growth efficiency  
- `K_threshold`: `np.linspace(8.0, 12.0, 8)` - AGI takeoff threshold

**Baseline**: `alpha=1.0`, `gamma=0.5`, `eta=0.2`, `beta=0.3`, `theta=0.8`, `K_threshold=10.0`

**Configuration**: `t_span=(0.0, 50.0)`, 501 points, initial `K0=[12.0, 9.0, 7.0]`, `S0=[0.2, 0.12, 0.15]`, `mode='arms_race'`

**Output**: 2x3 subplot grid (winner distribution, safety ratio at win vs parameter); statistical summary of win distributions and safest/most dangerous outcomes.

### `sensitivity_analysis.py`
Broader parameter sensitivity analysis examining final state outcomes.

**Parameter ranges**:
- `alpha`: `np.linspace(0.5, 2.0, 10)`
- `gamma`: `np.linspace(0.1, 1.0, 10)`
- `eta`: `np.linspace(0.0, 0.5, 10)` - safety spillover strength
- `theta`: `np.linspace(0.4, 1.2, 10)` - safety effectiveness
- `K_threshold`: `np.linspace(5.0, 15.0, 10)`

**Metrics**: Final capability/safety per player (`K_final`, `S_final`), total safety debt `sum(max(0, K - theta*S))`, maximum capability reached.

**Output**: 3x5 subplot matrix showing final capabilities, safety levels, and total debt vs each parameter; correlation analysis.

### `verify_scenarios.py`
Trajectory inspection tool for debugging and validation.

**Scenarios tested**: Baseline, hard alignment (`K_threshold=12.0`), treaty mode

**Configuration**: `t_span=(0.0, 20.0)`, 201 evaluation points

**Output**: 3x3 subplot grid per scenario (capability trajectories with thresholds, safety trajectories, US safety ratio percentage); prints control values and crossing points.

## Dependencies

All scripts import from `CST_ag`:
- `Params`, `State`, `simulate()`, `simple_scenario_policy_builder()`
- Additional: `safety_debt()`, `unpack_state()` where used

## Output

Plots are saved to `plots/` directory:
- `scenario_analysis_revised.png`
- `scenario_comparison.png`
- `sensitivity_analysis_v2.png`
- `sensitivity_analysis.png`
- `trajectory_verification.png`

All scripts print detailed numerical results and statistics to stdout.

