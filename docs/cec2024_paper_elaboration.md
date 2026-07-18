# CEC 2024 Path Planning — Comprehensive Research Elaboration

**Source paper**: Luo, W., Xu, P., Yang, S., & Shi, Y. (2024). *Benchmark for CEC 2024 Competition on Multiparty Multiobjective Optimization*. arXiv:2402.02033v1 [cs.AI]. 3 February 2024.

**Primary UAV planning reference**: Chen, K., Luo, W., Lin, X., Song, Z., & Yatong, C. (2023). Evolutionary biparty multiobjective UAV path planning: Problems and empirical comparisons. *IEEE Transactions on Emerging Topics in Computational Intelligence* (accepted 2023).

**Competition GitHub**: https://github.com/MiLab-HITSZ/CEC2024_MPMOP

---

## 1. Research Context and Motivation

### 1.1 The Gap the Paper Addresses

Classical multiobjective optimisation (MOO) assumes a single decision-maker who holds a coherent preference ordering over a fixed set of objectives. The Pareto-optimal set represents solutions where no objective can be improved without worsening another, and the goal is to find this set so the decision-maker can choose a preferred point.

Real-world systems, however, frequently involve **multiple independent decision-makers with structurally different and irreconcilable objectives**. A UAV operator wants efficient flight — minimum energy, minimum time, maximum mission coverage. A city authority or the urban public wants minimum risk — lowest pedestrian fatality probability, least noise impact, lowest property damage. These two parties cannot be collapsed into one by assigning weights, because the weights themselves are contested: the operator and the regulator do not agree on how much safety is worth per unit of efficiency.

The paper introduces **Multiparty Multiobjective Optimisation Problems (MPMOPs)** as the formal framework for this class of problems. The core definition:

```
min E(x) = [F₁(x), F₂(x), …, Fₘ(x)]
```

Each `Fᵢ(x)` is a full MOP belonging to a distinct decision-maker. A solution is MPMOP-optimal only when it simultaneously minimises the distance to the Pareto front of **every** party — a far stricter requirement than conventional Pareto optimality.

The paper notes that despite their prevalence in business, engineering, and policy, MPMOPs remain understudied compared to conventional MOPs. The CEC2024 competition benchmark is designed to fill this gap by providing both mathematical test problems (E1–E11 with known Pareto fronts) and a physically grounded real-world case study (C1–C6 UAV path planning).

### 1.2 Why UAV Path Planning as the Applied Case

Urban UAV deployment is a canonical example of a multi-party conflict:

- **The operator party** is rewarded by efficient task completion: short flight paths, low energy consumption, high data collection quality, fast mission turnaround. Every joule saved and every metre of detour avoided has direct economic value.
- **The public/regulatory party** is rewarded by safety and quality of life: minimal risk to pedestrians and vehicles in the event of a crash, minimal damage to buildings, and minimal noise intrusion over residential areas.

These two objectives are genuinely antithetical in an urban environment. The most efficient path (straight line through the city centre at low altitude) is simultaneously the most dangerous (highest pedestrian density, lowest flight altitude increasing crash exposure, maximum noise over populated areas). No tuning of a single algorithm can resolve this without a party-based framework that explicitly represents and balances both perspectives.

The benchmark therefore uses UAV path planning not merely as an engineering exercise but as a **model system for multi-party conflict resolution**, positioning the competition as relevant to urban air traffic management (UTM), U-space frameworks in Europe, and FAA UAS integration road maps in the US.

---

## 2. Benchmark Architecture: Two Parts, Two Metrics

The CEC2024 benchmark has two structurally different components:

| Part | Problems | Pareto fronts known? | Metric | Purpose |
|---|---|---|---|---|
| Part 1 | E1–E11 (abstract MPMOPs) | Yes — analytically derivable | MPIGD | Algorithm development and convergence testing on tractable problems |
| Part 2 | C1–C6 (biparty UAV path planning) | No — computationally intractable | MPHV | Real-world performance evaluation |

The two-part design is deliberate. E1–E11 allow rigorous convergence assessment (algorithms can be graded against the true Pareto front using Inverted Generational Distance). C1–C6 test whether convergence-validated algorithms also perform on problems where no reference front exists and where objectives are physically grounded, non-differentiable, and constrained.

This architecture forces competing algorithms to be both theoretically sound and practically competent — a stronger test than either component alone.

---

## 3. The Biparty Model: Why Not Merge All Objectives Into One MOP?

A natural question is why the benchmark structures the UAV problem as two competing parties rather than a single seven-objective MOP. The answer is both theoretical and practical.

**Theoretically**, merging all objectives into one party conflates structurally different preference orderings. A seven-objective Pareto front would contain solutions where efficiency objectives dominate safety objectives and vice versa. The standard MOP framework offers no principled way to exclude solutions that are Pareto-optimal but correspond to, say, an extremely efficient but fatally dangerous trajectory. The biparty MPMOP framework makes the exclusion explicit: a solution must satisfy **both** parties' Pareto non-dominance conditions simultaneously.

**Practically**, the two-party structure maps directly to real governance: the efficiency party represents the UAV operator (who files the flight plan), and the safety party represents the civil aviation authority or municipality (who approves it). The benchmark models the negotiation between these two institutional actors.

**For algorithm design**, the biparty structure requires evolutionary algorithms to simultaneously maintain two separate solution populations — one approximating the efficiency Pareto front and one approximating the safety Pareto front — while ensuring that the solutions in each population are non-dominated with respect to the other party's objectives. This is fundamentally harder than standard MOEA tasks and requires new operators (multiparty nondominated sorting, MPNDS) developed specifically for this problem class.

---

## 4. UAV Path Representation and Search Space

### 4.1 Trajectory Parameterisation

The UAV trajectory is discretised into a sequence of **n** three-dimensional waypoints:

```
p̄ᵢ = (xᵢ, yᵢ, zᵢ),   i = 0, 1, …, n−1
```

with start `p̄₀ = (1, 1, z₀)` and goal `p̄ₙ₋₁ = (45, 45, zₙ₋₁)` fixed in the horizontal plane (start altitude and goal altitude may be free variables). Consecutive waypoints define track segments:

```
gᵢ = p̄ᵢ₊₁ − p̄ᵢ
```

The objective functions are computed over the segments: energy per segment, risk per waypoint, noise per waypoint. This discretisation converts the infinite-dimensional trajectory optimisation problem into a finite-dimensional parameter optimisation problem suitable for evolutionary algorithms.

### 4.2 The d = 88 Decision Space

The benchmark specifies **d = 88** decision variables for C1–C6. With start and goal positions fixed in (x, y) and altitude treated as a free variable, 88 decision variables most naturally correspond to **44 intermediate waypoints parameterised in (x, y)** with altitude either fixed or separately encoded — or equivalently, approximately **29 intermediate waypoints in full (x, y, z) space** (88/3 ≈ 29.3, rounded). The exact parameterisation follows Chen et al. (TETCI, 2023), the companion paper from which the benchmark is drawn.

This d = 88 is comparable to **15–60 dimensional** problems common in evolutionary optimisation (the abstract E1–E11 use d ∈ {10, 30, 50}) but with the critical difference that the landscape is shaped by physical constraints and risk geography rather than analytically specified.

### 4.3 Continuous vs. Grid Representation

The CEC2024 benchmark uses a **continuous real-valued** search space: each waypoint coordinate is a real number in the mission area, and the path between waypoints is a straight line segment. This is fundamentally different from the graph-search perspective (A*, Theta*, RRT*) used in the repository, where paths are sequences of discrete grid cells.

In the continuous formulation, the challenges for optimisation are:
- **Non-convex feasible region**: turning angle and slope constraints create non-convex constraints on the relationship between consecutive waypoint triplets.
- **Non-differentiable objectives**: the fatality risk involves indicator functions (pedestrian density is piecewise), and noise is clipped at the 40 dB threshold.
- **High dimensionality**: 88 variables compared to the tens that evolutionary algorithms handle most reliably.
- **Multi-modal landscape**: the risk objectives have local minima corresponding to routing through less populated sub-areas of the city.

---

## 5. Efficiency Decision-Maker: Four Objectives in Depth

The efficiency party represents the UAV operator's economic and operational interests. Its four objectives are hierarchically related in physical realism.

### 5.1 Path Length (f_length)

```
f_length = Σᵢ₌₀ⁿ⁻¹ ‖gᵢ‖
```

Path length is the simplest and most interpretable efficiency metric. It is both directly related to flight time (at fixed speed V = 10 m/s) and a lower bound on energy consumption. In an unobstructed environment, the minimum is the Euclidean straight-line distance from (1,1) to (45,45), approximately 62 spatial units.

However, path length alone is blind to altitude: a path that meanders vertically while covering the same horizontal distance as the straight-line path will have a higher `f_length` only if the vertical deviations increase the 3D Euclidean distance. This limitation motivates `f_height` as a complementary objective.

### 5.2 Flight Energy Consumption (f_fuel)

```
fuelᵢ = W^(3/2) √(G³ / (2ρᵢζn)) · ‖gᵢ‖/V + Δ⁺·W·G

where: ρᵢ = ρ₀ · exp(−(zᵢ₊₁ + zᵢ) / (2 × 10.7))
       ρ₀ = 1.225 kg/m³,  W = 1.38 kg,  n = 4,  ζ = 0.1 m²,  V = 10 m/s
```

The energy model has two physically distinct terms:

**Forward flight term** (`W^(3/2) √(G³/2ρᵢζn) · ‖gᵢ‖/V`): This is the hover-equivalent power (derived from actuator disk theory) multiplied by flight time per segment. The key property is the `ρᵢ^(−1/2)` dependence: **at altitude, air is less dense, rotor efficiency drops, and energy per unit distance increases**. Specifically, at z = 100 m, ρ ≈ 1.225 · exp(−100/10.7) ≈ 0.71 kg/m³, so the forward flight cost is ≈ (1.225/0.71)^(1/2) ≈ 1.31× higher than at sea level.

**Climb term** (`Δ⁺·W·G`): The gravitational potential energy required to gain altitude, where Δ⁺ is the altitude increase for segment i (zero for descents, since kinetic energy recovery is not modelled in this simplified model). This term penalises trajectories that climb unnecessarily.

The combination creates a **sweet spot altitude**: flying too low keeps ρᵢ high (efficient rotors) but increases risk; flying too high reduces ρᵢ (inefficient rotors) and requires climbing energy. For a 1.38 kg quad-rotor with these parameters, the optimal cruise altitude for energy efficiency alone is near the minimum allowed altitude H_min.

This is the most physically sophisticated efficiency objective and the only one where altitude directly enters the efficiency party's cost function — making it crucial for the altitude-risk trade-off at the heart of the benchmark.

### 5.3 Path Height Changes (f_height)

```
f_height = Σᵢ₌₀ⁿ⁻¹ |zᵢ₊₁ − zᵢ|
```

This objective penalises altitude oscillation — a trajectory that climbs and descends repeatedly. The motivation is that altitude changes stress the UAV's power system: rapid altitude variations require high transient power demand (for climbing) and mechanical stress on the motor drive train (for descending). Minimising `f_height` produces **smooth altitude profiles** that extend UAV battery and mechanical life.

Crucially, `f_height` is independent of absolute altitude: a trajectory that cruises smoothly at 200 m has a lower `f_height` than one that fluctuates between 10 m and 50 m repeatedly, even if the latter spends more time at a fuel-efficient low altitude. This creates complementary behaviour with `f_fuel`: together they push toward a low, smooth altitude profile.

### 5.4 Mission Hover Point Distance (f_distance)

```
f_distance = Σₖ₌₀ᴷ⁻¹ minᵢ ‖p̄ᵢ − pₖʲᵒᵇ‖
```

For K = 3 UAV Hover Points (UHPs) at **(25, 30)**, **(34, 20)**, and **(40, 35)**, this objective measures how closely the trajectory passes each pre-specified data collection site. It captures the **mission payload value**: a UAV executing a surveillance or sensor-collection mission must physically approach each target location to gather useful data.

The geometric relationship between UHP locations and the straight-line path is important. The straight-line path from (1,1) to (45,45) follows the diagonal; the three UHPs are slightly off this diagonal. The tension is:
- Ignoring UHPs: f_distance is large, mission fails.
- Visiting all UHPs: trajectory deviates from the diagonal, f_length increases.
- Flying near but not over UHPs: f_distance is small but not zero.

Importantly, `f_distance` and `f_fatal` are **partially aligned** for UHPs near the city periphery (lower population density = lower risk AND the detour to visit them is short), but **in conflict** for UHPs near the city centre (high density = high risk, but the efficiency party must visit them).

---

## 6. Safety Decision-Maker: Three Objectives in Depth

The safety party represents the urban public, civil aviation authorities, and property owners. Its three objectives are grounded in published risk assessment methodologies for urban UAV operations.

### 6.1 Fatality Risk (f_fatal)

```
f_fatal = Σᵢ₌₀ⁿ [crₚ(xᵢ, yᵢ, zᵢ) + crᵥ(xᵢ, yᵢ, zᵢ)]

crₚ(x,y,z) = P_crash · Sₕ · σₚ(x,y) · Rₚᶠ(z)
crᵥ(x,y,z) = P_crash · Sₕ · σᵥ(x,y) · Rᵥᶠ(z)
```

This objective implements the **Third-Party Risk (TPR) model** from Pang et al. (2022), which is the standard risk quantification framework used in UAV regulatory analysis (including EASA's SORA methodology and NASA's UTM framework).

The risk model has three multiplicative components at each waypoint:

**Crash probability component** (P_crash · Sₕ): Fixed product of the UAV's hardware/software failure rate and its crash impact footprint. This is constant across the trajectory for a given UAV platform.

**Exposure component** (σₚ(x,y)): Population density at the horizontal position (x,y). Generated using a **radial basis function** centred on the metropolitan core, this creates a smooth spatial gradient with peak density at the city centre and decreasing density toward the periphery. The vehicle density σᵥ follows a similar model biased toward roads and traffic corridors.

**Altitude-dependent impact component** (Rₚᶠ(z)): This factor increases with altitude. Higher altitude → higher terminal velocity on impact → greater kinetic energy per crash event → higher fatality probability per exposed person. The altitude dependence is what makes `f_fatal` nuanced: flying high is not always safer, because the impact energy is greater.

The combined effect creates a **two-dimensional risk landscape** in (x,y) space (high in the populated city centre, low in the periphery) that is modulated by altitude (higher z increases per-person risk but reduces exposure time if the crash zone footprint is assumed fixed). The optimal `f_fatal` trajectory routes through low-density areas at moderate altitude — high enough to reduce ground-level exposure time but not so high that impact energy becomes lethal for a wider area.

### 6.2 Property Risk (f_eco)

```
f_eco = Σᵢ₌₀ⁿ crₚ,ₐ(zᵢ)

crₚ,ₐ(z) = ψ(eᵘ)  if 0 < z ≤ eᵘ
            ψ(z)   otherwise

ψ(z; μ, σ) = 1/(z·σ·√(2π)) · exp(−(ln z − μ)² / (2σ²))
             μ = 3.04670,  σ = 0.76023
```

The building height distribution follows a **lognormal model** calibrated to real urban data. With μ = 3.04670, the median building height is e^μ ≈ 21 m (a typical 7-storey European urban building). The standard deviation σ = 0.76023 produces a realistic mix: some buildings at 5 m (2 storeys), some at 100 m (30 storeys), reflecting the actual variance in a dense city.

The property risk function peaks at the **modal building height** (~15–21 m) — the altitude range where the UAV is most likely to collide with a building façade, breaking windows or damaging the structure. At very low altitude (below the lowest buildings), the UAV flies in the street canyon and property risk shifts to near-zero (open space). At very high altitude (above all buildings), property risk also approaches zero. But the constraint H_min prevents ground-hugging, and the energy cost of H_max-altitude flight is prohibitive.

This creates a **forbidden altitude band** around 15–30 m — the urban canyon ceiling — that is both geometrically constrained and risk-penalised. Effective trajectories must thread above this band (higher altitude, more energy) or carefully route to avoid building clusters.

### 6.3 Noise Pollution (f_noise)

```
f_noise = Σᵢ₌₀ⁿ C_noise(xᵢ, yᵢ, zᵢ)

C_noise(x,y,z) = k · σₚ(x,y) · Lₕ / (zᵢ² + d²)
                 (omitted when level < 40 dB threshold)
```

The noise model uses **spherical sound propagation** (inverse square law in the denominator `zᵢ² + d²`), weighted by population density. The 40 dB threshold corresponds to the WHO guideline for urban residential noise, below which the impact on quality of life is considered negligible.

The key physical property: noise intensity falls as `1/r²` with distance r. For a UAV directly overhead at altitude z, the dominant distance is z itself. Doubling altitude from 20 m to 40 m reduces the noise intensity at ground level by a factor of 4. This creates a **strong incentive to fly high** from the noise perspective.

However, the energy objective (`f_fuel`) creates the opposite incentive (lower altitude = denser air = efficient rotors). This efficiency–noise conflict is one of the most direct and quantifiable in the benchmark, and it is most acute in the C4–C6 cases where `f_noise` replaces `f_eco` as the safety party's second objective.

---

## 7. The Full Objective Conflict Landscape

The seven objectives define a rich network of trade-offs. Understanding which objectives cooperate and which conflict is essential for algorithm design and for understanding the difficulty of each benchmark case.

### 7.1 Intra-Party Conflicts (Within-Party Trade-offs)

**Efficiency party**:
- `f_length` vs. `f_distance`: **direct conflict**. The shortest path ignores the three UHPs; visiting all UHPs requires detours that increase path length.
- `f_fuel` vs. `f_height`: **partial alignment**. Minimising altitude changes (smooth profile) reduces transient power demands, and a smooth low-altitude profile also minimises energy. But the optimal fuel altitude (low) is not necessarily smooth if the terrain or airspace forces altitude variation.
- `f_length` vs. `f_height`: **independent in 2D horizontal plane**, but correlated in 3D: a long horizontal path at constant altitude has low `f_height`; any path that must change altitude to avoid obstacles incurs `f_height` cost regardless of horizontal length.

**Safety party**:
- `f_fatal` vs. `f_eco`: **partial alignment**. Both push away from populated areas and from the building-height band. But `f_fatal` also increases with altitude (impact energy), while `f_eco` decreases above the building height. So at high altitude, `f_fatal` is relatively high but `f_eco` is near zero — a subtle misalignment.
- `f_fatal` vs. `f_noise`: **strong alignment**. Both penalise flying over the city centre (high population density). A trajectory that avoids the city centre is safer AND quieter. This is the strongest within-party alignment in the benchmark.

### 7.2 Inter-Party Conflicts (Efficiency vs. Safety)

This is the defining tension of the benchmark:

| Efficiency objective | Safety objective | Relationship | Mechanism |
|---|---|---|---|
| f_length | f_fatal | **Major conflict** | Shortest path cuts through city centre (peak σₚ) |
| f_length | f_eco | **Moderate conflict** | Direct path at building height risks collisions |
| f_length | f_noise | **Major conflict** | Direct low path over city = maximum noise exposure |
| f_fuel | f_fatal | **Complex** | Low altitude → efficient fuel AND lower impact energy; but low altitude over city = high fatality from pedestrian exposure |
| f_fuel | f_noise | **Direct conflict** | Low altitude → efficient rotors BUT maximum noise at ground level |
| f_fuel | f_eco | **Direct conflict** | High altitude → safe from buildings but energy-costly |
| f_height | f_fatal | **Indirect** | Smooth altitude profile may pass through dangerous zones if the smooth profile happens to be at building height |
| f_distance | f_fatal | **Conditional** | If UHPs are in low-density areas: no conflict. If UHPs are near city centre: strong conflict |

The most fundamental inter-party conflict is between **f_fuel and f_noise**: energy efficiency demands low altitude (dense air, efficient rotors), while noise safety demands high altitude (inverse-square attenuation). This trade-off is analytically clean, physically direct, and impossible to resolve without explicit multi-party negotiation.

### 7.3 Altitude as the Central Variable

The altitude variable z is the dimension where nearly every objective conflict is mediated:

```
Low altitude (z → H_min):
  f_fuel    ↓  (dense air, efficient)
  f_height  (depends on terrain)
  f_fatal   ↓ (less impact energy per crash, BUT high pedestrian exposure time)
  f_eco     ↓ (below buildings, no collision risk)
  f_noise   ↑ (close to ground, maximum noise impact)

Medium altitude (z ≈ eᵘ ≈ 21 m, modal building height):
  f_fuel    moderate
  f_eco     ↑ (peak property collision risk)
  f_fatal   moderate-high
  f_noise   moderate-high

High altitude (z → H_max):
  f_fuel    ↑ (thin air, inefficient rotors + climbing energy)
  f_eco     ↓ (above all buildings)
  f_fatal   ↑ (high impact energy from terminal velocity)
  f_noise   ↓ (inverse-square attenuation)
```

No single altitude minimises all objectives. This altitude conflict structure is what makes the benchmark genuinely multi-dimensional and not reducible to a single-objective problem by fixing z.

---

## 8. The Six Benchmark Cases as a Designed Progression

The six cases C1–C6 are not arbitrary combinations. They form a deliberate **taxonomy of trade-off complexity**, from simplest to most physically complete:

### C1: Baseline Case — Length vs. Risk

```
Efficiency DM: (f_length, f_distance)
Safety DM:    (f_fatal, f_eco)
```

The minimal case establishing the fundamental tension. The efficiency party cares about horizontal path efficiency and data collection coverage. The safety party cares about crash fatality risk and building collision risk.

**Key trade-off**: The shortest horizontal path through the city exposes the safety party to maximum fatality and property risk. The algorithm must find the Pareto front of trajectories that trade horizontal efficiency for spatial risk avoidance. Altitude is not directly involved in either party's objectives (f_fatal has altitude dependence through Rᶠ(z), but the efficiency party has no reason to prefer any particular altitude).

### C2: Adding Altitude Smoothness

```
Efficiency DM: (f_length + f_height, f_distance)
Safety DM:    (f_fatal, f_eco)
```

Combines path length and altitude smoothness into the efficiency party's first objective (as a sum). This forces the algorithm to consider 3D trajectory smoothness, not just horizontal distance.

**Key addition**: Trajectories that escape city-centre risk by climbing must now account for the altitude oscillation penalty. A trajectory that rapidly climbs over the city centre, then descends on the far side, reduces `f_fatal` and `f_eco` but incurs a large `f_height` penalty. This case tests whether algorithms can find smooth altitude profiles that also provide spatial risk separation.

### C3: Physics-Based Energy vs. Risk

```
Efficiency DM: (f_fuel, f_distance)
Safety DM:    (f_fatal, f_eco)
```

The most physically grounded efficiency party. Energy consumption is altitude-dependent, creating the first case where the efficiency party has **intrinsic altitude preference** (low altitude = more efficient). This partially aligns the efficiency party with the safety party on `f_eco` (both prefer to fly below or above the building height band) but creates a new conflict with `f_fatal` (low altitude → efficient but higher impact-energy risk) and with `f_noise` (not present in C3 safety objectives, but would appear in C6).

**Key addition**: Altitude now enters both parties' cost calculations, creating a richer three-way tension: altitude affects energy (efficiency), impact energy on crash (safety/fatal), and building collision risk (safety/eco). C3 is where the physics of the benchmark become most consequential.

### C4: Replacing Property Risk with Noise Pollution

```
Efficiency DM: (f_length, f_distance)
Safety DM:    (f_fatal, f_noise)
```

Identical to C1 except `f_eco` (property collision risk) is replaced by `f_noise` (noise pollution). This shifts the safety party's concern from structural damage to quality-of-life impact.

**Key change**: `f_eco` penalises altitude around the building height band (a sharp spatial penalty); `f_noise` penalises low altitude everywhere over populated areas (a broad, continuous penalty). C4 therefore produces **stronger altitude pressure** on the safety party — the safety party now consistently prefers high altitude, while in C1 the altitude preference was non-monotonic (avoid building height, but high altitude increases impact energy through `f_fatal`).

### C5: Altitude Smoothness + Noise (C2 + C4 combined)

```
Efficiency DM: (f_length + f_height, f_distance)
Safety DM:    (f_fatal, f_noise)
```

Combines the altitude-smoothness efficiency objective from C2 with the noise-safety objective from C4. The efficiency party wants a smooth, short trajectory at low altitude; the safety party wants quiet flight at high altitude. This creates a **clean, direct conflict between altitude smoothness (efficiency) and altitude height (safety)** — the algorithm must balance staying low-and-smooth (efficient) vs. climbing high-and-staying-there (safe).

### C6: Most Complete Case — Energy + Noise

```
Efficiency DM: (f_fuel, f_distance)
Safety DM:    (f_fatal, f_noise)
```

The most physically comprehensive case, combining the altitude-dependent energy model (C3) with the noise pollution safety objective (C4). Both objectives directly depend on altitude — energy favours low altitude, noise favours high altitude — creating the **clearest and most direct efficiency–safety conflict** in the entire benchmark.

This case is most representative of real-world urban UAV deployment trade-offs: a delivery drone operator (minimise energy and reach drop-off locations) versus city residents (minimise noise over homes). C6 is the recommended case for highlighting the fundamental biparty tension of the benchmark.

---

## 9. Physical Grounding: Real-World Parameter Values

The benchmark parameters are calibrated to correspond to real hardware and regulatory standards:

| Parameter | Value | Real-world correspondence |
|---|---|---|
| UAV weight W | 1.38 kg | DJI Phantom 4 (1.38 kg, most widely studied research UAV) |
| Number of rotors n | 4 | Standard quadrotor configuration |
| Flight speed V | 10 m/s | Typical DJI Phantom/Mavic forward cruise speed |
| Rotor disk area ζ | 0.1 m² | Consistent with Phantom 4 rotor geometry |
| Air density ρ₀ | 1.225 kg/m³ | ISA sea level value |
| Scale height | 10.7 (in exp) | Atmospheric scale height (simplified barometric formula) |
| Max turning angle α_max | π/3 = 60° | Conservative manoeuvring limit for multi-rotor at V = 10 m/s |
| Max slope angle β_max | π/4 = 45° | Aggressive but achievable climb/dive rate at cruise speed |
| Building height μ | 3.04670 | Lognormal median ≈ 21 m (7-storey building, typical EU urban) |
| Building height σ | 0.76023 | Large variance: 5th percentile ≈ 4 m, 95th percentile ≈ 110 m |
| Noise threshold | 40 dB | WHO Environmental Noise Guidelines for urban areas (night-time limit) |
| Mission area | 45 × 45 units | Scalable: can represent ~4.5 km × 4.5 km at 100 m resolution |
| UHP locations | (25,30), (34,20), (40,35) | Off the direct diagonal, requiring meaningful detours |

The DJI Phantom 4 calibration is significant because it makes this benchmark **directly interpretable** in terms of a specific, commercially available platform. Results are not abstract — a lower f_fuel score corresponds to a measurably longer flight time on the Phantom 4's 5350 mAh LiPo battery.

The lognormal building height distribution (μ = 3.04670, σ = 0.76023) matches empirical data from dense European city centres. The 95th percentile building height of approximately 110 m defines the upper practical boundary for urban UAV operations in many jurisdictions (EASA U2 category upper limit: 120 m AGL).

---

## 10. Evaluation Protocol in Depth

### 10.1 Why MPHV Rather Than MPIGD for UAV Problems

For the abstract E1–E11 problems, the true Pareto front is analytically known, enabling MPIGD — a convergence-focused metric that measures average distance from the algorithm's solution set to the true Pareto front.

For C1–C6, the true Pareto front is **computationally intractable** for three reasons:

1. **Continuous search space**: d = 88 continuous variables. The Pareto front in objective space cannot be enumerated.
2. **Non-convex constraints**: The turning angle and slope constraints create non-convex feasible regions with no analytical characterisation.
3. **Physical complexity**: The risk objectives involve spatially varying density functions, altitude-dependent models, and threshold indicators that make the problem highly multimodal.

The **Multiparty Hypervolume (MPHV)** is therefore used:

```
MPHV = Σᵢ₌₁ᴷ HVᵢ
```

where HVᵢ is the standard hypervolume indicator for decision-maker i's solution set relative to a reference point. MPHV is **higher-is-better**: it rewards both convergence (solutions close to the unknown true Pareto front) and diversity (solutions spread across the trade-off space). By summing across both decision-makers, MPHV captures whether the algorithm satisfies both parties simultaneously.

The hypervolume indicator has well-known theoretical properties (it is the only scalar indicator that is Pareto-compliant and strictly monotone for any addition of non-dominated solutions), making it a principled choice when no reference front is available.

### 10.2 The 30-Run Statistical Protocol

Each algorithm is run independently 30 times with random seeds fixed from 1 to 30. Participants report the best, median, worst, mean, and standard deviation of MPHV across these 30 runs.

The fixed seeds (rather than random seeds) serve two purposes:
- **Reproducibility**: Any researcher can re-run the exact same 30 trials.
- **Fair comparison**: All algorithms see exactly the same sequence of random initializations, ensuring performance differences reflect algorithm quality rather than favourable random starts.

The 100,000 maximum fitness evaluations budget for C1–C6 is the key computational constraint. At 100k evaluations, typical evolutionary populations of size ~100 can complete ~1,000 generations. This is sufficient for convergence on low-dimensional problems but challenging for d = 88 with non-convex constraints.

---

## 11. What the Paper Does Not Discuss: Identified Gaps

The benchmark paper is a competition specification document, not a research paper. Several important aspects of the path planning problem are explicitly or implicitly excluded:

**Dynamic obstacles**: The benchmark assumes a static environment. Moving pedestrians, vehicles, and other UAVs are not modelled. Real urban environments require online re-planning.

**Actual 3D terrain and building data**: The building height distribution is generated stochastically from a lognormal model. Individual building locations are not specified. Real urban navigation requires full 3D occupancy data.

**Wind and weather effects**: The energy model assumes calm air and constant speed. Wind significantly affects multi-rotor energy consumption and introduces flight dynamics not captured by the benchmark.

**Battery capacity constraint**: `f_fuel` is minimised as an objective, but there is no hard constraint that the total energy must not exceed the UAV's battery capacity. A real mission plan must satisfy the energy budget as a hard constraint, not just minimise energy.

**Regulatory no-fly zones and corridors**: Beyond terrain altitude bounds, the benchmark imposes no airspace constraints (restricted zones, approach corridors, altitude bands reserved for manned aircraft).

**Sense-and-avoid**: The benchmark assumes perfect knowledge of the environment. Real UAVs require onboard sensors for obstacle detection and avoidance.

**Multi-UAV coordination**: The benchmark optimises a single UAV path. Fleet-level coordination, communication constraints, and collision avoidance between UAVs are out of scope.

**Resolution sensitivity**: The stochastic population model and building height model produce a different urban environment for each random seed. The benchmark does not study how results vary with model resolution or population distribution parameters.

**Statistical testing**: The paper specifies the results format (best/median/worst/mean/std over 30 runs) but does not prescribe statistical significance tests (Wilcoxon, Friedman) for comparing algorithms. This is left to the competition organisers.

---

## 12. Connecting to the Three Agentic Ideas

The CEC2024 BPMO-UAVPP benchmark illuminates why the three agentic roles in this repository are not just academically interesting but practically motivated by the benchmark's structure.

### 12.1 RefinerAgent in the CEC2024 Context

The Refiner's role — taking a classical planner's geometrically valid path and refining it using semantic environmental context — maps directly to the biparty structure:

- **Classical planner** (A* in 2D projection, or RRT* in continuous space) produces a path that satisfies the efficiency party's `f_length` objective but is blind to the safety party's objectives.
- **LLM Refiner** receives that path plus the semantic context: population density map, building height distribution, UHP locations, and risk sensitivity.
- **Refinement goal**: shift waypoints to reduce `f_fatal` and `f_noise` while preserving acceptable `f_length`.
- **Reflect step**: validate turning angles (|αᵢ| ≤ π/3) and slope angles (|βᵢ| ≤ π/4) — exactly the constraints that make the CEC2024 feasible region non-convex.

In the CEC2024 context, the Refiner is a **post-hoc safety party agent** that adjusts an efficiency-optimal trajectory to satisfy safety party constraints without re-planning from scratch.

### 12.2 GeneratorAgent in the CEC2024 Context

The Generator's strength — writing a mission-tailored planning algorithm from a natural language description — maps to the biparty framework's need for algorithms that encode specific party preferences:

- **C1 mission** (length + fatality): "Find the shortest path from (1,1) to (45,45) visiting hover points at (25,30), (34,20), (40,35), routing away from the city centre to reduce pedestrian fatality risk."
- **C6 mission** (fuel + noise): "Minimise fuel consumption. The flight must be quiet over residential areas — cruise at minimum 40 m altitude over any location with population density above threshold."

Different mission descriptions generate algorithms that implicitly weight different points on the efficiency–safety Pareto front. This makes the Generator a **natural-language Pareto point selector** — the user specifies the trade-off preference in words, and the LLM writes the algorithm that implements it.

The Generator also addresses the benchmark's most important limitation: **no off-the-shelf algorithm is designed for the specific combination of objectives in any given case**. C3 requires an algorithm that knows air density decreases with altitude and should avoid climbing when energy budget is tight; no classical planner encodes this. The Generator can write it from a physics-informed mission description.

### 12.3 SelectorAgent in the CEC2024 Context

The Selector's role — reasoning over real-time conditions to choose the best algorithm from a registry — maps to the benchmark's six cases as follows:

- Different cases (C1–C6) correspond to different environmental and operational conditions. The conditions that make C3 the appropriate formulation (energy budget is critical) differ from those that make C4 appropriate (noise complaints are the primary regulatory concern).
- The Selector can be prompted with: "The UAV is operating over a noise-sensitive residential area. The city council has imposed strict noise limits. Battery is 80%. Which planning algorithm is most appropriate?" and select from a registry that includes: energy-first A*, noise-avoiding RRT*, safety-weighted MPC.
- In the benchmark's biparty framing, the Selector is a **meta-level party negotiator**: it identifies which algorithmic approach best represents the party preferences most relevant to the current mission context.

The Selector addresses the benchmark's assumption that the objectives are fixed per case. In practice, the active objectives change in real time based on airspace conditions, regulatory reports, and weather — exactly the scenario the Selector is designed for.

---

## 13. Implications for the Repository's Grid-Based Simplification

The repository implements a 2D grid-based interpretation that is a **deliberate research abstraction** of the CEC2024 benchmark, not a misrepresentation of it. The key differences are:

| Dimension | Actual CEC2024 | This Repository |
|---|---|---|
| Space | 3D continuous (x, y, z) | 2D discrete grid |
| Optimisation | Multi-objective, 2 DMs, 4–7 objectives | Single quality metric |
| Safety model | f_fatal, f_eco, f_noise (population-based) | SD: min distance to obstacle cell |
| Energy model | Physics-based (altitude + rotor physics) | Not modelled |
| Mission constraints | Hover points, turning, slope, altitude | Collision-free only |
| Evaluation | MPHV, 30 runs, 100k evaluations | PLR, SD, SM, SR, 5–10 runs |
| Algorithms | MPMOEAs | Graph search + LLM |

The simplification is appropriate because:

1. **LLM role isolation**: The research question is whether LLMs add value over classical planners — not whether any planner performs well on the full CEC2024 problem. A simpler benchmark isolates the LLM contribution from physical modelling complexity.

2. **Interpretable baselines**: On a grid, A* is demonstrably PLR-optimal. This gives a precise lower bound for the efficiency objective. No such analytical baseline exists for the continuous f_fuel problem.

3. **Reproducibility**: Grid maps are deterministic. The stochastic urban model in CEC2024 requires separate sampling and model calibration, introducing confounding factors.

4. **Path to extension**: The three agentic ideas (Refiner, Generator, Selector) are designed to extend naturally to the full CEC2024 problem. The grid-based implementation is Phase 1; extending to continuous 3D with biparty objectives is a natural Phase 2.

The V1 classical results establish that the baseline methods work correctly on the simplified benchmark. Phase 2 would replace the grid with continuous waypoints, add the UAV physics model, and evaluate using MPHV against NSGA-II-type baselines.

---

## 14. Summary

The CEC2024 competition's Biparty Multi-objective UAV Path Planning benchmark (Section 2.3, cases C1–C6) is a physically grounded, multi-party optimisation problem that models the fundamental conflict between a UAV operator's efficiency interests and the urban public's safety interests. Its key contributions to the path planning field are:

1. **Formal biparty structure** that prevents the efficiency–safety conflict from being dissolved by weight assignment, forcing explicit multi-party Pareto optimisation.
2. **Physics-based efficiency objectives** (f_fuel with altitude-dependent air density) that make altitude a first-class optimisation variable, not an engineering afterthought.
3. **Population-based safety objectives** (f_fatal, f_eco, f_noise) grounded in published UAV third-party risk and noise models, calibrated to a realistic DJI Phantom 4 platform in a lognormal European urban environment.
4. **Six benchmark cases** that systematically vary the active objectives from the simplest (C1: length + risk) to the most complete (C6: energy + noise), providing a graded difficulty progression for algorithm development.
5. **MPHV evaluation over 30 fixed-seed runs** with 100,000 evaluations per run, providing a statistically rigorous framework without requiring a true Pareto front reference.

For this repository's research, the benchmark motivates all three agentic ideas: the Refiner as a post-hoc safety party agent, the Generator as a natural-language Pareto point selector, and the Selector as a meta-level algorithm choice agent that adapts to the active party objectives in real time.
