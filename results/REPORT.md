# IMDSOC evaluation report

Synthetic network, 14 test days, 1490 unique orders, seed 7. All results are from simulation, not deployment.

## Predictor diagnostics (held-out later period)

```json
{
  "n_test": 3500.0,
  "coverage_q50": 0.5126,
  "pinball_q50": 1.96,
  "coverage_q90": 0.8989,
  "pinball_q90": 1.6038,
  "coverage_q98": 0.9691,
  "pinball_q98": 0.5334,
  "excursion_brier": 0.0983,
  "excursion_ece": 0.0255,
  "excursion_rate": 0.4311,
  "failed_brier": 0.0237,
  "failed_ece": 0.0011
}
```

## Metrics by strategy

|  | imdsoc | fixed_site | nearest_eligible | lowest_shipping_cost | inventory_only | rules_only | abl_no_consolidation | abl_point_estimate | abl_no_transit_model | abl_no_excursion_model | abl_no_failed_model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| served_rate | 0.988 | 0.987 | 0.987 | 0.987 | 0.987 | 0.987 | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| escalation_rate | 0.052 | 0.013 | 0.013 | 0.013 | 0.013 | 0.013 | 0.052 | 0.012 | 0.012 | 0.052 | 0.052 |
| cost_per_rx | 55.323 | 224.796 | 231.664 | 459.958 | 280.252 | 379.329 | 58.231 | 52.924 | 54.113 | 71.727 | 63.610 |
| on_time_rate | 0.984 | 0.988 | 0.983 | 0.940 | 0.982 | 0.927 | 0.987 | 0.988 | 0.965 | 0.984 | 0.986 |
| excursion_rate | 0.019 | 0.080 | 0.095 | 0.540 | 0.111 | 0.383 | 0.019 | 0.020 | 0.022 | 0.034 | 0.022 |
| failed_rate | 0.020 | 0.024 | 0.024 | 0.029 | 0.022 | 0.023 | 0.021 | 0.022 | 0.026 | 0.020 | 0.024 |
| consolidation_rate | 0.433 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.450 | 0.455 | 0.431 | 0.431 |
| cross_site_rate | 0.435 | 0.217 | 0.439 | 0.439 | 0.746 | 0.339 | 0.363 | 0.433 | 0.429 | 0.416 | 0.430 |
| write_off_value | 1089.850 | 2597.250 | 5109.800 | 5109.800 | 3568.400 | 97.250 | 1097.250 | 100.250 | 95.850 | 1106.250 | 1094.250 |
| utilization_cv | 0.475 | 0.490 | 0.582 | 0.582 | 0.498 | 0.487 | 0.420 | 0.467 | 0.465 | 0.462 | 0.486 |
| backlog_variance | 1.006 | 3.161 | 5.302 | 5.302 | 5.581 | 3.056 | 0.828 | 0.874 | 0.590 | 0.863 | 1.182 |
| missed_pickup_rate | 0.035 | 0.064 | 0.092 | 0.092 | 0.077 | 0.046 | 0.024 | 0.037 | 0.037 | 0.034 | 0.041 |
| miles_per_rx | 192.566 | 209.781 | 180.555 | 180.555 | 347.638 | 252.605 | 236.111 | 188.031 | 187.393 | 193.174 | 193.851 |
| kg_co2e_per_rx | 0.689 | 0.574 | 0.522 | 0.342 | 0.811 | 0.432 | 0.740 | 0.665 | 0.637 | 0.621 | 0.691 |
| review_rate | 0.465 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.391 | 0.480 | 0.482 | 0.698 | 0.462 |
| solve_seconds | 0.052 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.033 | 0.076 | 0.081 | 0.053 | 0.062 |
| max_gap | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| violations | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |


## Hypotheses

- **H1** (imdsoc vs fixed_site): not supported. cost diff $-169.46 [-241.27, -103.28]; on-time diff -0.004 [-0.011, +0.003]
- **H1** (imdsoc vs nearest_eligible): supported. cost diff $-176.33 [-251.11, -110.81]; on-time diff +0.001 [-0.006, +0.008]
- **H2** (chance-constrained vs point-estimate plans): not supported. excursion rate 0.0189 vs 0.0203; paired diff -0.0014 [-0.0095, +0.0081]
- **H3** (consolidation vs none): supported. shipping+packaging per Rx $24.26 vs $27.02; paired diff -2.77 [-3.31, -2.25]
- **H4** (cross-site vs fixed-site): supported. write-off $1,090 vs $2,597; backlog variance 1.01 vs 3.16; utilization CV 0.48 vs 0.49
- **H5** (explanations vs none (pharmacist override rate)): not testable here. requires a user study with pharmacists under institutional review; not testable in simulation

## Paired per-order differences (IMDSOC minus strategy, 95% bootstrap CI)

| b | metric | n | mean_diff | ci_low | ci_high |
|---|---|---|---|---|---|
| fixed_site | realized_cost | 1470 | -169.463 | -241.268 | -103.279 |
| fixed_site | on_time | 1470 | -0.004 | -0.011 | 0.003 |
| nearest_eligible | realized_cost | 1470 | -176.331 | -251.112 | -110.805 |
| nearest_eligible | on_time | 1470 | 0.001 | -0.006 | 0.008 |
| lowest_shipping_cost | realized_cost | 1470 | -404.625 | -481.859 | -331.574 |
| lowest_shipping_cost | on_time | 1470 | 0.044 | 0.032 | 0.056 |
| inventory_only | realized_cost | 1471 | -224.943 | -305.779 | -148.753 |
| inventory_only | on_time | 1471 | 0.003 | -0.005 | 0.010 |
| rules_only | realized_cost | 1471 | -324.001 | -405.414 | -252.139 |
| rules_only | on_time | 1471 | 0.058 | 0.045 | 0.071 |
| abl_no_consolidation | realized_cost | 1472 | -2.909 | -8.977 | 3.167 |
| abl_no_consolidation | on_time | 1472 | -0.003 | -0.007 | 0.002 |
| abl_point_estimate | realized_cost | 1472 | 2.399 | -0.896 | 6.435 |
| abl_point_estimate | on_time | 1472 | -0.004 | -0.011 | 0.002 |
| abl_no_transit_model | realized_cost | 1472 | 1.210 | -2.206 | 5.286 |
| abl_no_transit_model | on_time | 1472 | 0.019 | 0.010 | 0.029 |
| abl_no_excursion_model | realized_cost | 1472 | -16.404 | -42.269 | 6.691 |
| abl_no_excursion_model | on_time | 1472 | 0.001 | -0.005 | 0.007 |
| abl_no_failed_model | realized_cost | 1472 | -8.287 | -19.015 | -0.992 |
| abl_no_failed_model | on_time | 1472 | -0.001 | -0.005 | 0.001 |

## Rural vs urban (RUCA >= 4 is rural)

| strategy | on_time_urban | on_time_rural | on_time_gap_rural_minus_urban | cost_urban | cost_rural | cost_gap_rural_minus_urban |
|---|---|---|---|---|---|---|
| abl_no_consolidation | 0.993 | 0.973 | -0.020 | 58.306 | 58.058 | -0.248 |
| abl_no_excursion_model | 0.991 | 0.966 | -0.025 | 72.673 | 69.530 | -3.144 |
| abl_no_failed_model | 0.990 | 0.975 | -0.015 | 61.759 | 67.910 | 6.151 |
| abl_no_transit_model | 0.966 | 0.964 | -0.002 | 52.970 | 56.768 | 3.797 |
| abl_point_estimate | 0.989 | 0.986 | -0.003 | 52.660 | 53.537 | 0.877 |
| fixed_site | 0.990 | 0.984 | -0.006 | 173.634 | 343.788 | 170.154 |
| imdsoc | 0.990 | 0.971 | -0.020 | 55.785 | 54.250 | -1.535 |
| inventory_only | 0.988 | 0.966 | -0.022 | 248.875 | 353.064 | 104.189 |
| lowest_shipping_cost | 0.950 | 0.916 | -0.034 | 447.139 | 489.773 | 42.634 |
| nearest_eligible | 0.983 | 0.982 | -0.002 | 181.036 | 349.413 | 168.377 |
| rules_only | 0.932 | 0.914 | -0.018 | 340.181 | 470.172 | 129.991 |

## Stress scenarios (discrete-event simulation)

| scenario | strategy | cost_per_rx | on_time_rate | excursion_rate | missed_pickup_rate | escalation_rate |
|---|---|---|---|---|---|---|
| baseline | imdsoc | 48.721 | 0.985 | 0.026 | 0.048 | 0.060 |
| baseline | fixed_site | 256.947 | 0.991 | 0.095 | 0.037 | 0.016 |
| baseline | nearest_eligible | 264.569 | 0.989 | 0.110 | 0.069 | 0.016 |
| carrier_delay_+30% | imdsoc | 66.367 | 0.970 | 0.094 | 0.048 | 0.060 |
| carrier_delay_+30% | fixed_site | 570.277 | 0.965 | 0.240 | 0.037 | 0.016 |
| carrier_delay_+30% | nearest_eligible | 574.386 | 0.961 | 0.232 | 0.069 | 0.016 |
| extreme_weather | imdsoc | 55.615 | 0.969 | 0.045 | 0.048 | 0.060 |
| extreme_weather | fixed_site | 424.652 | 0.965 | 0.179 | 0.037 | 0.016 |
| extreme_weather | nearest_eligible | 431.081 | 0.965 | 0.190 | 0.069 | 0.016 |
| heat_wave_+6C | imdsoc | 160.046 | 0.987 | 0.064 | 0.031 | 0.089 |
| heat_wave_+6C | fixed_site | 473.319 | 0.991 | 0.194 | 0.037 | 0.016 |
| heat_wave_+6C | nearest_eligible | 476.368 | 0.989 | 0.194 | 0.069 | 0.016 |
| fuel_surcharge_25% | imdsoc | 50.684 | 0.985 | 0.026 | 0.057 | 0.060 |
| fuel_surcharge_25% | fixed_site | 259.578 | 0.991 | 0.095 | 0.037 | 0.016 |
| fuel_surcharge_25% | nearest_eligible | 267.197 | 0.989 | 0.110 | 0.069 | 0.016 |
| demand_surge_+40% | imdsoc | 76.963 | 0.942 | 0.026 | 0.278 | 0.048 |
| demand_surge_+40% | fixed_site | 211.881 | 0.955 | 0.073 | 0.261 | 0.005 |
| demand_surge_+40% | nearest_eligible | 212.115 | 0.947 | 0.073 | 0.295 | 0.007 |

## Shadow mode

118 orders logged; IMDSOC agreed with the incumbent site on 57.6%. See shadow_mode.csv.
