# Rollout metrics summary

Scaled MAE is dimensionless: each error is divided by the corresponding training-set
normalisation scale (`edge_feat_max` for position, `node_vel_max` for velocity,
`node_angvel_max` for angular velocity), matching `evaluate_rollout`.

| Case | Experiment | Steps | Wall clock (s) | Pos MAE mean | Pos MAE final | Vel MAE mean | Vel MAE final | AngVel MAE mean | AngVel MAE final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 04 (homogeneous) | benchmark_oblique_10deg | 199 | 4.6 | 1.2390e-01 | 2.7893e-01 | 3.1707e-02 | 3.5200e-02 | 5.1566e-02 | 5.6700e-02 |
| Case 04 (homogeneous) | benchmark_oblique_30deg | 199 | 3.8 | 5.0572e-02 | 1.1559e-01 | 1.3259e-02 | 1.4788e-02 | 1.2887e-02 | 1.4487e-02 |
| Case 04 (homogeneous) | benchmark_oblique_45deg | 199 | 3.9 | 3.3102e-02 | 7.7216e-02 | 8.9135e-03 | 1.0130e-02 | 6.4163e-03 | 7.2945e-03 |
| Case 04 (homogeneous) | benchmark_oblique_60deg | 199 | 4.2 | 2.6866e-02 | 6.5965e-02 | 7.5264e-03 | 8.9264e-03 | 7.0163e-03 | 8.3873e-03 |
| Case 04 (homogeneous) | benchmark_oblique_90deg | 199 | 3.8 | 3.8796e-02 | 9.1025e-02 | 1.0603e-02 | 1.1866e-02 | 6.3228e-04 | 6.0564e-04 |
| Case 04 (homogeneous) | benchmark_oblique_sphere_collisions | 99 | 9.2 | 1.0098e-02 | 2.9490e-02 | 6.8943e-03 | 1.0213e-02 | 2.2237e-02 | 3.3681e-02 |
| Case 04 (homogeneous) | case_07_rollout | 1499 | 138.3 | 1.7658e+00 | 2.5131e+00 | 1.3800e-01 | 5.5722e-02 | 1.5828e-01 | 8.7037e-02 |
| Case 05 (gravity) | case_06_gravity_rollout | 1498 | 169.3 | 6.7527e-01 | 8.7947e-01 | 8.9076e-02 | 4.6871e-02 | 8.6102e-02 | 4.9345e-02 |
| Case 05 (gravity) | rotating_cylinder_rollout | 1998 | 1257.0 | 2.2910e+00 | 2.3010e+00 | 4.1013e-02 | 4.1031e-02 | 3.6748e-02 | 4.7208e-02 |

## Oblique wall impact - final post-collision angular velocity

| Angle | Predicted (wx, wy, wz) | DEM ground truth (wx, wy, wz) | L2 error | GT L2 norm |
| ---: | --- | --- | ---: | ---: |
| 10° | +2.6371e+01, +1.6222e+02, -3.2339e+00 | +6.2211e-12, +2.5169e+02, +1.6528e-12 | 9.3329e+01 | 2.5169e+02 |
| 30° | +6.5117e+00, +2.2165e+02, +2.6892e+00 | +2.4953e-12, +2.4287e+02, +1.7707e-12 | 2.2362e+01 | 2.4287e+02 |
| 45° | +1.3830e+00, +1.9968e+02, +2.3029e+00 | +1.0431e-12, +2.1131e+02, +1.0378e-12 | 1.1939e+01 | 2.1131e+02 |
| 60° | +5.3147e+00, +1.5122e+02, +2.1859e+00 | +5.1357e-14, +1.6133e+02, +4.3819e-14 | 1.1632e+01 | 1.6133e+02 |
| 90° | +0.0000e+00, +0.0000e+00, -1.2719e+00 | +0.0000e+00, +0.0000e+00, +0.0000e+00 | 1.2719e+00 | 0.0000e+00 |
