import numpy as np

# SWEEP_CONFIGS = [
#     {
#         "run_name": f"fpr_{fpr}_dfr_{dfr}",
#         "false_positive_rate": fpr,
#         "detection_failure_rate": dfr
#     }
#     for fpr in np.arange(0.0, 0.05, 0.01)
#     for dfr in np.arange(0.0, 0.61, 0.12)
# ]

# for 2025.09.21 grid search exp
# SWEEP_CONFIGS = [
#     {
#         "run_name": f"bao_vd_{vd}_lmp_{large_move_penalty}_ltp_{large_turn_penalty}_fdr_10_run_{run_num}",
#         "vergence_deviation": vd,
#         "food_detection_range": 10,
#         "large_move_penalty": large_move_penalty,
#         "large_turn_penalty": large_turn_penalty,
#         "seed": run_num
#     }
#     for run_num in range(3)
#     for vd in [0.0, -0.006, -0.012]
#     for large_move_penalty in [0.0, -0.01, -0.02]
#     for large_turn_penalty in [0.0, -0.01, -0.02]
# ]

# run_num = 0
# binocular_angle_flag = True
# binocular_depth_flag = False
# for binocular_depth_flag in [True, False]:
#     SWEEP_CONFIGS.append({
#         "run_name": f"bao_vd_{default_vd}_lmp_{default_large_move_penalty}_ltp_{default_large_turn_penalty}_fdr_10_bd_{int(binocular_depth_flag)}_ba_{int(binocular_angle_flag)}_run_{run_num}",
#         "vergence_deviation": default_vd,
#         "food_detection_range": 10,
#         "large_move_penalty": default_large_move_penalty,
#         "large_turn_penalty": default_large_turn_penalty,
#         "binocular_depth_only": binocular_depth_flag,
#         "binocular_angle_only": binocular_angle_flag,
#     })

SWEEP_CONFIGS = []

num_seeds = 10

default_vd = -0.006
default_large_move_penalty = -0.01
default_large_turn_penalty = -0.01

multipliers = np.array([0, 1, 2])
vd_options = default_vd * multipliers
multipliers_no_control = np.array([0, 2])
lmp_options = default_large_move_penalty * multipliers_no_control
ltp_options = default_large_turn_penalty * multipliers_no_control
default_binocular_depth_flag = True
default_binocular_angle_flag = False
default_num_walkerbots = 3
angle_noise_std_food = 0.21

for run_num in range(num_seeds):
    # # single one-off baseline run (10)
    # SWEEP_CONFIGS.append({
    #     "run_name": f"bao_vd_{default_vd}_lmp_{default_large_move_penalty}_ltp_{default_large_turn_penalty}_fdr_10_wb_{default_num_walkerbots}_run_{run_num}",
    #     "vergence_deviation": default_vd,
    #     "food_detection_range": 10,
    #     "large_move_penalty": default_large_move_penalty,
    #     "large_turn_penalty": default_large_turn_penalty,
    #     "binocular_depth_only": default_binocular_depth_flag,
    #     "binocular_angle_only": default_binocular_angle_flag,
    #     "walker_monocular_perception": True,
    #     "walker_pursuit": True,
    #     "angle_noise_std_food": angle_noise_std_food,
    #     "num_walkerbots": default_num_walkerbots,
    # })

    # for vd in vd_options:
    #     SWEEP_CONFIGS.append({
    #         "run_name": f"bao_vd_{vd}_lmp_{default_large_move_penalty}_ltp_{default_large_turn_penalty}_fdr_10_wb_{default_num_walkerbots}_run_{run_num}",
    #         "vergence_deviation": vd,
    #         "food_detection_range": 10,
    #         "large_move_penalty": default_large_move_penalty,
    #         "large_turn_penalty": default_large_turn_penalty,
    #         "binocular_depth_only": default_binocular_depth_flag,
    #         "binocular_angle_only": default_binocular_angle_flag,
    #         "walker_monocular_perception": True,
    #         "walker_pursuit": True,
    #         "angle_noise_std_food": angle_noise_std_food,
    #         "seed": run_num,
    #     })

    for lmp in lmp_options:
        SWEEP_CONFIGS.append({
            "run_name": f"bao_vd_{default_vd}_lmp_{lmp}_ltp_{default_large_turn_penalty}_fdr_10_wb_{default_num_walkerbots}_run_{run_num}",
            "vergence_deviation": default_vd,
            "food_detection_range": 10,
            "large_move_penalty": lmp,
            "large_turn_penalty": default_large_turn_penalty,
            "binocular_depth_only": default_binocular_depth_flag,
            "binocular_angle_only": default_binocular_angle_flag,
            "walker_monocular_perception": True,
            "walker_pursuit": True,
            "angle_noise_std_food": angle_noise_std_food,
            "seed": run_num,
        })

    # for ltp in ltp_options:
    #     SWEEP_CONFIGS.append({
    #         "run_name": f"bao_vd_{default_vd}_lmp_{default_large_move_penalty}_ltp_{ltp}_fdr_10_wb_{default_num_walkerbots}_run_{run_num}",
    #         "vergence_deviation": default_vd,
    #         "food_detection_range": 10,
    #         "large_move_penalty": default_large_move_penalty,
    #         "large_turn_penalty": ltp,
    #         "binocular_depth_only": default_binocular_depth_flag,
    #         "binocular_angle_only": default_binocular_angle_flag,
    #         "walker_monocular_perception": True,
    #         "walker_pursuit": True,
    #         "angle_noise_std_food": angle_noise_std_food,
    #         "seed": run_num,
    #     })

    # for num_walkerbots in [0, 1, 2, 4]:
    #     for vd in [0, default_vd]:
    #         SWEEP_CONFIGS.append({
    #             "run_name": f"bao_vd_{default_vd}_lmp_{default_large_move_penalty}_ltp_{default_large_turn_penalty}_fdr_10_wb_{num_walkerbots}_run_{run_num}",
    #             "vergence_deviation": vd,
    #             "food_detection_range": 10,
    #             "large_move_penalty": default_large_move_penalty,
    #             "large_turn_penalty": default_large_turn_penalty,
    #             "num_walkerbots": num_walkerbots,
    #             "walker_pursuit": True,
    #             "binocular_depth_only": default_binocular_depth_flag,
    #             "binocular_angle_only": default_binocular_angle_flag,
    #             "walker_monocular_perception": True,
    #             "angle_noise_std_food": angle_noise_std_food,
    #             "seed": run_num,
    #         })

    # for binocular_depth_only in [True, False]:
    #     for binocular_angle_only in [True, False]:
    #         SWEEP_CONFIGS.append({
    #             "run_name": f"bao_vd_{default_vd}_lmp_{default_large_move_penalty}_ltp_{default_large_turn_penalty}_fdr_10_bd_{int(binocular_depth_only)}_ba_{int(binocular_angle_only)}_run_{run_num}",
    #             "vergence_deviation": default_vd,
    #             "food_detection_range": 10,
    #             "large_move_penalty": default_large_move_penalty,
    #             "large_turn_penalty": default_large_turn_penalty,
    #             "binocular_depth_only": binocular_depth_only,
    #             "binocular_angle_only": binocular_angle_only,
    #             "seed": run_num,
    #         })

print(len(SWEEP_CONFIGS))

# SWEEP_CONFIGS = [
#     {
#         "run_name": "fmo_vergence_deviation_0",
#         "vergence_deviation": 0.0,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.001",
#         "vergence_deviation": -0.001,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.002",
#         "vergence_deviation": -0.002,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.003",
#         "vergence_deviation": -0.003,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.004",
#         "vergence_deviation": -0.004,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.005",
#         "vergence_deviation": -0.005,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.006",
#         "vergence_deviation": -0.006,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.007",
#         "vergence_deviation": -0.007,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.008",
#         "vergence_deviation": -0.008,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.009",
#         "vergence_deviation": -0.009,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.01",
#         "vergence_deviation": -0.01,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.012",
#         "vergence_deviation": -0.012,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.015",
#         "vergence_deviation": -0.015,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.02",
#         "vergence_deviation": -0.02,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.03",
#         "vergence_deviation": -0.03,
#     },
#     {
#         "run_name": "fmo_vergence_deviation_0.05",
#         "vergence_deviation": -0.05,
#     }
# ]
