/*
 * Test entrypoint for MuJoCo simulation experiments.
 * This keeps the default rl_sim_mujoco.cpp untouched.
 */

#define RL_MUJOCO_TEST_CSV
#define CSV_LOGGER
#include "rl_sim_mujoco.cpp"
