/*
 * CBMC Harness: Torque Bounds Verification
 *
 * Property P3: Accepted torque is within [-max_torque, +max_torque]
 *
 * This verifies that if steer_torque_cmd_checks() returns violation=false,
 * then the torque must be within the configured limits.
 */

#include "stubs.h"

/* Includes using full path from opendbc_repo */
#include "opendbc/safety/can.h"
#include "opendbc/safety/declarations.h"
#include "opendbc/safety/helpers.h"

/* Provide external definitions */
bool controls_allowed = false;
bool relay_malfunction = false;
bool gas_pressed = false;
bool gas_pressed_prev = false;
bool brake_pressed = false;
bool brake_pressed_prev = false;
bool regen_braking = false;
bool regen_braking_prev = false;
bool steering_disengage = false;
bool steering_disengage_prev = false;
bool cruise_engaged_prev = false;
struct sample_t vehicle_speed = {0};
bool vehicle_moving = false;
bool acc_main_on = false;
int cruise_button_prev = 0;
bool safety_rx_checks_invalid = false;
int desired_torque_last = 0;
int rt_torque_last = 0;
int valid_steer_req_count = 0;
int invalid_steer_req_count = 0;
struct sample_t torque_meas = {0};
struct sample_t torque_driver = {0};
uint32_t ts_torque_check_last = 0;
uint32_t ts_steer_req_mismatch_last = 0;
bool heartbeat_engaged = false;
uint32_t heartbeat_engaged_mismatches = 0;
uint32_t rt_angle_msgs = 0;
uint32_t ts_angle_check_last = 0;
int desired_angle_last = 0;
struct sample_t angle_meas = {0};
int alternative_experience = 0;
uint32_t safety_mode_cnt = 0U;
uint16_t current_safety_mode = 0;
uint16_t current_safety_param = 0;
safety_config current_safety_config = {0};

/* Toyota-like torque limits for verification */
#define TOYOTA_MAX_TORQUE 1500

/*
 * Simplified torque bounds check from lateral.h line 74:
 *   violation |= safety_max_limit_check(desired_torque, max_torque, -max_torque);
 */
static bool torque_bounds_check(int desired_torque, int max_torque) {
  return safety_max_limit_check(desired_torque, max_torque, -max_torque);
}

/*
 * Verification harness for torque bounds
 *
 * Proves: if torque passes the bounds check, it's within limits
 */
void verify_torque_bounds(void) {
  /* Nondeterministic torque value - full int16 range */
  int desired_torque;
  __CPROVER_assume(desired_torque >= -32768 && desired_torque <= 32767);

  /* Fixed max torque for verification */
  int max_torque = TOYOTA_MAX_TORQUE;

  /* Check if bounds are violated */
  bool violation = torque_bounds_check(desired_torque, max_torque);

  /* THE KEY INVARIANT:
   * If no violation, torque must be within bounds.
   * Contrapositive: outside bounds => violation
   */
  __CPROVER_assert(
    violation || (desired_torque >= -max_torque && desired_torque <= max_torque),
    "P3: Accepted torque must be within [-max_torque, +max_torque]"
  );
}

/*
 * Verify that out-of-bounds torque is always rejected
 */
void verify_out_of_bounds_rejected(void) {
  int desired_torque;
  int max_torque = TOYOTA_MAX_TORQUE;

  /* Precondition: torque is outside bounds */
  __CPROVER_assume(desired_torque > max_torque || desired_torque < -max_torque);

  /* Check bounds */
  bool violation = torque_bounds_check(desired_torque, max_torque);

  /* Postcondition: must be a violation */
  __CPROVER_assert(violation == true,
    "Out-of-bounds torque must always be rejected");
}

/*
 * Verify boundary conditions
 */
void verify_boundary_values(void) {
  int max_torque = TOYOTA_MAX_TORQUE;

  /* Exactly at max should pass */
  bool at_max = torque_bounds_check(max_torque, max_torque);
  __CPROVER_assert(at_max == false, "Torque exactly at max should pass");

  /* Exactly at -max should pass */
  bool at_min = torque_bounds_check(-max_torque, max_torque);
  __CPROVER_assert(at_min == false, "Torque exactly at -max should pass");

  /* One above max should fail */
  bool above_max = torque_bounds_check(max_torque + 1, max_torque);
  __CPROVER_assert(above_max == true, "Torque above max should fail");

  /* One below -max should fail */
  bool below_min = torque_bounds_check(-max_torque - 1, max_torque);
  __CPROVER_assert(below_min == true, "Torque below -max should fail");
}

/* Entry point for CBMC */
int main(void) {
  verify_torque_bounds();
  verify_out_of_bounds_rejected();
  verify_boundary_values();
  return 0;
}
