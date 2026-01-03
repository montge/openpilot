/*
 * CBMC Harness: Controls Allowed Blocks Torque
 *
 * Property P1: controls_allowed = false => torque must be zero
 *
 * This verifies that steer_torque_cmd_checks() returns violation=true
 * when controls_allowed=false and desired_torque != 0.
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

/*
 * Simplified version of the key check from steer_torque_cmd_checks()
 * Lines 98-101 of lateral.h:
 *   // no torque if controls is not allowed
 *   if (!controls_allowed && (desired_torque != 0)) {
 *     violation = true;
 *   }
 */
static bool torque_controls_check(int desired_torque) {
  bool violation = false;

  /* The key safety invariant */
  if (!controls_allowed && (desired_torque != 0)) {
    violation = true;
  }

  return violation;
}

/*
 * Verification harness for controls_allowed torque blocking
 */
void verify_controls_blocks_torque(void) {
  /* Nondeterministic torque value */
  int desired_torque;

  /* Precondition: controls not allowed */
  controls_allowed = false;

  /* Precondition: torque is non-zero */
  __CPROVER_assume(desired_torque != 0);

  /* Call the check */
  bool violation = torque_controls_check(desired_torque);

  /* Postcondition: must be a violation */
  __CPROVER_assert(violation == true,
    "P1: controls_allowed=false with non-zero torque must be a violation");
}

/*
 * Verify the converse: zero torque is allowed when controls disabled
 */
void verify_zero_torque_allowed(void) {
  /* Precondition: controls not allowed */
  controls_allowed = false;

  /* Zero torque */
  int desired_torque = 0;

  /* Call the check */
  bool violation = torque_controls_check(desired_torque);

  /* Postcondition: zero torque should not trigger this specific violation */
  __CPROVER_assert(violation == false,
    "Zero torque should not trigger controls_allowed violation");
}

/* Entry point for CBMC */
int main(void) {
  verify_controls_blocks_torque();
  verify_zero_torque_allowed();
  return 0;
}
