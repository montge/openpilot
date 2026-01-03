/*
 * CBMC Harness: Longitudinal Acceleration Bounds
 *
 * Property P4: Acceleration commands respect min/max limits
 *
 * This verifies that longitudinal_accel_checks() properly enforces
 * acceleration limits and inactive state requirements.
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

/* Toyota-like limits (scaled by 1000, so 2.0 m/s^2 = 2000) */
#define ACCEL_MAX 2000   /* 2.0 m/s^2 */
#define ACCEL_MIN -3500  /* -3.5 m/s^2 */
#define ACCEL_INACTIVE 0

/*
 * Inline get_longitudinal_allowed() from longitudinal.h
 */
static bool get_longitudinal_allowed_local(void) {
  return controls_allowed && !gas_pressed_prev;
}

/*
 * Inline longitudinal_accel_checks() from longitudinal.h
 */
static bool longitudinal_accel_checks_local(int desired_accel, const LongitudinalLimits limits) {
  bool accel_valid = get_longitudinal_allowed_local() &&
                     !safety_max_limit_check(desired_accel, limits.max_accel, limits.min_accel);
  bool accel_inactive = desired_accel == limits.inactive_accel;
  return !(accel_valid || accel_inactive);
}

/*
 * Verify that when controls are allowed, accel must be within limits
 */
void verify_accel_bounds_when_allowed(void) {
  /* Set up state: controls allowed, no gas pressed */
  controls_allowed = true;
  gas_pressed_prev = false;

  /* Nondeterministic acceleration */
  int desired_accel;

  /* Limits */
  LongitudinalLimits limits = {
    .max_accel = ACCEL_MAX,
    .min_accel = ACCEL_MIN,
    .inactive_accel = ACCEL_INACTIVE,
  };

  /* Check acceleration */
  bool violation = longitudinal_accel_checks_local(desired_accel, limits);

  /* THE KEY INVARIANT:
   * If no violation, accel must be within bounds OR be inactive value
   */
  __CPROVER_assert(
    violation ||
    (desired_accel >= ACCEL_MIN && desired_accel <= ACCEL_MAX) ||
    (desired_accel == ACCEL_INACTIVE),
    "P4: Accepted accel must be within [min, max] or inactive"
  );
}

/*
 * Verify that controls_allowed=false only allows inactive accel
 */
void verify_accel_blocked_when_not_allowed(void) {
  /* Controls not allowed */
  controls_allowed = false;
  gas_pressed_prev = false;

  /* Non-inactive acceleration */
  int desired_accel;
  __CPROVER_assume(desired_accel != ACCEL_INACTIVE);

  LongitudinalLimits limits = {
    .max_accel = ACCEL_MAX,
    .min_accel = ACCEL_MIN,
    .inactive_accel = ACCEL_INACTIVE,
  };

  bool violation = longitudinal_accel_checks_local(desired_accel, limits);

  /* Must be a violation */
  __CPROVER_assert(violation == true,
    "Non-inactive accel must be rejected when controls not allowed");
}

/*
 * Verify inactive accel is always allowed
 */
void verify_inactive_always_allowed(void) {
  /* Nondeterministic controls state */
  bool nondet_controls;
  bool nondet_gas;
  controls_allowed = nondet_controls;
  gas_pressed_prev = nondet_gas;

  /* Inactive acceleration */
  int desired_accel = ACCEL_INACTIVE;

  LongitudinalLimits limits = {
    .max_accel = ACCEL_MAX,
    .min_accel = ACCEL_MIN,
    .inactive_accel = ACCEL_INACTIVE,
  };

  bool violation = longitudinal_accel_checks_local(desired_accel, limits);

  /* Inactive should always pass */
  __CPROVER_assert(violation == false,
    "Inactive accel value must always be allowed");
}

/*
 * Verify gas_pressed blocks longitudinal commands
 */
void verify_gas_blocks_accel(void) {
  /* Controls allowed but gas pressed */
  controls_allowed = true;
  gas_pressed_prev = true;

  /* Valid acceleration within bounds (not inactive) */
  int desired_accel;
  __CPROVER_assume(desired_accel >= ACCEL_MIN && desired_accel <= ACCEL_MAX);
  __CPROVER_assume(desired_accel != ACCEL_INACTIVE);

  LongitudinalLimits limits = {
    .max_accel = ACCEL_MAX,
    .min_accel = ACCEL_MIN,
    .inactive_accel = ACCEL_INACTIVE,
  };

  bool violation = longitudinal_accel_checks_local(desired_accel, limits);

  /* Should be a violation because gas is pressed */
  __CPROVER_assert(violation == true,
    "Gas pressed must block longitudinal commands");
}

/* Entry point for CBMC */
int main(void) {
  verify_accel_bounds_when_allowed();
  verify_accel_blocked_when_not_allowed();
  verify_inactive_always_allowed();
  verify_gas_blocks_accel();
  return 0;
}
