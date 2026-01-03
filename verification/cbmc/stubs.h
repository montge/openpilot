/*
 * CBMC Verification Stubs
 *
 * Hardware-dependent functions are stubbed out for model checking.
 * CBMC will use nondeterministic values where appropriate.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

/* Stub for microsecond timer - returns nondeterministic value */
static uint32_t _cbmc_timer_value;
static inline uint32_t microsecond_timer_get(void) {
  uint32_t nondet_time;
  /* Allow CBMC to explore different timing scenarios */
  return nondet_time;
}

/* CBMC-specific macros for verification */
#ifndef __CPROVER
#define __CPROVER_assert(cond, msg) do { if (!(cond)) __builtin_trap(); } while(0)
#define __CPROVER_assume(cond) do { if (!(cond)) __builtin_trap(); } while(0)
#define __CPROVER_havoc_object(ptr) do {} while(0)
#endif
