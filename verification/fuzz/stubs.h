// Stubs for libFuzzer harnesses
// These provide minimal implementations for hardware-dependent functions

#ifndef FUZZ_STUBS_H
#define FUZZ_STUBS_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

// Stub for microsecond timer
static uint32_t fuzz_microseconds = 0;
static inline uint32_t microsecond_timer_get(void) {
  return fuzz_microseconds++;
}

// Define CANPacket_t to match can.h
#define CANPACKET_HEAD_SIZE 6U
#define CANPACKET_DATA_SIZE_MAX 64U
#define CAN_PACKET_VERSION 4

typedef struct {
  unsigned char fd : 1;
  unsigned char bus : 3;
  unsigned char data_len_code : 4;
  unsigned char rejected : 1;
  unsigned char returned : 1;
  unsigned char extended : 1;
  unsigned int addr : 29;
  unsigned char checksum;
  unsigned char data[CANPACKET_DATA_SIZE_MAX];
} __attribute__((packed, aligned(4))) CANPacket_t;

static const unsigned char dlc_to_len[] = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 12U, 16U, 20U, 24U, 32U, 48U, 64U};
#define GET_LEN(msg) (dlc_to_len[(msg)->data_len_code])

// Sample structure for measurements
struct sample_t {
  int values[6];
  int min;
  int max;
};

// Global state variables (matching safety.h)
extern bool controls_allowed;
extern bool relay_malfunction;
extern bool gas_pressed;
extern bool brake_pressed;
extern bool cruise_engaged_prev;

// Minimal stubs for functions not needed in fuzzing
static inline void update_sample(struct sample_t *sample, int value) {
  (void)sample;
  (void)value;
}

#endif // FUZZ_STUBS_H
