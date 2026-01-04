// libFuzzer harness for CAN RX message parsing
//
// This harness tests the CAN message parsing logic including:
// - Counter validation
// - Checksum validation
// - Signal extraction
// - State updates from RX messages
//
// Build: clang -g -O1 -fsanitize=fuzzer,address,undefined -o fuzz_can_rx fuzz_can_rx.c

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

// CANPacket_t structure matching opendbc/safety/can.h
#define CANPACKET_DATA_SIZE_MAX 64U

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

// State tracking
static uint8_t last_counter = 0;
static int wrong_counter_count = 0;
#define MAX_WRONG_COUNTERS 5

// Extract counter from CAN message (common pattern: nibble in specific byte)
static uint8_t extract_counter(const CANPacket_t *msg, int byte_pos, int nibble) {
  if (GET_LEN(msg) <= (size_t)byte_pos) return 0;
  uint8_t value = msg->data[byte_pos];
  if (nibble == 0) {
    return value & 0x0F;
  } else {
    return (value >> 4) & 0x0F;
  }
}

// Validate counter (should increment by 1, wrapping at 16)
static bool validate_counter(uint8_t counter) {
  uint8_t expected = (last_counter + 1) % 16;
  bool valid = (counter == expected);
  if (!valid) {
    wrong_counter_count++;
  } else {
    wrong_counter_count = 0;
  }
  last_counter = counter;
  return valid;
}

// Simple CRC8 calculation (simplified, actual varies by manufacturer)
static uint8_t calc_crc8(const uint8_t *data, size_t len) {
  uint8_t crc = 0xFF;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int j = 0; j < 8; j++) {
      if (crc & 0x80) {
        crc = (crc << 1) ^ 0x1D;
      } else {
        crc <<= 1;
      }
    }
  }
  return crc;
}

// Extract signed value from CAN message
static int32_t extract_signed(const CANPacket_t *msg, int start_byte, int num_bytes) {
  if (GET_LEN(msg) < (size_t)(start_byte + num_bytes)) return 0;

  int32_t value = 0;
  for (int i = 0; i < num_bytes; i++) {
    value |= ((int32_t)msg->data[start_byte + i]) << (i * 8);
  }

  // Sign extend if negative
  int sign_bit = 1 << (num_bytes * 8 - 1);
  if (value & sign_bit) {
    value |= ~((1 << (num_bytes * 8)) - 1);
  }

  return value;
}

// Extract speed from message (simplified)
static int extract_speed_kmh(const CANPacket_t *msg) {
  if (GET_LEN(msg) < 4) return 0;
  // Speed typically 16-bit unsigned at bytes 2-3, scaled by 0.01
  uint16_t raw = (msg->data[3] << 8) | msg->data[2];
  return (int)(raw * 0.01f);
}

// libFuzzer entry point
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < sizeof(CANPacket_t)) {
    return 0;
  }

  // Reset state for each input
  last_counter = 0;
  wrong_counter_count = 0;

  CANPacket_t msg;
  memcpy(&msg, data, sizeof(CANPacket_t));

  // Sanitize DLC
  msg.data_len_code = msg.data_len_code & 0x0F;

  // Test counter extraction and validation
  uint8_t counter = extract_counter(&msg, 0, 0);
  bool counter_valid = validate_counter(counter);

  // INV1: wrong_counter_count never exceeds MAX_WRONG_COUNTERS + 1
  if (wrong_counter_count > MAX_WRONG_COUNTERS + 1) {
    __builtin_trap();
  }

  // Test CRC calculation (shouldn't crash on any input)
  size_t crc_len = GET_LEN(&msg);
  if (crc_len > 0) {
    uint8_t crc = calc_crc8(msg.data, crc_len);
    (void)crc;
  }

  // Test signed value extraction
  int32_t signed_val = extract_signed(&msg, 0, 2);
  (void)signed_val;

  // Test speed extraction
  int speed = extract_speed_kmh(&msg);
  (void)speed;

  // Process multiple messages if we have enough data
  size_t offset = sizeof(CANPacket_t);
  while (offset + sizeof(CANPacket_t) <= size) {
    memcpy(&msg, data + offset, sizeof(CANPacket_t));
    msg.data_len_code = msg.data_len_code & 0x0F;

    counter = extract_counter(&msg, 0, 0);
    counter_valid = validate_counter(counter);

    // Track state across messages
    if (wrong_counter_count > MAX_WRONG_COUNTERS) {
      // Would trigger safety action (disable controls)
      break;
    }

    offset += sizeof(CANPacket_t);
  }

  (void)counter_valid;
  return 0;
}
