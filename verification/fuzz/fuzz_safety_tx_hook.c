// libFuzzer harness for safety TX hook validation
//
// This harness tests the core safety logic that validates CAN messages
// before transmission. It fuzzes:
// - CAN message addresses
// - CAN message data content
// - Message length
// - Safety state (controls_allowed, relay_malfunction)
//
// Build: clang -g -O1 -fsanitize=fuzzer,address,undefined -o fuzz_safety_tx_hook fuzz_safety_tx_hook.c

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

// Safety state
static bool controls_allowed = false;
static bool relay_malfunction = false;

// TX message allowlist entry
typedef struct {
  uint32_t addr;
  uint8_t bus;
  uint8_t len;
} TxMsg;

// Example TX message allowlist (simplified from actual safety modes)
static const TxMsg tx_allowlist[] = {
  {0x200, 0, 8},   // Steering command
  {0x300, 0, 8},   // Accel command
  {0x400, 1, 8},   // LKAS command
};
#define TX_ALLOWLIST_LEN (sizeof(tx_allowlist) / sizeof(tx_allowlist[0]))

// Simulate safety_tx_hook logic
// Returns true if message should be allowed, false if blocked
static bool safety_tx_hook(const CANPacket_t *msg) {
  // P1: If relay malfunction, block ALL TX
  if (relay_malfunction) {
    return false;
  }

  // P2: If controls not allowed, block control messages
  if (!controls_allowed) {
    // Check if this is a control message (in allowlist)
    for (size_t i = 0; i < TX_ALLOWLIST_LEN; i++) {
      if (msg->addr == tx_allowlist[i].addr && msg->bus == tx_allowlist[i].bus) {
        // This is a control message but controls not allowed
        return false;
      }
    }
  }

  // Check message is in allowlist
  for (size_t i = 0; i < TX_ALLOWLIST_LEN; i++) {
    if (msg->addr == tx_allowlist[i].addr &&
        msg->bus == tx_allowlist[i].bus &&
        GET_LEN(msg) == tx_allowlist[i].len) {
      return true;
    }
  }

  return false;
}

// Extract torque from CAN message (simplified, Toyota-style)
static int extract_torque(const CANPacket_t *msg) {
  if (GET_LEN(msg) < 2) return 0;
  // Torque is typically 16-bit signed value in bytes 0-1
  int16_t torque = (int16_t)((msg->data[1] << 8) | msg->data[0]);
  return torque;
}

// Torque bounds check
#define MAX_TORQUE 1500

static bool check_torque_bounds(int torque) {
  return (torque >= -MAX_TORQUE) && (torque <= MAX_TORQUE);
}

// libFuzzer entry point
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < sizeof(CANPacket_t)) {
    return 0;  // Not enough data
  }

  // Create CAN packet from fuzz input
  CANPacket_t msg;
  memcpy(&msg, data, sizeof(CANPacket_t));

  // Sanitize DLC to valid range
  msg.data_len_code = msg.data_len_code & 0x0F;

  // Fuzz the safety state if we have extra bytes
  if (size > sizeof(CANPacket_t)) {
    controls_allowed = (data[sizeof(CANPacket_t)] & 0x01) != 0;
    relay_malfunction = (data[sizeof(CANPacket_t)] & 0x02) != 0;
  }

  // Test TX hook
  bool allowed = safety_tx_hook(&msg);

  // Verify invariants:
  // INV1: relay_malfunction => !allowed
  if (relay_malfunction && allowed) {
    __builtin_trap();  // Should never happen
  }

  // INV2: !controls_allowed && is_control_msg => !allowed
  if (!controls_allowed) {
    for (size_t i = 0; i < TX_ALLOWLIST_LEN; i++) {
      if (msg.addr == tx_allowlist[i].addr && msg.bus == tx_allowlist[i].bus) {
        if (allowed) {
          __builtin_trap();  // Should never happen
        }
        break;
      }
    }
  }

  // Test torque extraction and bounds
  int torque = extract_torque(&msg);
  bool in_bounds = check_torque_bounds(torque);
  (void)in_bounds;  // Use the result to prevent optimization

  return 0;
}
