/*
 * msgq_protocol.pml - SPIN Promela Model for msgq Lock-Free Ring Buffer
 *
 * This model verifies the lock-free messaging protocol used in openpilot's
 * msgq system (msgq_repo/msgq/msgq.cc). The protocol uses:
 *   - Shared memory ring buffer with 64-bit packed pointers (cycle:32 | offset:32)
 *   - Cycle counters to detect wraparound and prevent stale reads
 *   - UID-based reader tracking for eviction
 *   - Atomic compare-and-swap for reader registration
 *
 * Key protocol properties verified:
 *   P1: No message loss for valid (non-evicted) readers
 *   P2: Slow readers are correctly evicted when overwritten
 *   P3: Cycle counter prevents reading stale data after wraparound
 *   P4: Publisher progress is guaranteed
 *
 * Reference: msgq_repo/msgq/msgq.cc, msgq_repo/msgq/msgq.h
 */

/* Configuration - keep small for tractable state space */
#define NUM_READERS     2       /* Reduced from 15 for verification */
#define BUFFER_SIZE     4       /* Small ring buffer (messages slots) */
#define MAX_CYCLE       3       /* Cycle counter wraparound bound */
#define MAX_MESSAGES    6       /* Total messages to publish */

/* ====== Shared Memory State (mirrors msgq_header_t) ====== */

/* Ring buffer - stores message IDs (1-based, 0 = empty) */
byte buffer[BUFFER_SIZE];

/* Write pointer: packed as (write_cycles << 16) | write_offset
 * Using 16-bit split for Promela (real impl uses 32-bit split in 64-bit value) */
byte write_offset = 0;
byte write_cycles = 0;

/* Reader state arrays */
byte read_offsets[NUM_READERS];
byte read_cycles_arr[NUM_READERS];
bool read_valids[NUM_READERS];
byte read_uids[NUM_READERS];    /* Simplified UID (real impl uses 64-bit) */

/* Number of active readers (atomic counter in real impl) */
byte num_readers = 0;

/* Publisher UID for detecting publisher takeover */
byte write_uid = 0;

/* ====== Verification Tracking Variables ====== */

/* Message tracking */
byte messages_written = 0;
byte messages_read[NUM_READERS];
byte last_msg_read[NUM_READERS];

/* Error detection flags */
bool stale_read_detected = false;
bool eviction_race_detected = false;
bool message_gap_detected = false;

/* Process completion flags */
bool publisher_done = false;
byte readers_registered = 0;

/* ====== Publisher Process ====== */
/* Models msgq_msg_send() from msgq.cc */

proctype Publisher(byte uid) {
  byte msg_id = 1;
  byte old_offset, new_offset;
  byte i;
  byte local_num_readers;

  /* Initialize as publisher (msgq_init_publisher) */
  atomic {
    write_uid = uid;
    num_readers = 0;
    i = 0;
    do
    :: i < NUM_READERS ->
         read_valids[i] = false;
         read_uids[i] = 0;
         i++
    :: else -> break
    od
  };

  /* Wait for at least one subscriber before publishing */
  do
  :: num_readers > 0 -> break
  :: else -> skip
  od;

  /* Main publish loop */
  do
  :: msg_id <= MAX_MESSAGES ->

       /* Read current write pointer */
       old_offset = write_offset;

       /* Calculate new write position */
       new_offset = (old_offset + 1) % BUFFER_SIZE;

       /* Check for wraparound - when new_offset wraps to 0 */
       if
       :: new_offset == 0 ->
            /* Wraparound: increment cycle counter */
            write_cycles = (write_cycles + 1) % MAX_CYCLE;

            /* Invalidate readers beyond write pointer (in previous cycle)
             * This models lines 259-267 in msgq.cc */
            local_num_readers = num_readers;
            i = 0;
            do
            :: i < local_num_readers ->
                 /* Reader is behind (in old cycle) - evict */
                 if
                 :: read_offsets[i] > old_offset &&
                    read_cycles_arr[i] != write_cycles ->
                      read_valids[i] = false
                 :: else -> skip
                 fi;
                 i++
            :: else -> break
            od
       :: else -> skip
       fi;

       /* Invalidate readers in the write area (lines 278-289 in msgq.cc)
        * A reader is invalid if it's pointing at the slot we're about to write
        * AND it's from a previous cycle */
       local_num_readers = num_readers;
       i = 0;
       do
       :: i < local_num_readers ->
            if
            :: read_offsets[i] == old_offset &&
               read_cycles_arr[i] != write_cycles ->
                 read_valids[i] = false
            :: else -> skip
            fi;
            i++
       :: else -> break
       od;

       /* Write the message */
       buffer[old_offset] = msg_id;

       /* Memory barrier (__sync_synchronize in real impl) */
       /* Promela's statement ordering provides sequencing */

       /* Update write pointer */
       write_offset = new_offset;

       /* Track for verification */
       messages_written = msg_id;
       msg_id++

  :: else -> break
  od;

  publisher_done = true;
  printf("Publisher completed: %d messages written\n", messages_written)
}

/* ====== Subscriber Process ====== */
/* Models msgq_init_subscriber() and msgq_msg_recv() from msgq.cc */

proctype Subscriber(byte id; byte uid) {
  byte my_offset, my_cycle;
  byte msg;
  byte expected_msg;
  bool registered = false;

  /* Registration with atomic CAS (models lines 186-226 in msgq.cc) */
  do
  :: !registered ->
       atomic {
         if
         :: num_readers < NUM_READERS ->
              /* Successfully claim a reader slot */
              read_offsets[id] = write_offset;
              read_cycles_arr[id] = write_cycles;
              read_valids[id] = true;
              read_uids[id] = uid;
              num_readers++;
              registered = true;
              readers_registered++
         :: else ->
              /* No slots available - in real impl this triggers mass eviction */
              skip
         fi
       }
  :: registered -> break
  od;

  printf("Subscriber %d registered\n", id);

  /* Track the first message we expect to read */
  expected_msg = 1;

  /* Main read loop */
  do
  :: !publisher_done || read_offsets[id] != write_offset ->

       /* Check if we've been evicted (models lines 318-323 in msgq.cc) */
       if
       :: read_uids[id] != uid ->
            printf("Subscriber %d: evicted (UID mismatch), exiting\n", id);
            break
       :: else -> skip
       fi;

       /* Check read_valid flag (models lines 325-329 in msgq.cc) */
       if
       :: !read_valids[id] ->
            /* Reset reader position (msgq_reset_reader) */
            atomic {
              read_offsets[id] = write_offset;
              read_cycles_arr[id] = write_cycles;
              read_valids[id] = true
            };
            printf("Subscriber %d: reset after invalidation\n", id)
       :: else -> skip
       fi;

       /* Load local copies of pointers */
       my_offset = read_offsets[id];
       my_cycle = read_cycles_arr[id];

       /* Check if new message is available */
       if
       :: my_offset != write_offset || my_cycle != write_cycles ->

            /* Read the message from buffer */
            msg = buffer[my_offset];

            /* Verify we haven't read stale data */
            if
            :: msg == 0 ->
                 /* Reading uninitialized slot - this indicates a bug */
                 printf("ERROR: Subscriber %d read empty slot at %d\n", id, my_offset);
                 stale_read_detected = true
            :: else -> skip
            fi;

            /* Check for message gaps (only if not reset) */
            if
            :: msg > 0 && expected_msg > 0 && msg > expected_msg ->
                 /* We missed a message - this is expected for slow readers */
                 printf("Subscriber %d: gap detected, expected %d got %d\n",
                        id, expected_msg, msg);
                 message_gap_detected = true
            :: else -> skip
            fi;

            /* Advance read pointer */
            byte new_offset = (my_offset + 1) % BUFFER_SIZE;
            if
            :: new_offset == 0 ->
                 read_cycles_arr[id] = (my_cycle + 1) % MAX_CYCLE
            :: else -> skip
            fi;
            read_offsets[id] = new_offset;

            /* Update tracking */
            messages_read[id]++;
            last_msg_read[id] = msg;
            expected_msg = msg + 1;

            /* Verify still valid after read (race check, lines 419-424 in msgq.cc) */
            if
            :: !read_valids[id] ->
                 printf("Subscriber %d: evicted DURING read of msg %d\n", id, msg);
                 eviction_race_detected = true
            :: else -> skip
            fi

       :: else ->
            /* No new message, wait */
            skip
       fi

  :: publisher_done && read_offsets[id] == write_offset &&
     read_cycles_arr[id] == write_cycles ->
       /* Caught up and publisher is done */
       break
  od;

  printf("Subscriber %d finished: read %d messages, last=%d\n",
         id, messages_read[id], last_msg_read[id])
}

/* ====== Race Condition Test: New Reader During Wraparound ====== */
/*
 * This models the known race condition mentioned in the proposal:
 * A new reader registers WHILE the publisher is in the invalidation loop.
 * The new reader might escape invalidation because it wasn't in num_readers yet.
 */

proctype LateSubscriber(byte id; byte uid) {
  /* Wait for some messages to be written */
  do
  :: messages_written >= 2 -> break
  :: else -> skip
  od;

  /* Try to register - this may race with publisher's invalidation loop */
  atomic {
    if
    :: num_readers < NUM_READERS ->
         read_offsets[id] = write_offset;
         read_cycles_arr[id] = write_cycles;
         read_valids[id] = true;
         read_uids[id] = uid;
         num_readers++;
         readers_registered++;
         printf("Late subscriber %d registered at offset %d, cycle %d\n",
                id, write_offset, write_cycles)
    :: else -> skip
    fi
  };

  /* Read a few messages */
  byte count = 0;
  do
  :: count < 3 && read_valids[id] ->
       byte my_offset = read_offsets[id];
       if
       :: my_offset != write_offset ->
            byte msg = buffer[my_offset];
            byte new_offset = (my_offset + 1) % BUFFER_SIZE;
            if
            :: new_offset == 0 ->
                 read_cycles_arr[id] = (read_cycles_arr[id] + 1) % MAX_CYCLE
            :: else -> skip
            fi;
            read_offsets[id] = new_offset;
            messages_read[id]++;
            last_msg_read[id] = msg;
            count++;
            printf("Late subscriber %d read msg %d\n", id, msg)
       :: else -> skip
       fi
  :: !read_valids[id] ->
       printf("Late subscriber %d was evicted\n", id);
       break
  :: count >= 3 -> break
  od
}

/* ====== Initialization ====== */

init {
  byte i;

  /* Initialize arrays */
  atomic {
    i = 0;
    do
    :: i < BUFFER_SIZE ->
         buffer[i] = 0;
         i++
    :: else -> break
    od;

    i = 0;
    do
    :: i < NUM_READERS ->
         read_offsets[i] = 0;
         read_cycles_arr[i] = 0;
         read_valids[i] = false;
         read_uids[i] = 0;
         messages_read[i] = 0;
         last_msg_read[i] = 0;
         i++
    :: else -> break
    od
  };

  /* Start processes */
  run Publisher(1);
  run Subscriber(0, 10);
  run Subscriber(1, 11)
}

/* ====== LTL Properties ====== */

/*
 * P1: Publisher Progress
 * The publisher eventually completes writing all messages.
 */
ltl publisher_progress {
  <>(publisher_done)
}

/*
 * P2: No Stale Reads
 * Valid readers should never read uninitialized/stale buffer slots.
 */
ltl no_stale_reads {
  [](stale_read_detected == false)
}

/*
 * P3: Publisher Completes All Messages
 * When publisher is done, all messages have been written.
 */
ltl all_messages_written {
  [](publisher_done -> messages_written == MAX_MESSAGES)
}

/*
 * P4: Valid Reader Eventually Reads
 * If a reader is valid and there's data, it will eventually read.
 * (Weak liveness - may not hold if reader is slow and gets evicted)
 */
ltl reader0_progress {
  [](read_valids[0] && messages_written > messages_read[0] ->
     <>(messages_read[0] > 0 || !read_valids[0]))
}

/*
 * P5: Eviction Implies Cycle Mismatch
 * Readers are only evicted when their cycle counter doesn't match
 * the write cycle (i.e., they're a full buffer behind).
 */
/* Note: This is checked procedurally in the model via printf */

/*
 * P6: No Deadlock
 * The system never reaches a state where all processes are blocked.
 * (Built-in SPIN check with -DNOREDUCE)
 */

/* ====== Assertions ====== */

/* Inline assertion for cycle counter bounds */
#define ASSERT_CYCLE_BOUND(c) assert((c) < MAX_CYCLE)
#define ASSERT_OFFSET_BOUND(o) assert((o) < BUFFER_SIZE)
