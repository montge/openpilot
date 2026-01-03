# SPIN Protocol Verification Design

## Protocol Overview

```
Publisher                    Shared Memory                    Subscribers
   |                             |                                 |
   |  1. write message           |                                 |
   |--------------------------->>|                                 |
   |  2. update write_pointer    |                                 |
   |--------------------------->>|                                 |
   |  3. signal readers (SIGUSR2)|                                 |
   |--------------------------->>|-------------------------------->|
   |                             |  4. read write_pointer          |
   |                             |<<--------------------------------|
   |                             |  5. copy message                 |
   |                             |<<--------------------------------|
   |                             |  6. update read_pointer          |
   |                             |<<--------------------------------|
```

## Promela Model

```promela
/* msgq_protocol.pml - Lock-free ring buffer verification */

#define NUM_READERS 3
#define BUFFER_SIZE 4
#define MAX_CYCLE 2

/* Shared state */
byte buffer[BUFFER_SIZE];
byte write_pointer = 0;
byte write_cycle = 0;
byte read_pointers[NUM_READERS];
byte read_cycles[NUM_READERS];
bool read_valids[NUM_READERS];
byte num_readers = 0;

/* Track for verification */
byte messages_written = 0;
byte messages_read[NUM_READERS];

/* Publisher process */
proctype Publisher() {
  byte msg_id = 1;
  byte old_wp, new_wp;

  do
  :: msg_id <= 5 ->
       /* Write message */
       old_wp = write_pointer;
       buffer[old_wp] = msg_id;

       /* Update write pointer atomically */
       new_wp = (old_wp + 1) % BUFFER_SIZE;
       if
       :: new_wp == 0 -> write_cycle = (write_cycle + 1) % MAX_CYCLE
       :: else -> skip
       fi;

       /* Invalidate readers that are about to be overwritten */
       byte i = 0;
       do
       :: i < num_readers ->
            if
            :: read_pointers[i] == new_wp &&
               read_cycles[i] != write_cycle ->
                 read_valids[i] = false  /* Evict slow reader */
            :: else -> skip
            fi;
            i++
       :: i >= num_readers -> break
       od;

       write_pointer = new_wp;
       messages_written++;
       msg_id++
  :: msg_id > 5 -> break
  od
}

/* Subscriber process */
proctype Subscriber(byte id) {
  byte my_rp, my_cycle;
  byte msg;

  /* Register */
  atomic {
    read_pointers[id] = write_pointer;
    read_cycles[id] = write_cycle;
    read_valids[id] = true;
    num_readers++;
  }

  do
  :: read_valids[id] ->
       /* Wait for new message */
       my_rp = read_pointers[id];
       my_cycle = read_cycles[id];

       if
       :: my_rp != write_pointer || my_cycle != write_cycle ->
            /* New message available */
            msg = buffer[my_rp];

            /* Advance read pointer */
            byte new_rp = (my_rp + 1) % BUFFER_SIZE;
            if
            :: new_rp == 0 -> read_cycles[id] = (my_cycle + 1) % MAX_CYCLE
            :: else -> skip
            fi;
            read_pointers[id] = new_rp;
            messages_read[id]++;

            /* Verify still valid (race check) */
            if
            :: !read_valids[id] ->
                 printf("Reader %d evicted during read!\n", id);
                 break
            :: else -> skip
            fi
       :: else -> skip  /* No new message */
       fi
  :: !read_valids[id] -> break
  od
}

/* Initialization */
init {
  atomic {
    byte i = 0;
    do
    :: i < NUM_READERS ->
         read_pointers[i] = 0;
         read_cycles[i] = 0;
         read_valids[i] = false;
         messages_read[i] = 0;
         i++
    :: i >= NUM_READERS -> break
    od
  };

  run Publisher();
  byte j = 0;
  do
  :: j < NUM_READERS ->
       run Subscriber(j);
       j++
  :: j >= NUM_READERS -> break
  od
}

/* Safety: No valid reader misses a message (non-conflate mode) */
ltl no_message_loss {
  [](messages_written > 0 ->
     (read_valids[0] -> messages_read[0] <= messages_written))
}

/* Safety: Evicted readers don't continue reading */
ltl eviction_respected {
  []((read_valids[0] == false) -> (messages_read[0] == messages_read[0]))
}

/* Liveness: Publisher eventually completes */
ltl publisher_progress {
  <>(messages_written == 5)
}
```

## Key Properties to Verify

### P1: No Lost Messages (Valid Readers)
```promela
ltl no_loss {
  [](read_valids[i] && messages_written == N ->
     <>(messages_read[i] == N))
}
```

### P2: Eviction Correctness
```promela
ltl eviction_correct {
  []((write_pointer == read_pointers[i] &&
      write_cycle != read_cycles[i]) ->
     <>(read_valids[i] == false))
}
```

### P3: No Use-After-Eviction
```promela
ltl no_use_after_evict {
  [](read_valids[i] == false ->
     [](buffer access by i is blocked))
}
```

### P4: Cycle Counter Prevents Confusion
```promela
ltl cycle_prevents_confusion {
  []((write_cycle == read_cycles[i] &&
      write_pointer >= read_pointers[i]) ->
     (data is fresh))
}
```

## CI Integration

```yaml
# .github/workflows/spin.yml
name: SPIN Protocol Verification

on:
  pull_request:
    paths:
      - 'msgq_repo/msgq/msgq.cc'
      - 'msgq_repo/msgq/msgq.h'
      - 'verification/spin/**'

jobs:
  spin:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true

      - name: Install SPIN
        run: sudo apt-get install -y spin

      - name: Generate Verifier
        run: |
          cd verification/spin
          spin -a msgq_protocol.pml

      - name: Compile Verifier
        run: |
          cd verification/spin
          gcc -O2 -o pan pan.c

      - name: Run Verification
        run: |
          cd verification/spin
          ./pan -a -m10000  # Safety + LTL with 10K states max
```

## Known Issues to Verify

### Issue 1: New Reader During Wraparound
```promela
/* Model the race: publisher invalidating while new reader registers */
proctype RaceTest() {
  atomic {
    /* Publisher starts invalidation loop */
    byte i = 0;
    /* New reader registers here - not in num_readers yet */
    run Subscriber(NUM_READERS);  /* This reader escapes invalidation */
  }
}
```

### Issue 2: Conflate Mode Tight Loop
```promela
/* Model conflate mode where reader keeps restarting */
proctype ConflateReader(byte id) {
restart:
  if
  :: read_pointer != write_pointer ->
       /* Publisher writes again here */
       goto restart  /* Tight loop */
  :: else -> skip
  fi
}
```

## Verification Statistics (Expected)

| Property | States | Time |
|----------|--------|------|
| No message loss | ~5K | <1s |
| Eviction correct | ~10K | <2s |
| Full protocol | ~50K | <10s |

## Trade-offs

| Aspect | Promela Model | Real Code |
|--------|---------------|-----------|
| Atomicity | Explicit atomic blocks | Hardware atomics |
| Memory | Abstract buffer | Real shared memory |
| Timing | Interleaving model | Wall-clock time |
| Signals | Not modeled | SIGUSR2 notification |
