# TLA+ State Machine Verification Design

## State Machine Model

### States
```tla
States == {"disabled", "preEnabled", "enabled", "softDisabling", "overriding"}

ActiveStates == {"enabled", "softDisabling", "overriding"}
EnabledStates == {"preEnabled", "enabled", "softDisabling", "overriding"}
```

### Event Types (from events.py)
```tla
EventTypes == {
  "ENABLE",           \* Request to enable controls
  "PRE_ENABLE",       \* Waiting for conditions (e.g., brake release)
  "NO_ENTRY",         \* Block engagement (camera malfunction, etc.)
  "SOFT_DISABLE",     \* Graceful disable with 3s warning
  "IMMEDIATE_DISABLE",\* Immediate disable (serious issue)
  "USER_DISABLE",     \* User-initiated disable (brake, cancel button)
  "OVERRIDE_LATERAL", \* User overriding steering
  "OVERRIDE_LONGITUDINAL" \* User overriding accel/brake
}
```

## TLA+ Specification

```tla
---------------------------- MODULE SelfDrived ----------------------------
EXTENDS Integers, TLC

CONSTANTS SOFT_DISABLE_TIME  \* = 300 (frames at 100Hz = 3 seconds)

VARIABLES
  state,              \* Current state machine state
  soft_disable_timer, \* Countdown timer (frames remaining)
  current_events,     \* Set of active event types this frame
  initialized         \* Whether system is initialized

vars == <<state, soft_disable_timer, current_events, initialized>>

TypeInvariant ==
  /\ state \in States
  /\ soft_disable_timer \in 0..SOFT_DISABLE_TIME
  /\ current_events \subseteq EventTypes
  /\ initialized \in BOOLEAN

Init ==
  /\ state = "disabled"
  /\ soft_disable_timer = 0
  /\ current_events = {}
  /\ initialized = FALSE

\* State transitions
Update ==
  IF state # "disabled" THEN
    \* Priority 1: User/Immediate disable (highest priority)
    IF "USER_DISABLE" \in current_events \/ "IMMEDIATE_DISABLE" \in current_events
    THEN state' = "disabled" /\ soft_disable_timer' = 0

    \* Priority 2: State-specific transitions
    ELSE CASE state = "enabled" ->
           IF "SOFT_DISABLE" \in current_events
           THEN state' = "softDisabling" /\ soft_disable_timer' = SOFT_DISABLE_TIME
           ELSE IF "OVERRIDE_LATERAL" \in current_events \/ "OVERRIDE_LONGITUDINAL" \in current_events
           THEN state' = "overriding"
           ELSE UNCHANGED <<state, soft_disable_timer>>

         [] state = "preEnabled" ->
           IF "PRE_ENABLE" \notin current_events
           THEN state' = "enabled"
           ELSE UNCHANGED <<state, soft_disable_timer>>

         [] state = "softDisabling" ->
           IF "SOFT_DISABLE" \notin current_events
           THEN state' = "enabled" /\ soft_disable_timer' = 0
           ELSE IF soft_disable_timer = 0
           THEN state' = "disabled"
           ELSE soft_disable_timer' = soft_disable_timer - 1

         [] state = "overriding" ->
           IF "SOFT_DISABLE" \in current_events
           THEN state' = "softDisabling" /\ soft_disable_timer' = SOFT_DISABLE_TIME
           ELSE IF "OVERRIDE_LATERAL" \notin current_events /\ "OVERRIDE_LONGITUDINAL" \notin current_events
           THEN state' = "enabled"
           ELSE UNCHANGED <<state, soft_disable_timer>>

  ELSE \* state = "disabled"
    IF "ENABLE" \in current_events /\ "NO_ENTRY" \notin current_events
    THEN IF "PRE_ENABLE" \in current_events
         THEN state' = "preEnabled"
         ELSE state' = "enabled"
    ELSE UNCHANGED <<state, soft_disable_timer>>

Next ==
  /\ \E events \in SUBSET EventTypes : current_events' = events
  /\ Update
  /\ initialized' = TRUE

\* Safety Properties (Invariants)

\* P1: Disable commands always honored
DisableAlwaysHonored ==
  ("USER_DISABLE" \in current_events \/ "IMMEDIATE_DISABLE" \in current_events)
    => state' = "disabled"

\* P2: NO_ENTRY blocks engagement
NoEntryBlocks ==
  "NO_ENTRY" \in current_events => state' # "enabled"

\* P3: Timer only active in softDisabling
TimerConsistency ==
  soft_disable_timer > 0 => state = "softDisabling"

\* P4: No deadlock in softDisabling
SoftDisableProgress ==
  state = "softDisabling" ~> (state = "disabled" \/ state = "enabled")

\* Temporal Properties (Liveness)

\* T1: System eventually reaches stable state
EventuallyStable ==
  <>[]( state \in {"disabled", "enabled"} )

\* T2: Alert before disable (soft disable case)
AlertBeforeDisable ==
  [](state = "softDisabling" /\ soft_disable_timer > 0
     => <>(state = "disabled" \/ state = "enabled"))

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

=============================================================================
```

## CI Integration

```yaml
# .github/workflows/tlaplus.yml
name: TLA+ Verification

on:
  pull_request:
    paths:
      - 'verification/tlaplus/**'
      - 'selfdrive/selfdrived/state.py'

jobs:
  tla:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Java
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: Download TLA+ Tools
        run: |
          wget https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

      - name: Run TLC Model Checker
        run: |
          java -jar tla2tools.jar \
            -config verification/tlaplus/SelfDrived.cfg \
            verification/tlaplus/SelfDrived.tla
```

## Model Configuration

```tla
\* SelfDrived.cfg
CONSTANTS
  SOFT_DISABLE_TIME = 5  \* Use small value for model checking

INVARIANTS
  TypeInvariant
  DisableAlwaysHonored
  NoEntryBlocks
  TimerConsistency

PROPERTIES
  SoftDisableProgress
  EventuallyStable
  AlertBeforeDisable
```

## Trade-offs

| Property Type | Verification Time | Completeness |
|---------------|-------------------|--------------|
| Invariants | Fast (seconds) | All reachable states |
| Safety temporal | Medium (minutes) | Bounded depth |
| Liveness | Slow (minutes) | May need fairness |

## Refinement Path

1. **Abstract model** (this proposal): Verify core state machine
2. **Event refinement**: Add specific event conditions
3. **Timer refinement**: Model exact timing constraints
4. **Integration**: Connect to panda safety properties
