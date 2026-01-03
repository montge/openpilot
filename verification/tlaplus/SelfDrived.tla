---------------------------- MODULE SelfDrived ----------------------------
(*
 * TLA+ specification for the openpilot selfdrived state machine.
 *
 * This module models the 5-state machine from selfdrive/selfdrived/state.py
 * that controls vehicle engagement and disengagement. It verifies critical
 * safety properties:
 *   - Disable commands are always honored
 *   - NO_ENTRY blocks engagement
 *   - Soft disable timer progresses correctly
 *   - System never deadlocks
 *
 * Reference: selfdrive/selfdrived/state.py
 *)
EXTENDS Integers, TLC

\* -------------------------------------------------------------------
\* Constants
\* -------------------------------------------------------------------

\* SOFT_DISABLE_TIME represents the timer countdown in frames.
\* In production: 300 frames at 100Hz = 3 seconds.
\* For model checking: use small value (e.g., 3-5) to reduce state space.
CONSTANTS SOFT_DISABLE_TIME

ASSUME SOFT_DISABLE_TIME \in Nat /\ SOFT_DISABLE_TIME > 0

\* -------------------------------------------------------------------
\* State and Event Definitions
\* -------------------------------------------------------------------

\* All possible states of the state machine (matching State enum in state.py)
States == {"disabled", "preEnabled", "enabled", "softDisabling", "overriding"}

\* States where the system is considered "active" (actuators engaged)
ActiveStates == {"enabled", "softDisabling", "overriding"}

\* States where the system is considered "enabled" (including pre-enabled)
EnabledStates == {"preEnabled", "enabled", "softDisabling", "overriding"}

\* Event types that can trigger state transitions (matching ET enum in events.py)
EventTypes == {
  "ENABLE",              \* Request to enable controls
  "PRE_ENABLE",          \* Waiting for conditions (e.g., brake release)
  "NO_ENTRY",            \* Block engagement (camera malfunction, etc.)
  "SOFT_DISABLE",        \* Graceful disable with 3s warning
  "IMMEDIATE_DISABLE",   \* Immediate disable (serious issue)
  "USER_DISABLE",        \* User-initiated disable (brake, cancel button)
  "OVERRIDE_LATERAL",    \* User overriding steering
  "OVERRIDE_LONGITUDINAL" \* User overriding accel/brake
}

\* -------------------------------------------------------------------
\* Variables
\* -------------------------------------------------------------------

VARIABLES
  state,              \* Current state machine state \in States
  soft_disable_timer, \* Countdown timer (frames remaining) \in 0..SOFT_DISABLE_TIME
  current_events      \* Set of active event types this frame \subseteq EventTypes

vars == <<state, soft_disable_timer, current_events>>

\* -------------------------------------------------------------------
\* Type Invariant
\* -------------------------------------------------------------------

TypeInvariant ==
  /\ state \in States
  /\ soft_disable_timer \in 0..SOFT_DISABLE_TIME
  /\ current_events \subseteq EventTypes

\* -------------------------------------------------------------------
\* Initial State
\* -------------------------------------------------------------------

\* System starts in disabled state with timer at 0 and no events
Init ==
  /\ state = "disabled"
  /\ soft_disable_timer = 0
  /\ current_events = {}

\* -------------------------------------------------------------------
\* State Transition Logic
\* -------------------------------------------------------------------

\* Helper: Check if any override event is present
HasOverride == "OVERRIDE_LATERAL" \in current_events \/ "OVERRIDE_LONGITUDINAL" \in current_events

\* Transition from disabled state (lines 78-90 in state.py)
TransitionFromDisabled ==
  IF "ENABLE" \in current_events
  THEN
    IF "NO_ENTRY" \in current_events
    THEN
      \* NO_ENTRY blocks engagement, stay disabled
      /\ state' = "disabled"
      /\ soft_disable_timer' = soft_disable_timer
    ELSE
      \* Can engage - determine target state
      IF "PRE_ENABLE" \in current_events
      THEN
        /\ state' = "preEnabled"
        /\ soft_disable_timer' = soft_disable_timer
      ELSE IF HasOverride
      THEN
        /\ state' = "overriding"
        /\ soft_disable_timer' = soft_disable_timer
      ELSE
        /\ state' = "enabled"
        /\ soft_disable_timer' = soft_disable_timer
  ELSE
    \* No ENABLE event, stay disabled
    /\ UNCHANGED <<state, soft_disable_timer>>

\* Transition from enabled state (lines 37-45 in state.py)
TransitionFromEnabled ==
  IF "SOFT_DISABLE" \in current_events
  THEN
    /\ state' = "softDisabling"
    /\ soft_disable_timer' = SOFT_DISABLE_TIME
  ELSE IF HasOverride
  THEN
    /\ state' = "overriding"
    /\ soft_disable_timer' = soft_disable_timer
  ELSE
    /\ UNCHANGED <<state, soft_disable_timer>>

\* Transition from softDisabling state (lines 48-57 in state.py)
\* Note: Timer decrement happens at start of update in Python code
TransitionFromSoftDisabling ==
  LET decremented_timer == IF soft_disable_timer > 0 THEN soft_disable_timer - 1 ELSE 0
  IN
    IF "SOFT_DISABLE" \notin current_events
    THEN
      \* Soft disable condition cleared, return to enabled
      /\ state' = "enabled"
      /\ soft_disable_timer' = decremented_timer
    ELSE IF decremented_timer > 0
    THEN
      \* Timer still running, stay in softDisabling
      /\ state' = "softDisabling"
      /\ soft_disable_timer' = decremented_timer
    ELSE
      \* Timer expired, disable
      /\ state' = "disabled"
      /\ soft_disable_timer' = 0

\* Transition from preEnabled state (lines 60-64 in state.py)
TransitionFromPreEnabled ==
  IF "PRE_ENABLE" \notin current_events
  THEN
    \* Pre-enable condition cleared, transition to enabled
    /\ state' = "enabled"
    /\ soft_disable_timer' = soft_disable_timer
  ELSE
    /\ UNCHANGED <<state, soft_disable_timer>>

\* Transition from overriding state (lines 67-75 in state.py)
TransitionFromOverriding ==
  IF "SOFT_DISABLE" \in current_events
  THEN
    /\ state' = "softDisabling"
    /\ soft_disable_timer' = SOFT_DISABLE_TIME
  ELSE IF ~HasOverride
  THEN
    \* Override released, return to enabled
    /\ state' = "enabled"
    /\ soft_disable_timer' = soft_disable_timer
  ELSE
    /\ UNCHANGED <<state, soft_disable_timer>>

\* Main state transition function
Update ==
  IF state # "disabled"
  THEN
    \* Priority 1: User/Immediate disable (highest priority, lines 27-33 in state.py)
    IF "USER_DISABLE" \in current_events \/ "IMMEDIATE_DISABLE" \in current_events
    THEN
      /\ state' = "disabled"
      /\ soft_disable_timer' = 0
    ELSE
      \* Priority 2: State-specific transitions
      CASE state = "enabled" -> TransitionFromEnabled
        [] state = "softDisabling" -> TransitionFromSoftDisabling
        [] state = "preEnabled" -> TransitionFromPreEnabled
        [] state = "overriding" -> TransitionFromOverriding
        [] OTHER -> UNCHANGED <<state, soft_disable_timer>>
  ELSE
    \* state = "disabled"
    TransitionFromDisabled

\* -------------------------------------------------------------------
\* Next-State Relation
\* -------------------------------------------------------------------

\* Non-deterministically choose any subset of events, then update state
Next ==
  /\ \E events \in SUBSET EventTypes : current_events' = events
  /\ Update

\* -------------------------------------------------------------------
\* Safety Properties (Invariants)
\* -------------------------------------------------------------------

\* P1: USER_DISABLE and IMMEDIATE_DISABLE always transition to disabled
\* (When in a non-disabled state and these events occur, we MUST go to disabled)
DisableAlwaysHonored ==
  state # "disabled" /\
  ("USER_DISABLE" \in current_events \/ "IMMEDIATE_DISABLE" \in current_events)
    => state' = "disabled"

\* P2: NO_ENTRY blocks engagement from disabled state
\* (Cannot transition from disabled to enabled/preEnabled/overriding if NO_ENTRY present)
NoEntryBlocksEngagement ==
  (state = "disabled" /\ "NO_ENTRY" \in current_events)
    => state' = "disabled"

\* P3: Timer is only positive when in softDisabling state
\* Note: Timer can momentarily be non-zero in other states during transition
TimerConsistency ==
  state' # "softDisabling" => soft_disable_timer' = 0 \/ soft_disable_timer' < soft_disable_timer

\* P4: Never in an undefined state
StateValid == state \in States

\* P5: Timer never exceeds maximum
TimerBounded == soft_disable_timer <= SOFT_DISABLE_TIME

\* P6: Once disabled with USER_DISABLE or IMMEDIATE_DISABLE, timer is reset
DisableResetsTimer ==
  (state # "disabled" /\
   ("USER_DISABLE" \in current_events \/ "IMMEDIATE_DISABLE" \in current_events))
    => soft_disable_timer' = 0

\* -------------------------------------------------------------------
\* Temporal Properties (Liveness)
\* -------------------------------------------------------------------

\* T1: softDisabling eventually leads to either disabled or enabled
\* (No deadlock in softDisabling state)
SoftDisableProgress ==
  state = "softDisabling" ~> (state = "disabled" \/ state = "enabled")

\* T2: System never deadlocks (always possible to take a step)
NoDeadlock == TRUE  \* Verified by TLC if it completes without deadlock

\* T3: preEnabled eventually leads to enabled (if no disables)
PreEnableProgress ==
  state = "preEnabled" ~> (state = "enabled" \/ state = "disabled")

\* T4: overriding eventually leads to enabled or softDisabling (if no disables)
OverrideProgress ==
  state = "overriding" ~> (state = "enabled" \/ state = "softDisabling" \/ state = "disabled")

\* -------------------------------------------------------------------
\* Specification
\* -------------------------------------------------------------------

\* Full specification with weak fairness to ensure progress
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Specification without fairness (for safety checking only)
SafetySpec == Init /\ [][Next]_vars

=============================================================================
\* Modification History
\* Created for openpilot TLA+ verification
\* Reference: selfdrive/selfdrived/state.py
