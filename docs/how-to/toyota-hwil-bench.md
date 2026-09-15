# Toyota hardware-in-the-loop bench

This page covers sourcing real Toyota modules for a bench, and which of them openpilot
actually talks to. It exists because the question "what parts do I need from a Toyota to
build a HWIL rig?" has no answer anywhere else in the repo — but the data to answer it does.

Read [what the repo already gives you](#what-the-repo-already-gives-you) first: if your goal
is regression testing rather than exercising real ECUs, the CAN replay path is much cheaper
and is the one openpilot already supports.

## What the repo already gives you

Three things exist today, and none of them is a bench parts list:

* **[Supported cars](../CARS.md) "Parts" column.** Every RAV4 row lists an OBD-C cable, a
  **Toyota A connector** harness, a comma device, a comma power v3, a harness box and a
  mount. That is what you buy to install a comma device in a *working* car. It says nothing
  about which modules to pull from a donor.
* **comma's hardware-in-the-loop CI.** Referenced in the repo `README.md` and in
  [SAFETY.md](../SAFETY.md), and implemented in the root `Jenkinsfile` as device stages
  (`tizi-needs-can`, `tizi-common`, `tici-os04c10`). The hardware in that loop is a comma
  device and a panda — not car modules.
* **`openpilot/tools/replay/can_replay.py`.** The in-tree tool for feeding a bench CAN: it
  loads a logged route and replays its CAN frames to every connected panda and
  [panda jungle](https://comma.ai/shop/panda-jungle) in a loop, with optional ignition and
  power cycling via the `ON`/`OFF` and `PWR_ON`/`PWR_OFF` environment variables.

  ```bash
  openpilot/tools/replay/can_replay.py <route>
  ON=30 OFF=5 openpilot/tools/replay/can_replay.py <route>   # cycle ignition
  ```

A jungle replaying a recorded RAV4 route reproduces every message in the tables below,
deterministically, for the price of one jungle. A bench of real ECUs buys you something
replay cannot: modules that *respond* — an EPS that reacts to `STEERING_LKA` and reports
`EPS_STATUS`, a DSU whose ACC state machine can reject your command. If you do not need
that closed loop, use replay.

## Generating the parts list

`openpilot/tools/car_porting/toyota_bench_parts.py` joins the opendbc FW fingerprints (which are
literal Toyota part numbers, keyed by ECU and diagnostic request address) with the CAN
messages the Toyota car interface reads and writes:

```bash
git submodule update --init opendbc_repo     # if opendbc is not checked out yet

openpilot/tools/car_porting/toyota_bench_parts.py --list
openpilot/tools/car_porting/toyota_bench_parts.py TOYOTA_RAV4
openpilot/tools/car_porting/toyota_bench_parts.py TOYOTA_RAV4 TOYOTA_RAV4H --format markdown
openpilot/tools/car_porting/toyota_bench_parts.py TOYOTA_RAV4_TSS2 --all-versions
```

It parses the opendbc sources with the standard library only, so it needs no build and no
venv. Everything below is its output for `TOYOTA_RAV4`, plus commentary that is *not*
derived from the repo and is flagged as such.

## Which RAV4 to buy

The generation matters more than the year:

| Platform | Cars | Architecture |
|---|---|---|
| `TOYOTA_RAV4` / `TOYOTA_RAV4H` | 2016-18 (TSS-P) | Separate **DSU** doing ACC, camera does lane keeping |
| `TOYOTA_RAV4_TSS2` | 2019-21 | `NO_DSU`, camera-integrated ADAS |
| `TOYOTA_RAV4_TSS2_2022` | 2022 | `NO_DSU`, `RADAR_ACC` — radar owns ACC, no radar DBC |
| `TOYOTA_RAV4_TSS2_2023` | 2023-25 | adds `ANGLE_CONTROL` |

**A 2016-18 TSS-P car is the easiest bench**, and it is the one that matches "old RAV4
parts": the DSU is a discrete box that openpilot impersonates, so the interception point is
a physical connector rather than a function inside the camera. The flags come from
`opendbc/car/toyota/values.py`; the reasoning about which is easiest to bench is ours.

Note that RAV4 Prime 2021+ is one of the [SecOC cars](../CARS.md#toyota-security) — message
authentication, no general openpilot support. Do not buy those modules for this.

## Modules to source — `TOYOTA_RAV4` (2016-18)

`Request` is the UDS diagnostic address openpilot queries to fingerprint the module;
`Known` is how many distinct firmware versions opendbc has recorded, which is a rough proxy
for how many variants are out there.

| ECU | Request | Subaddr | Known | Part | Part numbers |
|---|---|---|---|---|---|
| abs | 0x7b0 | - | 4 | brake actuator / skid control ECU | `F15260R102`, `F15260R103`, `F152642492`, `F152642493` |
| dsu | 0x791 | - | 3 | driving support ECU (DSU) | `881514201200`, `881514201300`, `881514201400` |
| engine | 0x7e0 | - | 9 | engine control module | `\x02342Q100054212000`, `\x02342Q110054212000`, `\x02342Q120054212000`, `\x02342Q130054212000`, `\x02342Q200054213000`, `\x02342Q210054213000`, `\x02342Q220054213000`, `\x02342Q300054214000`, `\x02342Q400054215000` |
| eps | 0x7a1 | - | 4 | EPS / power steering ECU | `8965B42063`, `8965B42073`, `8965B42082`, `8965B42083` |
| fwdCamera | 0x750 | 0x6d | 5 | forward recognition camera | `8646F4201100`, `8646F4201200`, `8646F4202001`, `8646F4202100`, `8646F4204000` |
| fwdRadar | 0x750 | 0xf | 3 | millimeter wave radar sensor | `8821F4702000`, `8821F4702100`, `8821F4702300` |

Fingerprints are null-padded and some carry a leading count byte or concatenate
sub-versions, so the tool escapes non-printable bytes rather than dropping them: the engine
entries above are a count byte plus two stacked identifiers (`342Q1000` and `54212000`).
Those are calibration IDs, not a `89663-xxxxx` part number — TSS-P engine ECUs are best
matched by donor year and drivetrain, whereas the TSS2 platforms do carry the part number
directly (`\x018966342E2000` → 89663-42E20). The `Part` column maps the Toyota prefix
(`8965B` → EPS, `8821F` → radar, `8646F` → camera, `88151` → DSU, `F1526` → ABS,
`89663` → ECM) and falls back to the ECU name; it is curated in the tool, not read from
opendbc.

The hybrid platform `TOYOTA_RAV4H` shares the radar and camera exactly, and differs on
`eps` (`8965B42102`, `8965B42103`, `8965B42112`, `8965B42162`, `8965B42163`), `abs`
(`F152642090`, `F152642110`, `F152642120`, `F152642400`), `dsu` (`881514202200`,
`881514202300`, `881514202400`) and of course the engine ECU. Run the tool for the exact
list.

## CAN messages openpilot needs

This is the part that decides which modules have to be *powered and talking* rather than
just present. Bus 0 is powertrain, bus 2 is the camera bus; both come out of
`get_can_parsers` in `opendbc/car/toyota/carstate.py`. The `DBC node` column is the
transmitting node as recorded in the DBC — `XXX` means opendbc does not attribute the
message to a node, not that it has no sender.

| Dir | Bus | Message | Address | DBC node |
|---|---|---|---|---|
| read | 2 | LKAS_HUD | 0x412 | DSU |
| read | 0 | BLINKERS_STATE | 0x614 | XXX |
| read | 0 | BODY_CONTROL_STATE | 0x620 | XXX |
| read | 0 | BODY_CONTROL_STATE_2 | 0x610 | XXX |
| read | 0 | BRAKE_MODULE | 0x224 | XXX |
| read | 0 | DSU_CRUISE | 0x365 | DSU |
| read | 0 | EPS_STATUS | 0x262 | EPS |
| read | 0 | ESP_CONTROL | 0x3b7 | ESP |
| read | 0 | GAS_PEDAL | 0x2c1 | XXX |
| read | 0 | GEAR_PACKET | 0x3bc | XXX |
| read | 0 | GEAR_PACKET_HYBRID | 0x127 | XXX |
| read | 0 | LIGHT_STALK | 0x622 | SCM |
| read | 0 | PCM_CRUISE | 0x1d2 | XXX |
| read | 0 | PCM_CRUISE_2 | 0x1d3 | XXX |
| read | 0 | PCM_CRUISE_ALT | 0x3f1 | XXX |
| read | 0 | PCM_CRUISE_SM | 0x399 | XXX |
| read | 0 | STEER_ANGLE_SENSOR | 0x25 | XXX |
| read | 0 | STEER_TORQUE_SENSOR | 0x260 | XXX |
| read | 0 | VSC1S07 | 0x320 | CGW |
| read | 0 | WHEEL_SPEEDS | 0xaa | XXX |
| write | 0 | ACC_CONTROL | 0x343 | DSU |
| write | 0 | LKAS_HUD | 0x412 | DSU |
| write | 0 | PCM_CRUISE | 0x1d2 | XXX |
| write | 0 | PCS_HUD | 0x411 | DSU |
| write | 0 | PRE_COLLISION | 0x283 | DSU |
| write | 0 | PRE_COLLISION_2 | 0x344 | DSU |
| write | 0 | STEERING_LKA | 0x2e4 | XXX |

Grouping that by donor module — our reading of the table, not something opendbc states:

* **EPS** — `STEER_TORQUE_SENSOR`, `STEER_ANGLE_SENSOR`, `EPS_STATUS`, and the receiver of
  `STEERING_LKA`. The one module a steering bench cannot fake if you want closed-loop
  behavior.
* **ABS / skid control** — `ESP_CONTROL`, `BRAKE_MODULE`, `WHEEL_SPEEDS`. Wheel speeds
  gate almost everything downstream, including whether openpilot thinks the car is moving.
* **Engine / PCM** — `PCM_CRUISE*`, `GAS_PEDAL`, `GEAR_PACKET`. Cruise state and gear are
  what let openpilot engage at all.
* **DSU** — `DSU_CRUISE`, and the module openpilot impersonates when it writes
  `ACC_CONTROL`, `PRE_COLLISION*`, `PCS_HUD` and `LKAS_HUD`.
* **Body / combination meter** — `BODY_CONTROL_STATE`, `BLINKERS_STATE`, `LIGHT_STALK`.
  Doors, seatbelt, parking brake and turn signals; missing these shows up as spurious
  disengagements rather than a hard fault.
* **Forward camera** — the bus 2 `LKAS_HUD` that openpilot reads back.

Anything on this list you do not have on the bench you will have to inject, which is what
`can_replay.py` and a jungle are for. A hybrid of real modules plus replayed filler traffic
is usually more practical than sourcing the whole car.

## Practical notes

These are engineering caveats, not repo facts:

* **ECUs fault without their sensors and actuators.** An EPS on a bench with no torque
  sensor, no motor load and no steering column will report a fault in `EPS_STATUS` fairly
  quickly. Plan for either the mechanical parts that go with each module, or for accepting
  fault states and testing around them.
* **Bus termination and power.** Toyota powertrain CAN is 500 kbps and needs 120 Ω at both
  ends. Each module wants switched ignition power and a solid ground; a bench supply sized
  for the EPS inrush is not optional if you power the steering motor.
* **Connector pinouts are not in this repo.** The Toyota A connector harness is built for
  in-car installation between the DSU/camera and the car. Bench wiring means pinning out
  each module's connector yourself.
* **The safety model does not change on a bench.** Actuation limits are enforced in panda
  firmware (`opendbc/safety/modes/toyota.h`), not in the code under test. Do not weaken or
  bypass them to make a bench behave; see [SAFETY.md](../SAFETY.md).
