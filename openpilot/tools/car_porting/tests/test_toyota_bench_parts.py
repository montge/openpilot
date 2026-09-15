import ast
import textwrap

import pytest

from openpilot.tools.car_porting import toyota_bench_parts as tbp


def write(tmp_path, name, src):
  path = tmp_path / name
  path.write_text(textwrap.dedent(src).lstrip())
  return path


class TestCleanFw:
  def test_strips_null_padding(self):
    assert tbp.clean_fw(b"8965B42063\x00\x00\x00\x00\x00\x00") == "8965B42063"

  def test_escapes_leading_count_byte(self):
    assert tbp.clean_fw(b"\x018821F3301100\x00\x00\x00\x00") == "\\x018821F3301100"

  def test_keeps_stacked_sub_versions_visible(self):
    # engine fingerprints concatenate two identifiers behind a count byte
    assert tbp.clean_fw(b"\x02342Q1000\x00\x0054212000\x00") == "\\x02342Q100054212000"

  def test_escapes_all_non_printable_bytes(self):
    assert tbp.clean_fw(b"A\x7fB") == "A\\x7fB"


class TestPartHint:
  @pytest.mark.parametrize(("part", "expected"), [
    ("8965B42063", "EPS / power steering ECU"),
    ("8821F4702000", "millimeter wave radar sensor"),
    ("8646F4201100", "forward recognition camera"),
    ("881514201200", "driving support ECU (DSU)"),
    ("F152642492", "brake actuator / skid control ECU"),
  ])
  def test_maps_known_prefixes(self, part, expected):
    assert tbp.part_hint("whatever", [part]) == expected

  def test_skips_leading_count_byte_when_matching(self):
    assert tbp.part_hint("fwdRadar", ["\\x018821F3301100"]) == "millimeter wave radar sensor"

  def test_falls_back_to_ecu_name(self):
    # TSS-P engine entries are calibration ids, not an 89663-xxxxx part number
    assert tbp.part_hint("engine", ["\\x02342Q100054212000"]) == "engine control module"

  def test_unknown_ecu_and_prefix(self):
    assert tbp.part_hint("mystery", ["ZZZ123"]) == "-"

  def test_no_versions_at_all(self):
    assert tbp.part_hint("srs", []) == "SRS airbag ECU"


class TestDbcFromNode:
  def parse_expr(self, src):
    return ast.parse(src, mode="eval").body

  def test_dbc_dict_call(self):
    assert tbp._dbc_from_node(self.parse_expr("dbc_dict('pt_gen', 'radar_gen')")) == {"pt": "pt_gen", "radar": "radar_gen"}

  def test_bus_dict_literal(self):
    assert tbp._dbc_from_node(self.parse_expr("{Bus.pt: 'pt_gen'}")) == {"pt": "pt_gen"}

  def test_unrelated_node(self):
    assert tbp._dbc_from_node(self.parse_expr("CarSpecs(mass=1.0)")) == {}


VALUES_SRC = """
  @dataclass
  class ToyotaTSS2PlatformConfig(PlatformConfig):
    dbc_dict: dict = field(default_factory=lambda: dbc_dict('nodsu_pt_generated', 'tss2_adas'))

    def init(self):
      self.flags |= ToyotaFlags.TSS2 | ToyotaFlags.NO_DSU

      if self.flags & ToyotaFlags.RADAR_ACC:
        self.dbc_dict = {Bus.pt: 'nodsu_pt_generated'}


  class CAR(Platforms):
    TOYOTA_RAV4 = PlatformConfig(
      [
        ToyotaCarDocs("Toyota RAV4 2016", "Toyota Safety Sense P"),
        ToyotaCarDocs("Toyota RAV4 2017-18"),
      ],
      CarSpecs(mass=1.0),
      dbc_dict('new_mc_pt_generated', 'adas'),
    )
    TOYOTA_RAV4_TSS2 = ToyotaTSS2PlatformConfig(
      [ToyotaCarDocs("Toyota RAV4 2019-21")],
      TOYOTA_RAV4.specs,
    )
    TOYOTA_RAV4_TSS2_2022 = ToyotaTSS2PlatformConfig(
      [ToyotaCarDocs("Toyota RAV4 2022")],
      TOYOTA_RAV4.specs,
      flags=ToyotaFlags.RADAR_ACC,
    )
    NOT_AN_ASSIGNMENT_TARGET: int = 3
"""


class TestParsePlatforms:
  @pytest.fixture
  def platforms(self, tmp_path):
    return tbp.parse_platforms(write(tmp_path, "values.py", VALUES_SRC))

  def test_reads_car_docs_model_names(self, platforms):
    assert platforms["TOYOTA_RAV4"].models == ["Toyota RAV4 2016", "Toyota RAV4 2017-18"]

  def test_positional_dbc_dict_wins_for_base_config(self, platforms):
    assert platforms["TOYOTA_RAV4"].dbc == {"pt": "new_mc_pt_generated", "radar": "adas"}

  def test_no_flags_on_plain_platform_config(self, platforms):
    assert platforms["TOYOTA_RAV4"].flags == set()

  def test_subclass_applies_its_default_dbc_and_flags(self, platforms):
    # the dataclass field default, not a positional arg
    assert platforms["TOYOTA_RAV4_TSS2"].dbc == {"pt": "nodsu_pt_generated", "radar": "tss2_adas"}
    assert platforms["TOYOTA_RAV4_TSS2"].flags == {"TSS2", "NO_DSU"}

  def test_if_test_flags_are_not_treated_as_set_flags(self, platforms):
    # RADAR_ACC is only referenced in an `if self.flags & ...` test, so it must not leak in
    assert "RADAR_ACC" not in platforms["TOYOTA_RAV4_TSS2"].flags

  def test_explicit_flag_triggers_conditional_dbc_override(self, platforms):
    platform = platforms["TOYOTA_RAV4_TSS2_2022"]
    assert platform.flags == {"TSS2", "NO_DSU", "RADAR_ACC"}
    assert platform.dbc == {"pt": "nodsu_pt_generated"}

  def test_ignores_non_call_class_members(self, platforms):
    assert "NOT_AN_ASSIGNMENT_TARGET" not in platforms


FINGERPRINTS_SRC = """
  FW_VERSIONS = {
    CAR.TOYOTA_RAV4: {
      (Ecu.eps, 0x7a1, None): [
        b'8965B42063\\x00\\x00\\x00\\x00\\x00\\x00',
        b'8965B42073\\x00\\x00\\x00\\x00\\x00\\x00',
      ],
      (Ecu.fwdCamera, 0x750, 0x6d): [
        b'8646F4201100\\x00\\x00\\x00\\x00',
      ],
    },
  }
"""


class TestParseFingerprints:
  @pytest.fixture
  def entries(self, tmp_path):
    return tbp.parse_fingerprints(write(tmp_path, "fingerprints.py", FINGERPRINTS_SRC))["TOYOTA_RAV4"]

  def test_keeps_ecu_address_join(self, entries):
    by_ecu = {e.ecu: e for e in entries}
    assert by_ecu["eps"].address == 0x7A1
    assert by_ecu["eps"].subaddr is None
    assert by_ecu["fwdCamera"].address == 0x750
    assert by_ecu["fwdCamera"].subaddr == 0x6D

  def test_part_numbers_are_cleaned_and_deduped(self, entries):
    by_ecu = {e.ecu: e for e in entries}
    assert by_ecu["eps"].part_numbers == ["8965B42063", "8965B42073"]

  def test_skips_entries_it_cannot_interpret(self, tmp_path):
    src = """
      FW_VERSIONS = {
        CAR.TOYOTA_RAV4: {
          (Ecu.eps, 0x7a1, None): [b'8965B42063'],
          SOME_CONSTANT: [b'ignored'],
          (Ecu.abs, 0x7b0, None): SHARED_LIST,
        },
        SOME_ALIAS: OTHER_VERSIONS,
      }
    """
    parsed = tbp.parse_fingerprints(write(tmp_path, "fingerprints.py", src))
    assert list(parsed) == ["TOYOTA_RAV4"]
    assert [e.ecu for e in parsed["TOYOTA_RAV4"]] == ["eps"]


class TestParseDbcMessages:
  @pytest.fixture
  def dbc_dir(self, tmp_path):
    write(tmp_path, "_shared.dbc", """
      BO_ 36 KINEMATICS: 8 XXX
      BO_ 951 ESP_CONTROL: 8 ESP
    """)
    write(tmp_path, "some_pt.dbc", """
      CM_ "IMPORT _shared.dbc";

      BO_ 610 EPS_STATUS: 5 EPS
      BO_ 550 BRAKE_MODULE: 8 XXX
    """)
    write(tmp_path, "other_pt.dbc", """
      BO_ 15 SECOC_SYNCHRONIZATION: 8 XXX
    """)
    return tmp_path

  def test_follows_import_directives(self, dbc_dir):
    messages = tbp.parse_dbc_messages(dbc_dir, "some_pt_generated")
    assert messages["EPS_STATUS"] == (610, "EPS")
    assert messages["ESP_CONTROL"] == (951, "ESP")

  def test_unattributed_messages_keep_xxx(self, dbc_dir):
    assert tbp.parse_dbc_messages(dbc_dir, "some_pt_generated")["BRAKE_MODULE"] == (550, "XXX")

  def test_excludes_messages_from_unimported_dbcs(self, dbc_dir):
    # this is what keeps SecOC messages off a TSS-P report
    assert "SECOC_SYNCHRONIZATION" not in tbp.parse_dbc_messages(dbc_dir, "some_pt_generated")

  def test_empty_dbc_name_yields_nothing(self, dbc_dir):
    assert tbp.parse_dbc_messages(dbc_dir, "") == {}

  def test_missing_file_is_not_an_error(self, dbc_dir):
    assert tbp.parse_dbc_messages(dbc_dir, "does_not_exist_generated") == {}


CARSTATE_SRC = """
  class CarState(CarStateBase):
    @staticmethod
    def get_can_parsers(CP):
      pt_messages = [("BLINKERS_STATE", float('nan'))]

      return {
        Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, 0),
        Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], [], 2),
      }

    def update(self, can_parsers):
      cp = can_parsers[Bus.pt]
      cp_cam = can_parsers[Bus.cam]
      ret.gas = cp.vl["GAS_PEDAL"]["GAS_PEDAL_USER"]
      ret.speed = cp.vl_all["WHEEL_SPEEDS"]["WHEEL_SPEED_FL"]
      ret.steer = cp.vl["EPS_STATUS"]["LKA_STATE"]
      self.hud = cp_cam.vl["LKAS_HUD"]["LDA_ON_MESSAGE"]
"""


class TestParseCarstate:
  @pytest.fixture
  def carstate(self, tmp_path):
    return write(tmp_path, "carstate.py", CARSTATE_SRC)

  def test_bus_map(self, carstate):
    assert tbp.parse_bus_map(carstate) == {"pt": 0, "cam": 2}

  def test_rx_messages_grouped_by_parser(self, carstate):
    rx = tbp.parse_rx_messages(carstate)
    assert rx["pt"] == {"GAS_PEDAL", "WHEEL_SPEEDS", "EPS_STATUS"}
    assert rx["cam"] == {"LKAS_HUD"}

  def test_cam_parser_is_not_matched_by_the_pt_pattern(self, carstate):
    # 'cp' must not match inside 'cp_cam'
    assert "LKAS_HUD" not in tbp.parse_rx_messages(carstate)["pt"]


class TestParseTxMessages:
  def test_reads_bus_from_message_builders(self, tmp_path):
    toyotacan = write(tmp_path, "toyotacan.py", """
      def create_steer_command(packer, steer, steer_req):
        return packer.make_can_msg("STEERING_LKA", 0, values)

      def create_accel_command(packer, accel):
        return packer.make_can_msg("ACC_CONTROL", 0, values)
    """)
    assert tbp.parse_tx_messages(toyotacan) == {"STEERING_LKA": 0, "ACC_CONTROL": 0}


class TestRender:
  @pytest.fixture
  def rendered(self, monkeypatch, tmp_path):
    write(tmp_path, "some_pt.dbc", """
      BO_ 610 EPS_STATUS: 5 EPS
      BO_ 740 STEERING_LKA: 8 XXX
    """)
    monkeypatch.setattr(tbp, "DBC_GENERATOR_DIR", tmp_path)

    platform = tbp.Platform(name="TOYOTA_RAV4", models=["Toyota RAV4 2016"],
                            dbc={"pt": "some_pt_generated"}, flags={"TSS2"})
    fingerprints = {"TOYOTA_RAV4": [tbp.EcuEntry("eps", 0x7A1, None, [b"8965B42063\x00"])]}

    def render(**kwargs):
      return tbp.render([platform], fingerprints, {"pt": {"EPS_STATUS"}}, {"STEERING_LKA": 0},
                        {"pt": 0}, **kwargs)
    return render

  def test_text_output(self, rendered):
    out = rendered(markdown=False, all_versions=True)
    assert "== TOYOTA_RAV4" in out
    assert "Donor cars: Toyota RAV4 2016" in out
    assert "Platform flags: TSS2" in out
    assert "8965B42063" in out

  def test_markdown_output_is_a_table(self, rendered):
    out = rendered(markdown=True, all_versions=True)
    assert "## TOYOTA_RAV4" in out
    assert "| ECU | Request | Subaddr | Known | Part | Part numbers |" in out
    assert "`8965B42063`" in out

  def test_read_and_write_rows_carry_bus_and_node(self, rendered):
    out = rendered(markdown=True, all_versions=True)
    assert "| read | 0 | EPS_STATUS | 0x262 | EPS |" in out
    assert "| write | 0 | STEERING_LKA | 0x2e4 | XXX |" in out

  def test_part_numbers_are_capped_without_all_versions(self, monkeypatch, tmp_path):
    monkeypatch.setattr(tbp, "DBC_GENERATOR_DIR", tmp_path)
    versions = [f"8965B4206{i}".encode() for i in range(9)]
    entry = tbp.EcuEntry("eps", 0x7A1, None, versions)
    out = tbp.render([tbp.Platform(name="P", dbc={})], {"P": [entry]}, {}, {}, {},
                     markdown=False, all_versions=False)
    assert f"... (+{9 - tbp.MAX_PART_NUMBERS})" in out

  def test_platform_without_dbc_reports_unknown(self, monkeypatch, tmp_path):
    monkeypatch.setattr(tbp, "DBC_GENERATOR_DIR", tmp_path)
    out = tbp.render([tbp.Platform(name="P", dbc={})], {}, {}, {}, {}, markdown=False, all_versions=False)
    assert "DBC: unknown" in out
    assert "Donor cars: unknown" in out


@pytest.mark.skipif(not tbp.TOYOTA_DIR.exists(), reason="opendbc submodule is not checked out")
class TestMainAgainstRealOpendbc:
  def test_list_mode(self, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["toyota_bench_parts.py", "--list"])
    assert tbp.main() == 0
    out = capsys.readouterr().out
    assert "TOYOTA_RAV4" in out
    assert "ECUs" in out

  def test_default_platform_report(self, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["toyota_bench_parts.py"])
    assert tbp.main() == 0
    out = capsys.readouterr().out
    assert f"== {tbp.DEFAULT_PLATFORM}" in out
    assert "Modules to source" in out
    assert "CAN messages openpilot needs" in out

  def test_unknown_platform_exits_nonzero(self, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["toyota_bench_parts.py", "NOT_A_PLATFORM"])
    assert tbp.main() == 1
    assert "unknown platform(s): NOT_A_PLATFORM" in capsys.readouterr().err


class TestMain:
  def test_missing_opendbc_checkout_exits_nonzero(self, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(tbp, "TOYOTA_DIR", tmp_path / "nope")
    monkeypatch.setattr("sys.argv", ["toyota_bench_parts.py"])
    assert tbp.main() == 1
    assert "git submodule update --init opendbc_repo" in capsys.readouterr().err
