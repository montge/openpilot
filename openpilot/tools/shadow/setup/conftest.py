# The test_* scripts in this directory are manual, on-device diagnostics that
# require a live camera server (camera_bridge.py / VisionIPC). They are meant to
# be run directly (python3 test_visionipc_consumer.py), not collected by pytest.
collect_ignore = ["test_visionipc_consumer.py"]
