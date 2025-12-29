#include "catch2/catch.hpp"
#define private public
#include "common/params.h"
#include "common/util.h"

TEST_CASE("params_nonblocking_put") {
  char tmp_path[] = "/tmp/asyncWriter_XXXXXX";
  const std::string param_path = mkdtemp(tmp_path);
  auto param_names = {"CarParams", "IsMetric"};
  {
    Params params(param_path);
    for (const auto &name : param_names) {
      params.putNonBlocking(name, "1");
      // param is empty
      REQUIRE(params.get(name).empty());
    }

    // check if thread is running
    REQUIRE(params.future.valid());
    REQUIRE(params.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
  }
  // check results
  Params p(param_path);
  for (const auto &name : param_names) {
    REQUIRE(p.get(name) == "1");
  }
}

TEST_CASE("params_basic_operations") {
  char tmp_path[] = "/tmp/params_basic_XXXXXX";
  const std::string param_path = mkdtemp(tmp_path);
  Params params(param_path);

  // Test put and get
  params.put("TestParam", "test_value");
  REQUIRE(params.get("TestParam") == "test_value");

  // Test key presence via get
  REQUIRE_FALSE(params.get("TestParam").empty());
  REQUIRE(params.get("NonExistent").empty());

  // Test remove
  params.remove("TestParam");
  REQUIRE(params.get("TestParam").empty());
}
