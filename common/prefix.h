#pragma once

#include <cassert>
#include <filesystem>
#include <string>

#include "common/params.h"
#include "common/util.h"
#include "system/hardware/hw.h"

class OpenpilotPrefix {
public:
  OpenpilotPrefix(std::string prefix = {}) {
    if (prefix.empty()) {
      prefix = util::random_string(15);
    }
    msgq_path = Path::shm_path() + "/msgq_" + prefix;
    bool ret = util::create_directories(msgq_path, 0777);
    assert(ret);
    setenv("OPENPILOT_PREFIX", prefix.c_str(), 1);
  }

  ~OpenpilotPrefix() {
    std::error_code ec;  // Use non-throwing overload
    auto param_path = Params().getParamPath();
    if (util::file_exists(param_path)) {
      std::string real_path = util::readlink(param_path);
      std::filesystem::remove_all(real_path, ec);
      unlink(param_path.c_str());
    }
    if (getenv("COMMA_CACHE") == nullptr) {
      std::filesystem::remove_all(Path::download_cache_root(), ec);
    }
    std::filesystem::remove_all(Path::comma_home(), ec);
    std::filesystem::remove_all(msgq_path, ec);
    unsetenv("OPENPILOT_PREFIX");
  }

private:
  std::string msgq_path;
};
