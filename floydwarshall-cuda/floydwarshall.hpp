//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _FLOYDWARSHALL_HPP
#define _FLOYDWARSHALL_HPP

#include <cstdlib>
#include <chrono>

// The TimeInterval is a simple RAII class.
// Construct the timer at the point you want to start timing.
// Use the Elapsed() method to return time since construction.

class TimeInterval {
 public:
  TimeInterval() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

 private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

#endif
