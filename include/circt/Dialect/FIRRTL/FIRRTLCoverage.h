//===- FIRRTLCoverage.h - FIRRTL coverage helpers ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for FIRRTL coverage helpers.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLCOVERAGE_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLCOVERAGE_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

namespace circt {
namespace firrtl {

Operation *findDefOp(Value v, Operation *consumerOp);

void annotateCoverPoint(Operation *op, const std::string &name,
                        const std::string &groupName, CircuitOp &circuit);

void annotateCoverPoint(Operation *op, const std::string &groupName,
                        CircuitOp &circuit);

class CoverPointInfo {
public:
  Operation *op;

  std::string name;
  std::string group;
  std::string modName;

  int width = -1;

  Operation *cover = nullptr;  // FIX: initialize to avoid undefined behavior
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLCOVERAGE_H
