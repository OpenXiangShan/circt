//===- FindBranchCoverPoint.cpp - Find branch coverage points ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the FindBranchCoverPoint pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLCoverage.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_FINDBRANCHCOVERPOINT
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
class FindBranchCoverPoint
    : public circt::firrtl::impl::FindBranchCoverPointBase<
          FindBranchCoverPoint> {
public:
  void runOnOperation() final;

private:
  Value getNonConstBranchCondition(Operation *op);
};
} // namespace

void FindBranchCoverPoint::runOnOperation() {
  auto circuit = getOperation();

  circuit.walk([&](Operation *op) {
    auto condVal = getNonConstBranchCondition(op);
    if (!condVal)
      return;
    annotateCoverPoint(findDefOp(condVal, op), "branch", circuit);
  });
}

Value FindBranchCoverPoint::getNonConstBranchCondition(Operation *op) {
  // when (cond) { ... }
  if (auto whenOp = dyn_cast<WhenOp>(op)) {
    auto cond = whenOp.getCondition();
    if (cond && !cond.getType().isConst())
      return cond;
  }
  // Mux(cond, ..., ...)
  else if (auto mux = dyn_cast<MuxPrimOp>(op)) {
    auto cond = mux.getSel();
    if (cond && !cast<FIRRTLType>(cond.getType()).isConst())
      return cond;
  }
  return nullptr;
}
