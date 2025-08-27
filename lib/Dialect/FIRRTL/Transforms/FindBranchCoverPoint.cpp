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

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLCoverage.h"

using namespace circt;
using namespace firrtl;

namespace {
class FindBranchCoverPointPass
    : public FindBranchCoverPointBase<FindBranchCoverPointPass> {
public:
  void runOnOperation() final;

private:
  Value getNonConstBranchCondition(Operation *op);
};
} // namespace

void FindBranchCoverPointPass::runOnOperation() {
  auto circuit = getOperation();

  circuit.walk([&](Operation *op) {
    auto condVal = getNonConstBranchCondition(op);
    if (!condVal)
      return;
    annotateCoverPoint(findDefOp(condVal, op), "branch", circuit);
  });
}

Value FindBranchCoverPointPass::getNonConstBranchCondition(Operation *op) {
  // when (cond) { ... }
  if (auto whenOp = dyn_cast<WhenOp>(op)) {
    auto cond = whenOp.getCondition();
    if (cond && !cond.getType().isConst())
      return cond;
  }
  // Mux(cond, ..., ...)
  else if (auto mux = dyn_cast<MuxPrimOp>(op)) {
    auto cond = mux.getSel();
    if (cond && !cond.getType().cast<FIRRTLType>().isConst())
      return cond;
  }
  return nullptr;
}

std::unique_ptr<mlir::Pass> circt::firrtl::createFindBranchCoverPointPass() {
  return std::make_unique<FindBranchCoverPointPass>();
}
