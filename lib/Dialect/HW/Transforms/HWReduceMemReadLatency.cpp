//===- HWReduceMemReadLatency.cpp - Reduce FirMem read latency --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hw-reduce-mem-read-latency"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWREDUCEMEMREADLATENCY
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct HWReduceMemReadLatencyPass
    : circt::hw::impl::HWReduceMemReadLatencyBase<HWReduceMemReadLatencyPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void HWReduceMemReadLatencyPass::runOnOperation() {
  auto moduleOp = getOperation();
  mlir::OpBuilder builder(moduleOp.getContext());

  moduleOp.walk([&](seq::FirMemOp op) {
    auto readLatency = op.getReadLatency();
    if (readLatency > latency) {
      op.setReadLatency(latency);
      LLVM_DEBUG(llvm::dbgs() << "Found FirMem with read latency "
                              << readLatency << ": " << op << "\n");
      for (auto *user : op->getUsers()) {
        auto readOp = llvm::dyn_cast<seq::FirMemReadOp>(user);
        auto rwOp = llvm::dyn_cast<seq::FirMemReadWriteOp>(user);
        auto *rOp = readOp ? readOp : rwOp;
        if (!rOp)
          continue;
        if (readOp)
          LLVM_DEBUG(llvm::dbgs() << "  Found readOp: " << readOp << "\n");
        else
          LLVM_DEBUG(llvm::dbgs() << "  Found rwOp: " << rwOp << "\n");
        builder.setInsertionPointAfter(rOp);
        mlir::Value next = rOp->getResult(0), first;
        for (size_t i = latency; i < readLatency; i++) {
          auto clk = readOp ? readOp.getClk() : rwOp.getClk();
          next = builder.create<seq::FirRegOp>(
              rOp->getLoc(), next, clk, builder.getStringAttr(""),
              static_cast<hw::InnerSymAttr>(nullptr),
              static_cast<Attribute>(nullptr));
          if (i == latency)
            first = next;
          LLVM_DEBUG(llvm::dbgs()
                     << "    Inserted FirRegOp: "
                     << llvm::dyn_cast<seq::FirRegOp>(next.getDefiningOp())
                     << "\n");
        }
        rOp->getResult(0).replaceAllUsesExcept(
            next,
            llvm::SmallPtrSet<mlir::Operation *, 1>{first.getDefiningOp()});
      }
    }
  });
}
