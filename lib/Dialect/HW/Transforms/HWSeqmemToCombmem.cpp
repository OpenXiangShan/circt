//===- HWSeqmemToCombmem.cpp - Convert Seqmem to Combmem --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <circt/Dialect/Seq/SeqOps.h>

#define DEBUG_TYPE "hw-seqmem-to-combmem"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWSEQMEMTOCOMBMEM
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
// A test pass that simply replaces all wire names with foo_<n>
struct HWSeqmemToCombmemPass
    : circt::hw::impl::HWSeqmemToCombmemBase<HWSeqmemToCombmemPass> {
  void runOnOperation() override;
};
} // namespace

void HWSeqmemToCombmemPass::runOnOperation() {
  auto moduleOp = getOperation();
  mlir::OpBuilder builder(moduleOp.getContext());

  moduleOp.walk([&](seq::FirMemOp op) {
    auto readLatency = op.getReadLatency();
    if (readLatency > 0) {
      op.setReadLatency(0);
      LLVM_DEBUG(llvm::dbgs() << "Found seqmem: " << op << "\n");
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
        for (size_t i = 0; i < readLatency; i++) {
          auto clk = readOp ? readOp.getClk() : rwOp.getClk();
          next = builder.create<seq::FirRegOp>(
              rOp->getLoc(), next, clk, builder.getStringAttr(""),
              static_cast<hw::InnerSymAttr>(nullptr),
              static_cast<Attribute>(nullptr));
          if (i == 0)
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
