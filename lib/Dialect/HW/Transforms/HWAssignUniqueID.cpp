//===- HWAssignUniqueID.cpp - Assign unique ID ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <string>

#define DEBUG_TYPE "hw-assign-unique-id"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWASSIGNUNIQUEID
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
// A test pass that simply replaces all wire names with foo_<n>
struct HWAssignUniqueID
    : circt::hw::impl::HWAssignUniqueIDBase<HWAssignUniqueID> {
  void runOnOperation() override;
};
} // namespace

void HWAssignUniqueID::runOnOperation() {
  auto module = getOperation(); // mlir::ModuleOp
  MLIRContext *ctx = &getContext();
  size_t id = 0;

  module.walk([&](Operation *op) {
    if (isa<seq::FirRegOp>(op) || isa<seq::FirMemReadOp>(op) ||
        isa<seq::FirMemWriteOp>(op) || isa<seq::FirMemReadWriteOp>(op)) {
      auto idAttr = IntegerAttr::get(ctx, APSInt(APInt(64, id++)));
      op->setAttr("unique_id", idAttr);
    }
  });
}
