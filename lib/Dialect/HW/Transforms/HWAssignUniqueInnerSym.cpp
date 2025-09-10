//===- HWAssignUniqueInnerSym.cpp - Assign unique InnerSym ------*- C++ -*-===//
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

#define DEBUG_TYPE "hw-assign-unique-inner-sym"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWASSIGNUNIQUEINNERSYM
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
// A test pass that simply replaces all wire names with foo_<n>
struct HWAssignUniqueInnerSymPass
    : circt::hw::impl::HWAssignUniqueInnerSymBase<HWAssignUniqueInnerSymPass> {
  void runOnOperation() override;
};
} // namespace

void HWAssignUniqueInnerSymPass::runOnOperation() {
  auto module = getOperation(); // mlir::ModuleOp
  MLIRContext *ctx = &getContext();

  hw::InnerSymbolNamespace innerSymNS(module);

  auto assignIfMissing = [&](Operation *op, auto getAttrFn, auto setAttrFn,
                             StringRef name) {
    auto attr = getAttrFn();
    if (attr && attr.getSymIfExists(0))
      return;

    auto sym = StringAttr::get(ctx, innerSymNS.newName(name));
    auto prop = hw::InnerSymPropertiesAttr::get(sym);

    SmallVector<hw::InnerSymPropertiesAttr, 1> props{prop};
    if (attr)
      llvm::append_range(props, attr.getProps());
    auto newAttr = hw::InnerSymAttr::get(ctx, props);
    setAttrFn(newAttr);
  };

  module.walk([&](Operation *op) {
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      assignIfMissing(
          op, [&]() { return reg.getInnerSymAttr(); },
          [&](auto a) { reg.setInnerSymAttr(a); }, "unique_firreg");
      return;
    }
    if (auto r = dyn_cast<seq::FirMemOp>(op)) {
      assignIfMissing(
          op, [&]() { return r.getInnerSymAttr(); },
          [&](auto a) { r.setInnerSymAttr(a); }, "unique_firmem");
      return;
    }
  });

  // InnerSymbolTableCollection innerSymTables;
  // SymbolTable symbolTable(module);
  // auto *mod = symbolTable.lookup("SimTop");
  // InnerSymbolTable &innerSymTable = innerSymTables.getInnerSymbolTable(mod);
  // auto firreg = innerSymTable.lookupOp<seq::FirRegOp>("unique_firreg_3");
  // LLVM_DEBUG(llvm::dbgs() << "Found firreg: " << firreg << "\n");
  // auto firmem = innerSymTable.lookupOp<seq::FirMemOp>("unique_firmem_0");
  // LLVM_DEBUG(llvm::dbgs() << "Found firmem: " << firmem << "\n");
}
