//===- HWSplitSeqComb.cpp - Split flattened module into S/C -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass takes a flattened HW module and splits it into two modules:
// S (sequential) and C (combinational).
// - S contains only seq::FirRegOp, seq::FirMemReadOp, seq::FirMemWriteOp,
//   seq::FirMemReadWriteOp, seq::FirMemOp, and hw::ConstantOp.
// - C contains the remaining logic (combinational and wires, constants kept).
// Boundary values between S and C are materialized as new IO ports.
// Constants are kept in both modules and never routed via IO.
//
// The transform duplicates the original hw.module into two modules, then:
//  - In S: add inputs for non-constant operands of seq ops, add outputs for
//          results of seq ops, and erase other ops.
//  - In C: add outputs for values driving seq-op operands, add inputs for
//          results of seq ops, replace their uses, then erase seq ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

#define DEBUG_TYPE "hw-split-seq-comb"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWSPLITSEQCOMB
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

// The boundary seq ops which define the S<->C interface.
static bool isSeqBoundaryOp(Operation *op) {
  return isa<seq::FirRegOp, seq::FirMemReadOp, seq::FirMemWriteOp,
             seq::FirMemReadWriteOp>(op);
}

static bool isConst(Value v) { return v.getDefiningOp<hw::ConstantOp>(); }

// Collect a unique input for a given value if not yet present.
// Returns the index in the collection, or the existing mapped value's index.
static size_t
collectUniqueInput(DenseMap<Value, size_t> &map,
                   SmallVector<std::pair<StringAttr, Type>> &inputs, Value v,
                   StringRef baseName, MLIRContext *ctx) {
  if (auto it = map.find(v); it != map.end())
    return it->second; // Return existing index

  size_t idx = inputs.size();
  auto ty = v.getType();
  std::string name = (baseName + Twine(idx)).str();
  inputs.emplace_back(StringAttr::get(ctx, name), ty);
  map.try_emplace(v, idx);

  return idx;
}

// Collect a unique output for a given value if not yet present.
// Returns true if the value was added to the collection.
static void
collectUniqueOutput(DenseSet<Value> &added,
                    SmallVector<std::pair<StringAttr, Value>> &outputs, Value v,
                    StringRef baseName, MLIRContext *ctx) {
  if (added.contains(v))
    return; // Already added

  std::string name = (baseName + Twine(added.size())).str();
  outputs.emplace_back(StringAttr::get(ctx, name), v);
  added.insert(v);
}

struct HWSplitSeqCombPass
    : circt::hw::impl::HWSplitSeqCombBase<HWSplitSeqCombPass> {
  void runOnOperation() override {
    auto top = getOperation(); // mlir::ModuleOp

    SmallVector<HWModuleOp> hwMods;
    top.walk([&](HWModuleOp m) { hwMods.push_back(m); });
    if (hwMods.empty())
      return;
    // Split the first (and should be the only) flattened module.
    HWModuleOp orig = hwMods.front();

    // Clone two copies: S and C directly into the module after the original.
    OpBuilder builder(top.getContext());
    builder.setInsertionPointAfter(orig);
    auto sClone = cast<HWModuleOp>(builder.clone(*orig));
    builder.setInsertionPointAfter(sClone);
    auto cClone = cast<HWModuleOp>(builder.clone(*orig));

    // Give them new names.
    auto sym = SymbolTable::getSymbolName(orig);
    auto sName = StringAttr::get(top.getContext(), (sym.str() + "_S"));
    auto cName = StringAttr::get(top.getContext(), (sym.str() + "_C"));
    sClone.setSymName(sName);
    cClone.setSymName(cName);

    // Remove the original module.
    orig->erase();

    // Record original port counts for S before edits.
    size_t sOrigInputs = sClone.getNumInputPorts();
    size_t sOrigOutputs = sClone.getNumOutputPorts();

    // Build S: add inputs for non-const operands of seq ops; outputs for
    // results of seq ops; erase other ops (except constants and terminator).
    {
      // Step 1: remove original outputs to clear the port list and replace
      // original terminator with empty output op.
      SmallVector<unsigned> eraseOutputs(sOrigOutputs);
      std::iota(eraseOutputs.begin(), eraseOutputs.end(), 0);
      sClone.erasePorts({}, eraseOutputs);
      auto *term = sClone.getBodyBlock()->getTerminator();
      auto outOp = cast<hw::OutputOp>(term);
      OpBuilder outBuilder(outOp);
      outBuilder.create<hw::OutputOp>(outOp.getLoc());
      outOp.erase();

      // Step 2: walk seq ops to add inputs/outputs for their operands/results.
      DenseMap<Value, size_t> valToInputIdx;
      DenseSet<Value> exported;
      SmallVector<std::pair<StringAttr, Type>> sInputs;
      SmallVector<std::pair<StringAttr, Value>> sOutputs;

      // Step 2.1: collect inputs and outputs for operands and record which
      // operands need replacement
      SmallVector<std::pair<OpOperand *, size_t>> operandsToReplace;
      for (Operation &op : *sClone.getBodyBlock()) {
        if (!isSeqBoundaryOp(&op))
          continue;

        for (OpOperand &operand : op.getOpOperands()) {
          Value v = operand.get();
          // Skip constants and memory handles.
          if (isConst(v) || v.getDefiningOp<seq::FirMemOp>())
            continue;

          size_t idx = collectUniqueInput(valToInputIdx, sInputs, v, "s_in_",
                                          sClone.getContext());
          operandsToReplace.push_back({&operand, idx});
        }

        for (Value res : op.getResults())
          collectUniqueOutput(exported, sOutputs, res, "s_out_",
                              sClone.getContext());
      }

      // Step 2.2: Add all collected inputs and outputs at once
      if (!sInputs.empty()) {
        // Add all inputs to the module at once using appendInputs
        sClone.appendInputs(sInputs);

        // Now update the operands with the newly created block arguments
        Block *sBody = sClone.getBodyBlock();
        for (auto [operand, idx] : operandsToReplace) {
          BlockArgument arg = sBody->getArgument(sOrigInputs + idx);
          operand->set(arg);
        }
      }

      if (!sOutputs.empty())
        sClone.appendOutputs(sOutputs);

      // Step 3: erase non-allowed ops in S by dropping all uses then deleting.
      SmallVector<Operation *> toErase;
      for (Operation &op : llvm::make_early_inc_range(*sClone.getBodyBlock())) {
        if (isa<hw::OutputOp>(&op) || isa<seq::FirMemOp>(&op) ||
            isSeqBoundaryOp(&op) || isa<hw::ConstantOp>(&op))
          continue;
        op.dropAllUses();
        op.erase();
      }

      // Step 4: Remove all original inputs and the corresponding block args.
      SmallVector<unsigned> eraseInputs(sOrigInputs);
      std::iota(eraseInputs.begin(), eraseInputs.end(), 0);
      sClone.erasePorts(eraseInputs, {});
      Block *body = sClone.getBodyBlock();
      body->eraseArguments(0, sOrigInputs);
    }

    // Build C: add outputs for seq-op operands (to S), inputs for seq-op
    // results (from S). Replace seq-op results uses and then erase seq ops.
    {
      DenseSet<Value> cOutAdded; // values exported to S

      // Collect boundary seq ops and memory decls first to avoid invalidation
      // during edits.
      SmallVector<Operation *> seqOps;
      SmallVector<Operation *> memOps;
      cClone.walk([&](Operation *op) {
        if (isSeqBoundaryOp(op))
          seqOps.push_back(op);
        else if (isa<seq::FirMemOp>(op))
          memOps.push_back(op);
      });

      // Build mapping of values to their connected sequential ops' unique_ids
      SmallVector<std::pair<StringAttr, Value>> cOutputs;

      // Collect outputs for all unique values with complete id sets
      for (Operation *op : seqOps)
        for (Value v : op->getOperands()) {
          // Skip constants and memory handles.
          if (isConst(v) || v.getDefiningOp<seq::FirMemOp>())
            continue;

          collectUniqueOutput(cOutAdded, cOutputs, v, "to_s_",
                              cClone.getContext());
        }

      // Add all collected outputs at once
      if (!cOutputs.empty())
        cClone.appendOutputs(cOutputs);

      // Collect all unique seq results that need inputs and record them for
      // replacement
      SmallVector<std::pair<StringAttr, Type>> cInputs;
      DenseMap<Value, size_t> cValueToInputIdx;
      SmallVector<Value> valuesToReplace;
      for (Operation *op : seqOps)
        for (Value res : op->getResults()) {
          // Collect this value if not already collected
          size_t oldSize = cInputs.size();
          collectUniqueInput(cValueToInputIdx, cInputs, res, "from_s_",
                             cClone.getContext());
          // If this is a newly collected value, record it for replacement
          if (cInputs.size() > oldSize)
            valuesToReplace.push_back(res);
        }

      // Add all collected inputs at once and update uses
      if (!cInputs.empty()) {
        size_t numOrigInputs = cClone.getNumInputPorts();

        // Add all inputs to the module at once using appendInputs
        cClone.appendInputs(cInputs);

        // Replace all uses with the newly created block arguments
        Block *cBody = cClone.getBodyBlock();
        for (Value res : valuesToReplace) {
          auto it = cValueToInputIdx.find(res);
          size_t idx = it->second;
          BlockArgument arg = cBody->getArgument(numOrigInputs + idx);
          res.replaceAllUsesWith(arg);
        }
      }

      // Finally erase all seq ops and memory ops from C.
      for (Operation *op : seqOps)
        op->erase();
      for (Operation *op : memOps)
        op->erase();
    }
  }
};

} // namespace
