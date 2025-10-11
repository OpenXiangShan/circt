//===- HWPartitionModules.cpp - Partition S/C modules by RepCut -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass partitions S and C modules based on repcut_partitions attributes.
// For each unique partition ID, it creates a separate module containing only
// operations assigned to that partition.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

#define DEBUG_TYPE "hw-partition-modules"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWPARTITIONMODULES
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

// Helper function to get partition IDs from an operation
static DenseSet<uint64_t> getPartitionIds(Operation *op) {
  DenseSet<uint64_t> ids;

  auto attr = op->getAttrOfType<ArrayAttr>("repcut_partitions");
  if (!attr)
    return ids;

  for (Attribute elem : attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(elem))
      ids.insert(intAttr.getUInt());
  }

  return ids;
}

// Helper function to collect all partition IDs used in a module
static DenseSet<uint64_t> collectAllPartitionIds(HWModuleOp module) {
  DenseSet<uint64_t> allIds;

  module.walk([&](Operation *op) {
    auto ids = getPartitionIds(op);
    allIds.insert(ids.begin(), ids.end());
  });

  return allIds;
}

// Helper function to check if an operation belongs to a partition
static bool belongsToPartition(Operation *op, uint64_t partitionId) {
  // hw.output always belongs to all partitions
  if (isa<hw::OutputOp>(op))
    return true;

  auto ids = getPartitionIds(op);
  return ids.contains(partitionId);
}

struct HWPartitionModulesPass
    : circt::hw::impl::HWPartitionModulesBase<HWPartitionModulesPass> {
  void runOnOperation() override {
    auto top = getOperation(); // mlir::ModuleOp

    // Collect all HW modules (should be S and C modules)
    SmallVector<HWModuleOp> hwMods;
    top.walk([&](HWModuleOp m) { hwMods.push_back(m); });

    if (hwMods.empty())
      return;

    OpBuilder builder(top.getContext());
    SmallVector<HWModuleOp> modulesToErase;

    // Process each module
    for (HWModuleOp module : hwMods) {
      // Collect all partition IDs used in this module
      DenseSet<uint64_t> partitionIds = collectAllPartitionIds(module);

      if (partitionIds.empty())
        continue; // No partitions found, keep the module as is

      // For each partition ID, create a new module
      for (uint64_t partitionId : partitionIds) {
        // Clone the module
        builder.setInsertionPointAfter(module);
        auto clonedModule = cast<HWModuleOp>(builder.clone(*module));

        // Rename the cloned module
        auto originalName = SymbolTable::getSymbolName(module);
        auto newName = StringAttr::get(
            top.getContext(),
            (originalName.str() + "_P" + Twine(partitionId)).str());
        clonedModule.setSymName(newName);

        Block *body = clonedModule.getBodyBlock();
        auto outputOp = cast<hw::OutputOp>(body->getTerminator());

        // Step 1: Collect valid outputs before erasing operations
        SmallVector<Value> validOutputs;
        SmallVector<unsigned> outputsToKeep;
        for (unsigned i = 0; i < outputOp.getNumOperands(); i++) {
          Value outputValue = outputOp.getOperand(i);
          Operation *defOp = outputValue.getDefiningOp();

          // Keep output if it's from an op in this partition or a block arg
          // Actually block arg should not be here, because we inserted wires
          // for module inputs, but just in case.
          bool keep = isa<BlockArgument>(outputValue) ||
                      (defOp && belongsToPartition(defOp, partitionId));

          if (keep) {
            validOutputs.push_back(outputValue);
            outputsToKeep.push_back(i);
          }
        }

        // Step 2: Remove operations not in this partition
        SmallVector<Operation *> opsToErase;
        clonedModule.walk([&](Operation *op) {
          // Skip the module itself and hw.output
          if (isa<hw::HWModuleOp>(op) || isa<hw::OutputOp>(op))
            return;

          if (!belongsToPartition(op, partitionId))
            opsToErase.push_back(op);
        });

        // Erase operations
        for (Operation *op : opsToErase) {
          op->dropAllUses();
          op->erase();
        }

        // Step 3: Rebuild output operation with valid outputs
        OpBuilder outputBuilder(outputOp);
        outputBuilder.create<hw::OutputOp>(outputOp.getLoc(), validOutputs);
        outputOp->erase();

        // Step 4: Identify unused inputs
        SmallVector<unsigned> unusedInputs;
        for (unsigned i = 0; i < clonedModule.getNumInputPorts(); i++) {
          BlockArgument arg = body->getArgument(i);
          // Check if this input is used by any remaining operation
          if (arg.use_empty())
            unusedInputs.push_back(i);
        }

        // Step 5: Compute which outputs to remove (all outputs not in
        // outputsToKeep)
        SmallVector<unsigned> unusedOutputs;
        for (unsigned i = 0; i < clonedModule.getNumOutputPorts(); i++) {
          if (std::find(outputsToKeep.begin(), outputsToKeep.end(), i) ==
              outputsToKeep.end())
            unusedOutputs.push_back(i);
        }

        // Step 6: Remove unused ports
        llvm::sort(unusedInputs);
        llvm::sort(unusedOutputs);
        clonedModule.erasePorts(unusedInputs, unusedOutputs);
        // Remove corresponding block arguments for inputs
        for (auto it = unusedInputs.rbegin(); it != unusedInputs.rend(); it++)
          body->eraseArgument(*it);
      }

      // Mark original module for erasure
      modulesToErase.push_back(module);
    }

    // Erase original modules after processing all
    for (HWModuleOp module : modulesToErase)
      module->erase();
  }
};

} // namespace
