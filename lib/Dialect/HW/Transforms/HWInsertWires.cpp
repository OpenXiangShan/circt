//===- HWInsertWires.cpp - Insert wires for direct connections --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass inserts hw.wire operations between direct connections involving
// hw.module IOs and seq register/memory operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "hw-insert-wires"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWINSERTWIRES
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {

// Helper function to check if an operation is a target seq operation
static bool isTargetSeqOp(Operation *op) {
  return isa<seq::FirRegOp, seq::FirMemReadOp, seq::FirMemWriteOp,
             seq::FirMemReadWriteOp>(op);
}

// Helper function to check if a value comes from module input
static bool isModuleInput(Value value) { return isa<BlockArgument>(value); }

// Helper function to check if a value needs wire insertion
static bool needsWireInsertion(Value value) {
  // Need wire if it's a module input or comes from a target seq operation
  return isModuleInput(value) ||
         (value.getDefiningOp() && isTargetSeqOp(value.getDefiningOp()));
}

struct HWInsertWiresPass
    : circt::hw::impl::HWInsertWiresBase<HWInsertWiresPass> {
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(&getContext());

    // Map from value to the wire that replaces it to avoid duplicate wires
    DenseMap<Value, Value> valueToWire;

    // Collect all values that need wires and their users
    DenseMap<Value, SmallVector<Operation *>> valueToUsers;

    // Scan all seq operations for input connections that need wires
    module.walk([&](Operation *op) {
      if (isTargetSeqOp(op))
        for (auto operand : op->getOperands())
          if (needsWireInsertion(operand))
            valueToUsers[operand].push_back(op);
    });

    // Scan output operations for connections that need wires
    module.walk([&](hw::OutputOp outputOp) {
      for (auto operand : outputOp.getOperands())
        if (needsWireInsertion(operand))
          valueToUsers[operand].push_back(outputOp);
    });

    // For each value that needs a wire, create one wire and replace all usages
    for (auto &[value, users] : valueToUsers) {
      if (users.empty())
        continue;

      // Create wire only once for this value
      if (valueToWire.find(value) == valueToWire.end()) {
        // Insert after the defining operation or at module start
        if (auto *defOp = value.getDefiningOp())
          builder.setInsertionPointAfter(defOp);
        else
          builder.setInsertionPointToStart(&module.getBody().front());

        auto wire =
            builder.create<hw::WireOp>(value.getLoc(), value.getType(), value);
        valueToWire[value] = wire.getResult();
      }

      Value wireValue = valueToWire[value];

      // Replace all usages in the collected users
      for (auto *user : users)
        for (auto &operand : user->getOpOperands())
          if (operand.get() == value)
            operand.set(wireValue);
    }
  }
};

} // namespace
