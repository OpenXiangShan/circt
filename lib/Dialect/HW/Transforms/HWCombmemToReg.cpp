//===- HWCombmemToReg.cpp - Convert combmem to register ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass convert firmem with readLatency=0 to firreg
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "hw-combmem-to-reg"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWCOMBMEMTOREG
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace seq;

namespace {

struct HWCombmemToRegPass
    : circt::hw::impl::HWCombmemToRegBase<HWCombmemToRegPass> {
  void runOnOperation() override;

private:
  // Convert firmem with readLatency=0 to firreg
  void convertToReg(FirMemOp memOp, OpBuilder &builder);

  // Helper to apply write mask for selective bit writing
  Value applyWriteMask(Value writeData, Value writeMask, Value oldElement,
                       Location loc, OpBuilder &builder);
};

} // namespace

void HWCombmemToRegPass::runOnOperation() {
  auto moduleOp = getOperation();
  OpBuilder builder(moduleOp.getContext());

  // Collect all zero read latency FirMemOp to process
  SmallVector<FirMemOp> memOpsToProcess;
  moduleOp.walk([&](FirMemOp op) {
    if (op.getReadLatency() == 0)
      memOpsToProcess.push_back(op);
  });

  // Convert zero latency memory to registers
  for (auto memOp : memOpsToProcess)
    convertToReg(memOp, builder);
}

void HWCombmemToRegPass::convertToReg(FirMemOp memOp, OpBuilder &builder) {
  LLVM_DEBUG(llvm::dbgs() << "Converting zero-latency memory to registers\n");

  auto memType = memOp.getType();
  auto firmemType = cast<seq::FirMemType>(memType);
  // Get element type from FirMemType width
  auto elementType =
      IntegerType::get(memOp->getContext(), firmemType.getWidth());
  auto depth = firmemType.getDepth();

  LLVM_DEBUG(llvm::dbgs() << "Memory depth: " << depth << ", element width: "
                          << firmemType.getWidth() << "\n");

  builder.setInsertionPoint(memOp);

  // Find read and write operations
  SmallVector<FirMemReadOp> readOps;
  SmallVector<FirMemWriteOp> writeOps;
  SmallVector<FirMemReadWriteOp> readWriteOps;

  for (auto *user : memOp->getUsers())
    if (auto readOp = dyn_cast<FirMemReadOp>(user))
      readOps.push_back(readOp);
    else if (auto writeOp = dyn_cast<FirMemWriteOp>(user))
      writeOps.push_back(writeOp);
    else if (auto rwOp = dyn_cast<FirMemReadWriteOp>(user))
      readWriteOps.push_back(rwOp);

  if (readOps.empty() && writeOps.empty() && readWriteOps.empty()) {
    // No users, just remove the memory
    memOp.erase();
    return;
  }

  // TODO: multiple clocks not supported yet
  // Find a representative clock from the operations
  Value clock;
  if (!readOps.empty())
    clock = readOps.front().getClk();
  else if (!writeOps.empty())
    clock = writeOps.front().getClk();
  else
    clock = readWriteOps.front().getClk();

  // --- Create a register array to replace the memory ---

  // Create input placeholder for the register array
  SmallVector<Value> placeholders;
  Value zeroConst = builder.create<hw::ConstantOp>(
      memOp.getLoc(), elementType, builder.getIntegerAttr(elementType, 0));
  for (size_t i = 0; i < depth; i++)
    placeholders.push_back(zeroConst);
  Value input = builder.create<hw::ArrayCreateOp>(memOp.getLoc(), placeholders);

  // Create the register array, and input will be updated later
  // TODO: random init support
  auto regArrayOp = builder.create<FirRegOp>(
      memOp.getLoc(), input, clock, builder.getStringAttr("mem_reg_array"),
      hw::InnerSymAttr{}, Attribute{});

  LLVM_DEBUG(llvm::dbgs() << "Created register array: " << regArrayOp << "\n");

  // Read operations
  for (auto readOp : readOps) {
    builder.setInsertionPointAfter(regArrayOp);

    // Create array read: regArray[address]
    Value readData = builder.create<hw::ArrayGetOp>(
        readOp.getLoc(), regArrayOp.getResult(), readOp.getAddress());

    readOp.replaceAllUsesWith(readData);
    readOp.erase();
    LLVM_DEBUG(llvm::dbgs() << "Replaced read operation with array access\n");
  }

  // Read parts of read-write operations
  for (auto rwOp : readWriteOps) {
    builder.setInsertionPointAfter(regArrayOp);
    Value readData = builder.create<hw::ArrayGetOp>(
        rwOp.getLoc(), regArrayOp.getResult(), rwOp.getAddress());
    rwOp.replaceAllUsesWith(readData);
    LLVM_DEBUG(llvm::dbgs() << "Replaced read operation with array access\n");
  }

  // Write operations
  if (!writeOps.empty() || !readWriteOps.empty()) {
    builder.setInsertionPoint(regArrayOp);

    Value arrayResult = regArrayOp.getResult(); // array result

    SmallVector<Value> nextArrayElements;

    for (size_t i = 0; i < depth; i++) {
      Value index = builder.create<hw::ConstantOp>(
          memOp.getLoc(),
          writeOps.empty() ? readWriteOps.front().getAddress().getType()
                           : writeOps.front().getAddress().getType(),
          builder.getIntegerAttr(
              writeOps.empty() ? readWriteOps.front().getAddress().getType()
                               : writeOps.front().getAddress().getType(),
              i));

      Value newElement =
          builder.create<hw::ArrayGetOp>(memOp.getLoc(), arrayResult, index);

      // Process all write operations for this array element
      for (auto writeOp : writeOps) {
        auto writeAddr = writeOp.getAddress();
        auto writeData = writeOp.getData();
        auto writeEnable = writeOp.getEnable();
        auto writeMask = writeOp.getMask();

        Value addrMatch = builder.create<comb::ICmpOp>(
            writeOp.getLoc(), comb::ICmpPredicate::eq, writeAddr, index);

        Value writeThis = addrMatch;
        if (writeEnable) {
          writeThis = builder.create<comb::AndOp>(writeOp.getLoc(), writeEnable,
                                                  addrMatch);
        }

        // Apply write mask if present
        Value finalWriteData =
            writeMask ? applyWriteMask(writeData, writeMask, newElement,
                                       writeOp.getLoc(), builder)
                      : writeData;

        newElement = builder.create<comb::MuxOp>(writeOp.getLoc(), writeThis,
                                                 finalWriteData, newElement);
      }

      // Process all read-write operations for this array element
      for (auto rwOp : readWriteOps) {
        auto rwAddr = rwOp.getAddress();
        auto writeMode = rwOp.getMode();
        auto writeData = rwOp.getWriteData();
        auto enable = rwOp.getEnable();
        auto writeMask = rwOp.getMask();

        Value addrMatch = builder.create<comb::ICmpOp>(
            rwOp.getLoc(), comb::ICmpPredicate::eq, rwAddr, index);

        Value writeCondition = writeMode;
        if (enable) {
          writeCondition =
              builder.create<comb::AndOp>(rwOp.getLoc(), enable, writeMode);
        }

        Value writeThis = builder.create<comb::AndOp>(
            rwOp.getLoc(), writeCondition, addrMatch);

        // Apply write mask if present
        Value finalWriteData =
            writeMask ? applyWriteMask(writeData, writeMask, newElement,
                                       rwOp.getLoc(), builder)
                      : writeData;

        newElement = builder.create<comb::MuxOp>(rwOp.getLoc(), writeThis,
                                                 finalWriteData, newElement);
      }

      nextArrayElements.push_back(newElement);
    }

    // Update next operator of regArrayOp
    std::reverse(nextArrayElements.begin(), nextArrayElements.end());
    Value nextArray =
        builder.create<hw::ArrayCreateOp>(memOp.getLoc(), nextArrayElements);
    regArrayOp.setOperand(0, nextArray);

    // Clean up old operations
    for (auto writeOp : writeOps) {
      writeOp.erase();
      LLVM_DEBUG(llvm::dbgs() << "Write operation removed\n");
    }

    for (auto rwOp : readWriteOps) {
      rwOp.erase();
      LLVM_DEBUG(llvm::dbgs() << "Read-write operation removed\n");
    }
  }

  // Remove the original memory
  memOp.erase();
  LLVM_DEBUG(llvm::dbgs() << "Zero-latency memory conversion completed\n");
}

Value HWCombmemToRegPass::applyWriteMask(Value writeData, Value writeMask,
                                         Value oldElement, Location loc,
                                         OpBuilder &builder) {
  // Determine mask granularity from memory type
  auto dataType = cast<IntegerType>(writeData.getType());
  auto maskType = cast<IntegerType>(writeMask.getType());
  size_t dataWidth = dataType.getWidth();
  size_t maskWidth = maskType.getWidth();
  size_t maskGran = dataWidth / maskWidth; // granularity: bits per mask bit

  // Expand mask to data width by replicating each mask bit
  SmallVector<Value> maskBits;
  for (size_t i = 0; i < maskWidth; ++i) {
    Value maskBit = builder.create<comb::ExtractOp>(loc, writeMask, i, 1);
    Value replicatedMaskBit =
        builder.create<comb::ReplicateOp>(loc, maskBit, maskGran);
    maskBits.push_back(replicatedMaskBit);
  }

  // Concatenate replicated mask bits to form full-width mask
  // Reverse the order to match little-endian bit ordering
  std::reverse(maskBits.begin(), maskBits.end());
  Value fullMask = builder.create<comb::ConcatOp>(loc, dataType, maskBits);

  // Apply mask to selectively write bits
  // finalData = (fullMask & writeData) | (~fullMask & oldElement)
  Value maskedWriteData = builder.create<comb::AndOp>(loc, fullMask, writeData);
  Value invertedMask = builder.create<comb::XorOp>(
      loc, fullMask,
      builder.create<hw::ConstantOp>(loc, APInt::getAllOnes(dataWidth)));
  Value maskedOldData =
      builder.create<comb::AndOp>(loc, invertedMask, oldElement);

  return builder.create<comb::OrOp>(loc, maskedWriteData, maskedOldData);
}
