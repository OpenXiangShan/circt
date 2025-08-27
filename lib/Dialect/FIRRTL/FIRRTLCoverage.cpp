//===- FIRRTLCoverage.cpp - Code for working with Coverage ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helpers for working with FIRRTL coverage.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLCoverage.h"

using namespace circt;
using namespace firrtl;

Operation *circt::firrtl::findDefOp(Value v, Operation *consumerOp) {
  auto defOp = v.getDefiningOp();
  if (!defOp || defOp->getNumResults() > 1) {
    OpBuilder builder(consumerOp);
    builder.setInsertionPoint(consumerOp);
    auto defName = builder.getStringAttr("cover_dummy");
    defOp = builder.create<WireOp>(consumerOp->getLoc(), v.getType(), defName);
    builder.create<ConnectOp>(defOp->getLoc(), defOp->getResult(0), v);
  }
  return defOp;
};

void circt::firrtl::annotateCoverPoint(Operation *op, const std::string &name,
                                       const std::string &groupName,
                                       CircuitOp &circuit) {
  MLIRContext *context = circuit.getContext();
  OpBuilder builder(circuit);

  auto coverAnno = mlir::DictionaryAttr::get(
      context,
      {
          builder.getNamedAttr(
              "class", builder.getStringAttr("xfuzz.CoverPointAnnotation")),
          builder.getNamedAttr("name", builder.getStringAttr(name)),
          builder.getNamedAttr("group", builder.getStringAttr(groupName)),
      });
  op->setAttr("annotations", builder.getArrayAttr({coverAnno}));
}

void circt::firrtl::annotateCoverPoint(Operation *op,
                                       const std::string &groupName,
                                       CircuitOp &circuit) {
  std::string name;
  if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
    name = nameAttr.str();
  else
    name = op->getName().getStringRef().str();

  annotateCoverPoint(op, name, groupName, circuit);
}
