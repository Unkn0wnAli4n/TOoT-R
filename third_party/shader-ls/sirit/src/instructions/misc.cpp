/* This file is part of the sirit project.
 * Copyright (c) 2019 sirit
 * This software may be used and distributed according to the terms of the
 * 3-Clause BSD License
 */

#include "sirit/sirit.h"

#include "stream.h"

namespace Sirit {

Id Module::OpUndef(Id result_type) {
    code->Reserve(3);
    return *code << OpId{spv::Op::OpUndef, result_type} << EndOp{};
}

void Module::OpEmitVertex() {
    code->Reserve(1);
    *code << spv::Op::OpEmitVertex << EndOp{};
}

void Module::OpEndPrimitive() {
    code->Reserve(1);
    *code << spv::Op::OpEndPrimitive << EndOp{};
}

void Module::OpEmitStreamVertex(Id stream) {
    code->Reserve(2);
    *code << spv::Op::OpEmitStreamVertex << stream << EndOp{};
}

void Module::OpEndStreamPrimitive(Id stream) {
    code->Reserve(2);
    *code << spv::Op::OpEndStreamPrimitive << stream << EndOp{};
}

void Module::OpEmitMeshTasksEXT(Id group_count_x, Id group_count_y, Id group_count_z,
                                std::optional<Id> payload) {
    code->Reserve(5);
    *code << spv::Op::OpEmitMeshTasksEXT << group_count_x << group_count_y << group_count_z
          << payload << EndOp{};
}

void Module::OpSetMeshOutputsEXT(Id vertex_count, Id primitive_count) {
    code->Reserve(3);
    *code << spv::Op::OpSetMeshOutputsEXT << vertex_count << primitive_count << EndOp{};
}

} // namespace Sirit
