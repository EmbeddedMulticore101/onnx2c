/* This file is part of onnx2c.
 *
 * DequantizeLinear node.
 *
 * Normal ONNX path:
 *   y = (x - x_zero_point) * x_scale
 *
 * Accumulator path used by modified QGemm/QLinearConv:
 *   x       = raw int32 accumulator
 *   x_scale = accumulator scale, already rewritten by the producer to
 *             a_scale * b_scale or x_scale * w_scale
 *   y       = (float)x * x_scale
 *
 * In the accumulator path x_zero_point is intentionally ignored. A raw int32
 * accumulator has zero-point 0, while the ONNX graph may still pass the old
 * uint8 output zero-point from the original quantized operator.
 */

#pragma once

#include "node.h"

namespace toC {

class DequantizeLinear : public Node {
	public:
	DequantizeLinear()
	{
		op_name = "DequantizeLinear";
		axis = 1;
	}

	// Attributes
	int axis;

	virtual void parseAttributes(onnx::NodeProto& node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream& dst) const override;

	private:
	bool is_single_value_param(const Tensor* t) const
	{
		return t->rank() == 0 ||
		       (t->rank() == 1 && t->data_dim[0] == 1);
	}

	std::string param_index_for(const Tensor* t) const
	{
		if (is_single_value_param(t)) {
			return "[0]";
		}

		return "[i" + std::to_string(axis) + "]";
	}
};

void DequantizeLinear::parseAttributes(onnx::NodeProto& node)
{
	for (const auto& a : node.attribute()) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;

		if (a.name() == "axis") {
			axis = parse_attribute_int(a);
		}
		else {
			LOG(ERROR) << "Ignoring attribute " << a.name()
			           << " for node DequantizeLinear/" << onnx_name << std::endl;
		}
	}
}

void DequantizeLinear::resolve(void)
{
	Tensor* x = get_input_tensor(0);
	Tensor* x_scale = get_input_tensor(1);

	name_input(0, "x");
	name_input(1, "x_scale");

	if (axis < 0) {
		axis += x->data_dim.size();
	}

	if (axis < 0 || axis >= (int)x->data_dim.size()) {
		ERROR("DequantizeLinear axis out of range");
	}

	if (get_number_of_inputs() == 3 && get_input_tensor(2)) {
		name_input(2, "x_zero_point");
	}

	Tensor* t = new Tensor;
	t->data_dim = x->data_dim;

	/*
	 * DequantizeLinear output type follows x_scale.
	 * Therefore, if x is int32_t and x_scale is float,
	 * the output y becomes float.
	 */
	t->data_type = x_scale->data_type;

	register_output(t, "y");
}

void DequantizeLinear::print(std::ostream& dst) const
{
	INDT_1 << "/* DequantizeLinear */" << std::endl;

	Tensor* x = get_input_tensor(0);
	Tensor* x_scale = get_input_tensor(1);
	Tensor* x_zero_point =
		(get_number_of_inputs() == 3 && get_input_tensor(2)) ?
		get_input_tensor(2) : nullptr;

	const bool x_is_int32_accumulator =
		x->data_type == onnx::TensorProto_DataType_INT32;

	std::string index;

	for (unsigned loop_axis = 0; loop_axis < x->rank(); loop_axis++) {
		std::string name = "i" + std::to_string(loop_axis);

		INDT_1 << "for (unsigned " << name << " = 0; "
		       << name << " < " << x->data_dim[loop_axis] << "; "
		       << name << "++)" << std::endl;

		index += "[" + name + "]";
	}

	std::string scale_index = param_index_for(x_scale);
	std::string zero_point_index = x_zero_point ?
		param_index_for(x_zero_point) : "[0]";

	INDT_1 << "{" << std::endl;

	if (x_is_int32_accumulator) {
		/*
		 * Modified QGemm/QLinearConv return a raw int32 accumulator.
		 * Its zero point is exactly 0. The producer has rewritten x_scale
		 * to the accumulator-domain scale, so do not subtract the stale
		 * ONNX y_zero_point input here.
		 */
		INDT_2 << "y" << index << " = ((float)x" << index
		       << ") * x_scale" << scale_index << ";" << std::endl;
	}
	else {
		INDT_2 << "y" << index << " = ((float)x" << index;

		if (x_zero_point) {
			dst << " - (float)x_zero_point" << zero_point_index;
		}

		dst << ") * x_scale" << scale_index << ";" << std::endl;
	}

	INDT_1 << "}" << std::endl;
}

} // namespace toC
