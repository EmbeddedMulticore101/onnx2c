/* This file is part of onnx2c.
 *
 * MatMulInteger
 * Matrix multiplication with integers.
 * In contrast to MatMul (which allows floats only)
 * MatMulInteger takes optional input zero-point bias terms.
 */

#include "abstractmatmul.h"

namespace toC {

class MatMulInteger : public AbstractMatMul {
 private:
	bool has_input(unsigned input_no) const
	{
		return get_number_of_inputs() > input_no &&
		       get_input_tensor(input_no) != nullptr &&
		       get_input_tensor(input_no)->is_used();
	}

	void validate_per_tensor_zero_point(unsigned zero_point_input,
	                                    unsigned data_input,
	                                    const char* input_name) const
	{
		const Tensor* zero_point = get_input_tensor(zero_point_input);

		// ONNX allows scalar zero points. Some exporters instead encode the
		// same per-tensor value as a one-element vector, so accept both.
		const bool scalar = zero_point->is_scalar();
		const bool one_element_vector =
		    zero_point->rank() == 1 && zero_point->data_dim[0] == 1;

		if (!scalar && !one_element_vector) {
			ERROR(input_name
			      << " must be a scalar or a one-element tensor in this "
			         "onnx2c implementation (per-axis zero points are not yet implemented)");
		}

		if (zero_point->data_type != get_input_tensor(data_input)->data_type) {
			ERROR(input_name << " must have the same element type as its matrix input");
		}
	}

 public:
	MatMulInteger()
	{
		op_name = "MatMulInteger";
	}

	void print_multiply_accumulate(std::ostream& dst,
	                               const std::string& y_idx,
	                               const std::string& a_idx,
	                               const std::string& b_idx) const override
	{
		// Scalar tensors are passed to generated node functions as pointers,
		// while a one-element vector also supports [0]. Optional zero points
		// default to zero according to the ONNX specification.
		const std::string a_zero_point = has_input(2) ? "a_zero_point[0]" : "0";
		const std::string b_zero_point = has_input(3) ? "b_zero_point[0]" : "0";

		INDT_4 << y_idx << " += (" << a_idx << " - " << a_zero_point << ") * ("
		       << b_idx << " - " << b_zero_point << ");" << std::endl;
	}

	void resolve(void) override
	{
		name_input(0, "A");
		name_input(1, "B");

		if (has_input(2)) {
			name_input(2, "a_zero_point");
			validate_per_tensor_zero_point(2, 0, "a_zero_point");
		}

		if (has_input(3)) {
			name_input(3, "b_zero_point");
			validate_per_tensor_zero_point(3, 1, "b_zero_point");
		}

		Tensor* y = new Tensor;
		y->data_dim = resolve_shape();
		y->data_type = onnx::TensorProto_DataType_INT32;
		register_output(y, "Y");
	}
};

} // namespace toC
