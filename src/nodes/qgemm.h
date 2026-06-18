/* This file is part of onnx2c.
 *
 * "Quantized GEneral Matrix Multiplication" (QGemm)
 *
 * Modified behavior:
 *
 * Original QGemm calculated a requantized uint8/int8 output.
 * This version calculates only the raw int32 accumulator:
 *
 *   Y_int32 = sum_i((A - a_zp) * (B - b_zp)) + C
 *
 * During resolve(), the ONNX y_scale input is rewritten to the accumulator
 * domain scale:
 *
 *   y_scale = alpha * a_scale * b_scale
 *
 * Therefore, a following DequantizeLinear can correctly compute:
 *
 *   Y_float = (float)Y_int32 * y_scale
 *
 * The following DequantizeLinear must ignore y_zero_point for int32
 * accumulators. If C is present and alpha != 1, this representation is not
 * exact because alpha applies to AB but not to C in QGemm semantics.
 */

#pragma once

#include "node.h"

namespace toC {

class QGemm : public Node {
	public:
	QGemm() {
		op_name = "QGemm";
		alpha = 1.0;
		transA = transB = 0;
	}

	/* Node attributes */
	float alpha;
	int transA; // boolean for 'do the transpose'
	int transB;
	int OC1 = 0;

	/* Parse attributes, if this node has them. */
	virtual void parseAttributes(onnx::NodeProto &node) override {
		for (const auto& a : node.attribute()) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;

			if (a.name() == "alpha") {
				alpha = parse_attribute_float(a);
			}
			else if (a.name() == "transA") {
				transA = parse_attribute_int(a);
			}
			else if (a.name() == "transB") {
				transB = parse_attribute_int(a);
			}
			else {
				ERROR("unknown attribute: " << a.name());
			}
		}
	}

	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{
		const Tensor *A = get_input_tensor(0);
		// Input 1: a_scale, Input 2: a_zero_point
		const Tensor *B = get_input_tensor(3);
		// Input 4: b_scale, Input 5: b_zero_point
		const Tensor *C = (get_number_of_inputs() > 6 && get_input_tensor(6)->is_used()) ?
			get_input_tensor(6) : nullptr;
		// Input 7: y_scale, Input 8: y_zero_point are intentionally ignored here.
		// y_scale was rewritten in resolve() for the following DequantizeLinear.

		std::vector<int> A_dim(
			A->data_dim.begin() + A->data_dim.size() - 2,
			A->data_dim.end()
		);

		std::vector<int> B_dim(
			B->data_dim.begin() + B->data_dim.size() - 2,
			B->data_dim.end()
		);

		int32_t rows = A_dim[0];
		int32_t inner = A_dim[1];
		int32_t bd = B_dim[0];

		if (inner == 0) {
			inner = 1;
		}

		if (options.quant) {
			std::string type_a = "uint8_t";
			std::string type_b = "int8_t";

			std::string C_arg = C ? "(int32_t*) C" : "(int32_t*) 0";

			dst << "\tQGemm("
			    << "(int32_t*) Y, "
			    << "(" << type_a << "*) A, "
			    << "(" << type_b << "*) B, "
			    << C_arg << ", "
			    << "(" << type_a << "*) a_zero_point, "
			    << "(" << type_b << "*) b_zero_point, "
			    << rows << ", "
			    << inner << ", "
			    << bd
			    << ");\n";
		}
		else {
			int C0, C1;
			C0 = C1 = 0;

			if (C && C->data_dim.size() > 0) {
				C0 = C->data_dim[0];

				if (C->rank() > 1) {
					C1 = C->data_dim[1];
				}
			}

			int M = transA ? A->data_dim[1] : A->data_dim[0]; // rows
			int K = transA ? A->data_dim[0] : A->data_dim[1]; // inner
			int N = transB ? B->data_dim[0] : B->data_dim[1]; // columns

			dst << "\t/* QGemm */" << std::endl;
			dst << "\t/* Modified QGemm: outputs int32 accumulator." << std::endl;
			dst << "\t   DequantizeLinear converts the accumulator to float." << std::endl;
			dst << "\t   y_scale has been rewritten to alpha * a_scale * b_scale." << std::endl;
			dst << "\t   alpha   = " << alpha << std::endl;
			dst << "\t   transA  = " << transA << std::endl;
			dst << "\t   transB  = " << transB << std::endl;
			dst << "\t */" << std::endl;

			dst << "\tconst int M = " << M << ";" << std::endl;
			dst << "\tconst int K = " << K << ";" << std::endl;
			dst << "\tconst int N = " << N << ";" << std::endl;


			dst << "\tint32_t a_zp = "
			    << constant_acces_code("a_zero_point[0]")
			    << ";"
			    << std::endl;

			dst << "\tint32_t b_zp = "
			    << constant_acces_code("b_zero_point[0]")
			    << ";"
			    << std::endl;

			std::string A_el = transA ? "A[i][r]" : "A[r][i]";
			std::string B_idx = transB ? "[c][i]" : "[i][c]";

			std::string C_idx;

			if (C) {
				C_idx = "";

				int dim;

				switch (C->rank()) {
					case 0:
						ERROR("Unimplemented: scalar C in QGemm");
						break;

					case 1:
						dim = C->data_dim[0];

						if (dim == M) {
							C0 = M;
							C1 = 1;
						}
						else if (dim == N) {
							C0 = 1;
							C1 = N;
						}
						else if (dim == 1) {
							C0 = 1;
							C1 = 1;
						}
						else {
							ERROR("C dimension mismatch in QGemm");
						}

						break;

					case 2:
						C0 = C->data_dim[0];
						C1 = C->data_dim[1];
						break;

					default:
						ERROR("C has too many dimensions in QGemm");
				}

				if (C0 <= 1) {
					C_idx += "[0]";
				}
				else {
					C_idx += "[r]";
				}

				if (C1 <= 1) {
					C_idx += "[0]";
				}
				else {
					C_idx += "[c]";
				}

				/*
				 * In this modified QGemm, C is an int32 additive tensor in
				 * accumulator domain.
				 */
				INDT_1 << "int32_t (*C_)[" << C1 << "] = "
				       << "(int32_t(*)[" << C1 << "])C;"
				       << std::endl;
			}

			INDT_1 << "for (uint32_t r = 0; r < M; r++)" << std::endl;
			INDT_2 << "for (uint32_t c = 0; c < N; c++) {" << std::endl;

			INDT_3 << "int32_t ABrc = 0;" << std::endl;

			INDT_3 << "for (uint32_t i = 0; i < K; i++) {" << std::endl;

			INDT_4 << "int32_t a_val = (int32_t)"
			       << A_el
			       << " - a_zp;"
			       << std::endl;

			INDT_4 << "int32_t b_val = (int32_t)"
			       << constant_acces_code("B" + B_idx)
			       << " - b_zp;"
			       << std::endl;

			INDT_4 << "ABrc += a_val * b_val;" << std::endl;

			INDT_3 << "}" << std::endl;

			if (C) {
				INDT_3 << "ABrc += C_" << C_idx << ";" << std::endl;
			}

			INDT_3 << "Y[r][c] = ABrc;" << std::endl;

			INDT_2 << "}" << std::endl;
		}
	}

	virtual void resolve(void) override
	{
		if (get_number_of_inputs() < 6) {
			ERROR("Not enough inputs for QGemm");
		}

		const Tensor *A = get_input_tensor(0);
		const Tensor *B = get_input_tensor(3);
		const Tensor *C = (get_number_of_inputs() > 6 && get_input_tensor(6)->is_used()) ?
			get_input_tensor(6) : nullptr;

		if (C && alpha != 1.0f) {
			ERROR("QGemm: int32-accumulator split does not support C with alpha != 1 exactly");
		}

		OC1 = B->data_dim[1];

		name_input(0, "A");
		name_input(1, "a_scale");
		name_input(2, "a_zero_point");
		name_input(3, "B");
		name_input(4, "b_scale");
		name_input(5, "b_zero_point");

		if (get_number_of_inputs() > 6 && get_input_tensor(6)->is_used()) {
			name_input(6, "C");
		}

		/*
		 * Inputs 7 and 8 may still exist in the ONNX node. QGemm no longer
		 * consumes them directly, but y_scale is rewritten so the following
		 * DequantizeLinear receives the accumulator-domain scale.
		 */
		if (get_number_of_inputs() > 7 && get_input_tensor(7)) {
			name_input(7, "y_scale");
		}

		if (get_number_of_inputs() > 8 && get_input_tensor(8)) {
			name_input(8, "y_zero_point");
		}

		rewrite_y_params_for_int32_accumulator();

		int M = transA ? A->data_dim[1] : A->data_dim[0];
		int N = transB ? B->data_dim[0] : B->data_dim[1];

		Tensor *t = new Tensor;
		t->data_dim.push_back(M);
		t->data_dim.push_back(N);

		t->data_type = onnx::TensorProto_DataType_INT32;

		register_output(t, "Y");
	}

	private:
	bool is_single_value_tensor(const Tensor* t) const
	{
		return t->rank() == 0 ||
		       (t->rank() == 1 && t->data_dim[0] == 1);
	}

	float read_scalar_float(const Tensor* t, const std::string& name) const
	{
		if (!t || !t->isConst || t->data_buffer == nullptr) {
			ERROR("QGemm: " << name << " must be a constant scalar float to emit int32 accumulator output");
		}
		if (!is_single_value_tensor(t)) {
			ERROR("QGemm: " << name << " must be scalar");
		}
		if (t->data_type != onnx::TensorProto_DataType_FLOAT) {
			ERROR("QGemm: " << name << " must be FLOAT");
		}

		return ((float*)t->data_buffer)[0];
	}

	void write_scalar_float(Tensor* t, float value, const std::string& name) const
	{
		if (!t || !t->isConst || t->data_buffer == nullptr) {
			ERROR("QGemm: " << name << " must be a constant scalar float to rewrite accumulator scale");
		}
		if (!is_single_value_tensor(t)) {
			ERROR("QGemm: " << name << " must be scalar");
		}
		if (t->data_type != onnx::TensorProto_DataType_FLOAT) {
			ERROR("QGemm: " << name << " must be FLOAT");
		}

		((float*)t->data_buffer)[0] = value;
	}

	void write_scalar_zero_point(Tensor* t, const std::string& name) const
	{
		if (!t || !t->isConst || t->data_buffer == nullptr) {
			return;
		}
		if (!is_single_value_tensor(t)) {
			ERROR("QGemm: " << name << " must be scalar");
		}

		switch (t->data_type) {
			case onnx::TensorProto_DataType_INT8:
				((int8_t*)t->data_buffer)[0] = 0;
				break;
			case onnx::TensorProto_DataType_UINT8:
				((uint8_t*)t->data_buffer)[0] = 0;
				break;
			case onnx::TensorProto_DataType_INT32:
				((int32_t*)t->data_buffer)[0] = 0;
				break;
			default:
				ERROR("QGemm: unsupported " << name << " data type");
		}
	}

	void rewrite_y_params_for_int32_accumulator() const
	{
		if (get_number_of_inputs() <= 7 || !get_input_tensor(7)->is_used()) {
			return;
		}

		float a_scale = read_scalar_float(get_input_tensor(1), "a_scale");
		float b_scale = read_scalar_float(get_input_tensor(4), "b_scale");
		float accumulator_scale = alpha * a_scale * b_scale;

		write_scalar_float(get_input_tensor(7), accumulator_scale, "y_scale");

		if (get_number_of_inputs() > 8 && get_input_tensor(8)) {
			write_scalar_zero_point(get_input_tensor(8), "y_zero_point");
		}
	}
};

} // namespace toC
