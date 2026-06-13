/* This file is part of onnx2c.
 *
 * "Quantized GEneral Matrix Multiplication" (QGemm)
 *
 * Modified behavior:
 *
 * Original QGemm calculated:
 *   Y = Quantize(alpha * ((A - a_zp) * (B - b_zp)) + C)
 *
 * This version calculates only the int32 accumulator:
 *   Y_int32 = ((A - a_zp) * (B - b_zp)) + C
 *
 * DequantizeLinear should then consume this int32 output:
 *   Y_float = Y_int32 * output_scale
 *
 * Usually:
 *   output_scale = a_scale * b_scale
 *
 * If alpha is needed and C is not used:
 *   output_scale = alpha * a_scale * b_scale
 *
 * Note:
 *   If C is used and alpha != 1, the original ONNX QGemm math cannot be
 *   represented exactly by a single DequantizeLinear after this int32 QGemm,
 *   because alpha applies to ABrc but not to C in the original implementation.
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
		const Tensor *C = get_number_of_inputs() > 6 ? get_input_tensor(6) : nullptr;
		// Input 7: y_scale, Input 8: y_zero_point are intentionally ignored here.

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
			dst << "\t   DequantizeLinear should convert this output to float." << std::endl;
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
				 * In QGemm, C is an int32 bias/additive tensor in accumulator domain.
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

		OC1 = B->data_dim[1];

		name_input(0, "A");
		name_input(1, "a_scale");
		name_input(2, "a_zero_point");
		name_input(3, "B");
		name_input(4, "b_scale");
		name_input(5, "b_zero_point");

		if (get_number_of_inputs() > 6 && get_input_tensor(6)) {
			name_input(6, "C");
		}

		/*
		 * Inputs 7 and 8 may still exist in the ONNX node, but QGemm no longer
		 * consumes them. The following DequantizeLinear node should use the
		 * appropriate scale instead.
		 */
		if (get_number_of_inputs() > 7 && get_input_tensor(7)) {
			name_input(7, "y_scale");
		}

		if (get_number_of_inputs() > 8 && get_input_tensor(8)) {
			name_input(8, "y_zero_point");
		}

		int M = transA ? A->data_dim[1] : A->data_dim[0];
		int N = transB ? B->data_dim[0] : B->data_dim[1];

		Tensor *t = new Tensor;
		t->data_dim.push_back(M);
		t->data_dim.push_back(N);

		t->data_type = onnx::TensorProto_DataType_INT32;

		register_output(t, "Y");
	}
};

} // namespace toC