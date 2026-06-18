/* This file is part of onnx2c.
 *
 * QLinearConv
 * Quantized Convolution
 *
 * Modified behavior:
 *   The operator writes the raw int32 accumulator instead of the requantized
 *   uint8/int8 output. During resolve(), the ONNX y_scale input is rewritten
 *   to the accumulator-domain scale x_scale * w_scale. A following
 *   DequantizeLinear can therefore compute:
 *
 *     y_float = (float)y_int32 * y_scale
 *
 *   with y_zero_point ignored for int32 accumulators.
 */

#include "spatialfilter.h"

namespace toC {
class QLinearConv : public SpatialFilter {
	public:
	QLinearConv()
	{
		op_name = "QLinearConv";
	}

	const Tensor* get_X(void) const override { return get_input_tensor(0); }
	Tensor* get_W(void) const override { return get_input_tensor(3); }

	void print_output_cell_init(std::ostream& dst, const std::string& y_idx) const override
	{
		INDT_3 << "int32_t a = ";
		if (get_number_of_inputs() >= 9 && get_input_tensor(8)->is_used()) {
			dst << "bias[m]";
		}
		else {
			dst << "0";
		}
		dst << ";" << std::endl;
	}

	void print_output_cell_calc(std::ostream& dst,
	                            const std::string& x_idx,
	                            const std::string& w_idx,
	                            const std::string& y_idx) const override
	{
		INDT_4 << "a += ((int32_t)x" << x_idx << " - x_zero_point[0]) * ((int32_t)w" << w_idx << " - w_zero_point[0]);" << std::endl;
	}

	void print_output_cell_finalize(std::ostream& dst, const std::string& y_idx) const override
	{
		/*
		 * Return the raw int32 convolution accumulator.
		 * Scaling is performed by the following DequantizeLinear node.
		 */
		INDT_3 << "y" << y_idx << " = a;" << std::endl;
	}

	void print(std::ostream& dst) const override
	{
		print_header_info_comment(dst);

		if (options.quant) {
			const Tensor* x_tensor = get_X(); // input 0
			const Tensor* w_tensor = get_W(); // input 3, because QLinearConv overrides get_W()

			int in_channels  = x_tensor->data_dim[1];
			int out_channels = w_tensor->data_dim[0];

			int num_spatial = (int)get_numDataDim();

			int groups = group;
			int bias_enabled = (get_number_of_inputs() >= 9 && get_input_tensor(8)->is_used()) ? 1 : 0;

			std::string input_var  = "x";
			std::string output_var = "y";
			std::string kernel_var = "w";

			std::string x_type = x_tensor->data_type_str();
			std::string w_type = w_tensor->data_type_str();
			std::string bias = bias_enabled ? "(int32_t*) bias" : "(int32_t*) 0";
			if (num_spatial == 1) {
				// X: [N, C, L]
				int width = x_tensor->data_dim[2];

				int kernel_width = (int)kernel_shape[0];
				int stride_width = (int)strides[0];

				// pads: [pad_left, pad_right]
				int pad_left  = (int)pads[0];
				int pad_right = (int)pads[1];

				dst << "QConv1d(" << "(int32_t*)" << output_var << ", "
					<< "(" << x_type << "*)" << input_var << ", "
					<< "(" << w_type << "*)" << kernel_var << ", "
					<< bias << ", "
					<< "x_zero_point, w_zero_point, "
					<< width << ", "
					<< in_channels << ", " << out_channels << ", "
					<< kernel_width << ", "
					<< stride_width << ", "
					<< dilations[0] << ", "
					<< pad_left << ", " << pad_right << ", "
					<< groups
					<< ");" << std::endl;
			}
			else if (num_spatial == 2) {
				// X: [N, C, H, W]
				int height = x_tensor->data_dim[2];
				int width  = x_tensor->data_dim[3];

				int kernel_height = (int)kernel_shape[0];
				int kernel_width  = (int)kernel_shape[1];

				int stride_height = (int)strides[0];
				int stride_width  = (int)strides[1];

				// pads: [top, left, bottom, right]
				int pad_top    = (int)pads[0];
				int pad_left   = (int)pads[1];
				int pad_bottom = (int)pads[2];
				int pad_right  = (int)pads[3];
				dst << "QConv2d(" << "(int32_t*)" << output_var << ", "
					<< "(" << x_type << "*)" << input_var << ", "
					<< "(" << w_type << "*)" << kernel_var << ", "
					<< bias << ", "
					<< "x_zero_point, w_zero_point, "
					<< height << ", " << width << ", "
					<< in_channels << ", " << out_channels << ", "
					<< kernel_height << ", " << kernel_width << ", "
					<< stride_height << ", " << stride_width << ", "
					<< dilations[0] << ", " << dilations[1] << ", "
					<< pad_top << ", " << pad_left << ", " << pad_bottom << ", " << pad_right << ", "
					<< groups
					<< ");" << std::endl;
			}
			else {
				ERROR("QLinearConv: only 1D and 2D convolution supported for options.quant path");
			}
		}
		else {
			print_loop_with_padding_checks(dst);
		}
	}

	void name_scalar_input(unsigned input_no, std::string name)
	{
		name_input(input_no, name);
		if (!(get_input_tensor(input_no)->data_dim.size() == 0 ||
		      (get_input_tensor(input_no)->data_dim.size() == 1 && get_input_tensor(input_no)->data_dim[0] == 1))) {
			ERROR(name << " must be scalar");
		}
	}

	void resolve() override
	{
		/*
		 * This implementation supports per-tensor quantization parameters only.
		 * The accumulator scale is therefore one scalar: x_scale * w_scale.
		 */

		name_input(0, "x");
		name_scalar_input(1, "x_scale");
		name_scalar_input(2, "x_zero_point");

		name_input(3, "w");
		name_scalar_input(4, "w_scale");
		name_scalar_input(5, "w_zero_point");

		name_scalar_input(6, "y_scale");
		name_scalar_input(7, "y_zero_point");

		if (get_number_of_inputs() >= 9 && get_input_tensor(8)->is_used()) {
			name_input(8, "bias");
		}

		rewrite_y_params_for_int32_accumulator();

		resolve_strides();
		resolve_dilations();
		resolve_pads();
		resolve_kernel_shape();

		Tensor* rv = new Tensor;
		rv->data_dim = resolve_output_size();

		/*
		 * The modified QLinearConv returns its raw convolution accumulator
		 * rather than a requantized int8/uint8 value.
		 */
		rv->data_type = onnx::TensorProto_DataType_INT32;

		register_output(rv, "y");

		if (options.quant) {
			if (get_W()->data_buffer == nullptr) {
				ERROR("QLinearConv: quantized weights must be initialized");
			}

			int8_t* w = (int8_t*)get_W()->data_buffer;
			int8_t* new_w = new int8_t[get_W()->data_num_elem()];

			int out_channels = get_W()->data_dim[0];
			int weight_in_channels = get_W()->data_dim[1];
			int num_spatial = (int)get_numDataDim();

			std::vector<int> ddim;

			if (num_spatial == 1) {
				if (get_W()->data_dim.size() != 3) {
					ERROR("QLinearConv: invalid weight dimensions for QConv1d");
				}

				int kernel_width = get_W()->data_dim[2];

				// Reshape weights from
				// [out_channels][in_channels][k]
				// to
				// [out_channels][k][in_channels].
				for (int m = 0; m < out_channels; m++) {
					for (int c = 0; c < weight_in_channels; c++) {
						for (int k = 0; k < kernel_width; k++) {
							int old_idx =
								(m * weight_in_channels + c) *
									kernel_width +
								k;

							int new_idx =
								(m * kernel_width + k) *
									weight_in_channels +
								c;

							new_w[new_idx] = w[old_idx];
						}
					}
				}

				ddim.push_back(out_channels);
				ddim.push_back(kernel_width);
				ddim.push_back(weight_in_channels);
			}
			else if (num_spatial == 2) {
				if (get_W()->data_dim.size() != 4) {
					ERROR("QLinearConv: invalid weight dimensions for QConv2d");
				}

				int kernel_height = get_W()->data_dim[2];
				int kernel_width = get_W()->data_dim[3];

				// Reshape weights from
				// [out_channels][in_channels][kH][kW]
				// to
				// [out_channels][kH][kW][in_channels].
				for (int m = 0; m < out_channels; m++) {
					for (int c = 0; c < weight_in_channels; c++) {
						for (int kh = 0; kh < kernel_height; kh++) {
							for (int kw = 0; kw < kernel_width; kw++) {
								int old_idx =
									((m * weight_in_channels + c) *
										kernel_height +
									kh) *
										kernel_width +
									kw;

								int new_idx =
									((m * kernel_height + kh) *
										kernel_width +
									kw) *
										weight_in_channels +
									c;

								new_w[new_idx] = w[old_idx];
							}
						}
					}
				}

				ddim.push_back(out_channels);
				ddim.push_back(kernel_height);
				ddim.push_back(kernel_width);
				ddim.push_back(weight_in_channels);
			}
			else {
				ERROR("QLinearConv: only 1D and 2D convolution supported for options.quant path");
			}

			get_W()->data_buffer = new_w;
			get_W()->data_dim = ddim;
		}
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
			ERROR("QLinearConv: " << name << " must be a constant scalar float to emit int32 accumulator output");
		}
		if (!is_single_value_tensor(t)) {
			ERROR("QLinearConv: " << name << " must be scalar");
		}
		if (t->data_type != onnx::TensorProto_DataType_FLOAT) {
			ERROR("QLinearConv: " << name << " must be FLOAT");
		}

		return ((float*)t->data_buffer)[0];
	}

	void write_scalar_float(Tensor* t, float value, const std::string& name) const
	{
		if (!t || !t->isConst || t->data_buffer == nullptr) {
			ERROR("QLinearConv: " << name << " must be a constant scalar float to rewrite accumulator scale");
		}
		if (!is_single_value_tensor(t)) {
			ERROR("QLinearConv: " << name << " must be scalar");
		}
		if (t->data_type != onnx::TensorProto_DataType_FLOAT) {
			ERROR("QLinearConv: " << name << " must be FLOAT");
		}

		((float*)t->data_buffer)[0] = value;
	}

	void write_scalar_zero_point(Tensor* t, const std::string& name) const
	{
		if (!t || !t->isConst || t->data_buffer == nullptr) {
			return;
		}
		if (!is_single_value_tensor(t)) {
			ERROR("QLinearConv: " << name << " must be scalar");
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
				ERROR("QLinearConv: unsupported " << name << " data type");
		}
	}

	void rewrite_y_params_for_int32_accumulator() const
	{
		float x_scale = read_scalar_float(get_input_tensor(1), "x_scale");
		float w_scale = read_scalar_float(get_input_tensor(4), "w_scale");
		float accumulator_scale = x_scale * w_scale;

		write_scalar_float(get_input_tensor(6), accumulator_scale, "y_scale");
		write_scalar_zero_point(get_input_tensor(7), "y_zero_point");
	}
};
} // namespace toC
