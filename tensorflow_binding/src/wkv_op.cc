#include <cuda.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/shape_inference.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Externals.
//

#ifdef __cplusplus
extern "C" {
#endif

void cuda_wkv_forward(int B, int T, int C, const float *w, const float *u, const float *k, const float *v, float *y);
void cuda_wkv_backward(int B, int T, int C, const float *w, const float *u, const float *k, const float *v, const float *y,
    const float *gy, float *gw, float *gu, float *gk, float *gv);

#ifdef __cplusplus
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Register ops.
//

REGISTER_OP("WKV_FORWARD")
    .Input("w: float32")        // 0 - w    C
    .Input("u: float32")        // 1 - u:   C
    .Input("k: float32")        // 2 - k:   BxTxC
    .Input("v: float32")        // 3 - v:   BxTxC

    .Output("y: float32")       // 0 - y:   BxTxC

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));  // w
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input));  // u
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &input));  // k
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &input));  // v
      
      c->set_output(0, c->input(3));
      return ::tensorflow::Status::OK();
    });


REGISTER_OP("WKV_BACKWARD")
    .Input("w: float32")        // 0 - w    C
    .Input("u: float32")        // 1 - u    C
    .Input("k: float32")        // 2 - k    BxTxC
    .Input("v: float32")        // 3 - v    BxTxC
    .Input("y: float32")        // 4 - y    BxTxC
    .Input("gy: float32")       // 5 - gy   BxTxC

    .Output("gw: float32")      // 0 - gw   BxC
    .Output("gu: float32")      // 1 - gu   BxC
    .Output("gk: float32")      // 2 - gk   BxTxC
    .Output("gv: float32")      // 3 - gv   BxTxC

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));  // w
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input));  // u
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &input));  // k
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &input));  // v
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 3, &input));  // y
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 3, &input));  // gy
      
      auto B = c->Dim(input, 0);
      auto C = c->Dim(input, 2);
      c->set_output(0, c->Matrix(B, C));    // gw
      c->set_output(1, c->Matrix(B, C));    // gu
      c->set_output(2, c->input(4));        // gk
      c->set_output(3, c->input(4));        // gv
      return ::tensorflow::Status::OK();
    });

namespace tf = tensorflow;

namespace wkv {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  WkvForwardOp
//

class WkvForwardOp : public tf::OpKernel {
  public:
    explicit WkvForwardOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    }

    void Compute(tf::OpKernelContext* ctx) override {
        // Get inputs.
        //
        const tf::Tensor* w;
        const tf::Tensor* u;
        const tf::Tensor* k;
        const tf::Tensor* v;
        OP_REQUIRES_OK(ctx, ctx->input("w", &w));
        OP_REQUIRES_OK(ctx, ctx->input("u", &u));
        OP_REQUIRES_OK(ctx, ctx->input("k", &k));
        OP_REQUIRES_OK(ctx, ctx->input("v", &v));

        // Check inputs.
        //
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(w->shape()),
            tf::errors::InvalidArgument("w is not a vector"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(u->shape()),
            tf::errors::InvalidArgument("u is not a vector"));
        OP_REQUIRES(ctx, k->shape().dims() == 3,
            tf::errors::InvalidArgument("k is not a 3-Tensor"));
        OP_REQUIRES(ctx, v->shape().dims() == 3,
            tf::errors::InvalidArgument("v is not a 3-Tensor"));

        const auto& shape_BTC = v->shape();
        const auto B = shape_BTC.dim_size(0);
        const auto T = shape_BTC.dim_size(1);
        const auto C = shape_BTC.dim_size(2);

        OP_REQUIRES(ctx, w->dim_size(0) == C,
            tf::errors::InvalidArgument("len(w) != channels. ",
                                        "len(w):  ", w->dim_size(0),
                                        " channels: ", C));
        OP_REQUIRES(ctx, u->dim_size(0) == C,
            tf::errors::InvalidArgument("len(u)) != channels. ",
                                        "len(u):  ", u->dim_size(0),
                                        " channels: ", C));
        OP_REQUIRES(ctx, k->dim_size(0) == B,
            tf::errors::InvalidArgument("shape(k)[0] != batch_size. ",
                                        "shape(k)[0]:  ", k->dim_size(0),
                                        " batch_size: ", B));
        OP_REQUIRES(ctx, k->dim_size(1) == T,
            tf::errors::InvalidArgument("shape(k)[1] != time. ",
                                        "shape(k)[1]:  ", k->dim_size(1),
                                        " time: ", T));
        OP_REQUIRES(ctx, k->dim_size(2) == C,
            tf::errors::InvalidArgument("shape(k)[0] != channels. ",
                                        "shape(k)[0]:  ", k->dim_size(2),
                                        " channels: ", C));

        // Prepare tensors.
        //
        auto w_t = w->tensor<float, 1>();
        auto u_t = u->tensor<float, 1>();
        auto k_t = k->tensor<float, 3>();
        auto v_t = v->tensor<float, 3>();
        
        // Allocate outputs.
        //
        tf::Tensor* y = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("y", shape_BTC, &y));
        auto y_t = y->tensor<float, 3>();

        // Run.
        //
        cuda_wkv_forward(B, T, C, w_t.data(), u_t.data(), k_t.data(), v_t.data(), y_t.data());
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  WkvBackwardOp
//

class WkvBackwardOp : public tf::OpKernel {
  public:
    explicit WkvBackwardOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    }

    void Compute(tf::OpKernelContext* ctx) override {
        // Get inputs.
        //
        const tf::Tensor* w;
        const tf::Tensor* u;
        const tf::Tensor* k;
        const tf::Tensor* v;
        const tf::Tensor* y;
        const tf::Tensor* gy;
        OP_REQUIRES_OK(ctx, ctx->input("w", &w));
        OP_REQUIRES_OK(ctx, ctx->input("u", &u));
        OP_REQUIRES_OK(ctx, ctx->input("k", &k));
        OP_REQUIRES_OK(ctx, ctx->input("v", &v));
        OP_REQUIRES_OK(ctx, ctx->input("y", &y));
        OP_REQUIRES_OK(ctx, ctx->input("gy", &gy));

        // Check inputs.
        //
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(w->shape()),
            tf::errors::InvalidArgument("w is not a vector"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(u->shape()),
            tf::errors::InvalidArgument("u is not a vector"));
        OP_REQUIRES(ctx, k->shape().dims() == 3,
            tf::errors::InvalidArgument("k is not a 3-Tensor"));
        OP_REQUIRES(ctx, v->shape().dims() == 3,
            tf::errors::InvalidArgument("v is not a 3-Tensor"));
        OP_REQUIRES(ctx, y->shape().dims() == 3,
            tf::errors::InvalidArgument("y is not a 3-Tensor"));
        OP_REQUIRES(ctx, gy->shape().dims() == 3,
            tf::errors::InvalidArgument("gy is not a 3-Tensor"));

        const auto& shape_BTC = gy->shape();
        const auto B = shape_BTC.dim_size(0);
        const auto T = shape_BTC.dim_size(1);
        const auto C = shape_BTC.dim_size(2);

        OP_REQUIRES(ctx, w->dim_size(0) == C,
            tf::errors::InvalidArgument("len(w) != channels. ",
                                        "len(w):  ", w->dim_size(0),
                                        " channels: ", C));
        OP_REQUIRES(ctx, u->dim_size(0) == C,
            tf::errors::InvalidArgument("len(u)) != channels. ",
                                        "len(u):  ", u->dim_size(0),
                                        " channels: ", C));
        OP_REQUIRES(ctx, k->dim_size(0) == B,
            tf::errors::InvalidArgument("shape(k)[0] != batch_size. ",
                                        "shape(k)[0]:  ", k->dim_size(0),
                                        " batch_size: ", B));
        OP_REQUIRES(ctx, k->dim_size(1) == T,
            tf::errors::InvalidArgument("shape(k)[1] != time. ",
                                        "shape(k)[1]:  ", k->dim_size(1),
                                        " time: ", T));
        OP_REQUIRES(ctx, k->dim_size(2) == C,
            tf::errors::InvalidArgument("shape(k)[2] != channels. ",
                                        "shape(k)[2]:  ", k->dim_size(2),
                                        " channels: ", C));
        OP_REQUIRES(ctx, v->dim_size(0) == B,
            tf::errors::InvalidArgument("shape(v)[0] != batch_size. ",
                                        "shape(v)[0]:  ", v->dim_size(0),
                                        " batch_size: ", B));
        OP_REQUIRES(ctx, v->dim_size(1) == T,
            tf::errors::InvalidArgument("shape(v)[1] != time. ",
                                        "shape(v)[1]:  ", v->dim_size(1),
                                        " time: ", T));
        OP_REQUIRES(ctx, v->dim_size(2) == C,
            tf::errors::InvalidArgument("shape(v)[2] != channels. ",
                                        "shape(v)[2]:  ", v->dim_size(2),
                                        " channels: ", C));
        OP_REQUIRES(ctx, y->dim_size(0) == B,
            tf::errors::InvalidArgument("shape(y)[0] != batch_size. ",
                                        "shape(y)[0]:  ", y->dim_size(0),
                                        " batch_size: ", B));
        OP_REQUIRES(ctx, y->dim_size(1) == T,
            tf::errors::InvalidArgument("shape(y)[1] != time. ",
                                        "shape(y)[1]:  ", y->dim_size(1),
                                        " time: ", T));
        OP_REQUIRES(ctx, y->dim_size(2) == C,
            tf::errors::InvalidArgument("shape(y)[2] != channels. ",
                                        "shape(y)[2]:  ", y->dim_size(2),
                                        " channels: ", C));

        // Prepare tensors.
        //
        auto w_t = w->tensor<float, 1>();
        auto u_t = u->tensor<float, 1>();
        auto k_t = k->tensor<float, 3>();
        auto v_t = v->tensor<float, 3>();
        auto y_t = y->tensor<float, 3>();
        auto gy_t = gy->tensor<float, 3>();

        // Allocate outputs.
        //
        tf::TensorShape shape_BC{B, C};
        tf::Tensor* gw = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("gw", shape_BC, &gw));
        auto gw_t = gw->tensor<float, 2>();

        tf::Tensor* gu = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("gu", shape_BC, &gu));
        auto gu_t = gu->tensor<float, 2>();

        tf::Tensor* gk = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("gk", shape_BTC, &gk));
        auto gk_t = gk->tensor<float, 3>();

        tf::Tensor* gv = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("gv", shape_BTC, &gv));
        auto gv_t = gv->tensor<float, 3>();

        // Run.
        //
        cuda_wkv_backward(B, T, C, w_t.data(), u_t.data(), k_t.data(), v_t.data(), y_t.data(), gy_t.data(),
            gw_t.data(), gu_t.data(), gk_t.data(), gv_t.data());
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Register kernels.
//

REGISTER_KERNEL_BUILDER(
    Name("WKV_FORWARD")
            .Device(::tensorflow::DEVICE_GPU),
        WkvForwardOp
    );

REGISTER_KERNEL_BUILDER(
    Name("WKV_BACKWARD")
            .Device(::tensorflow::DEVICE_GPU),
        WkvBackwardOp
    );
}
