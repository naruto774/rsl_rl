/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INFERENCE_RUNTIME_HPP
#define INFERENCE_RUNTIME_HPP

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include "logger.hpp"

#ifdef USE_TORCH
#include <torch/script.h>
#endif

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace InferenceRuntime
{

/**
 * @brief Flattened named output tensor for multi-IO inference.
 *
 * values 以 row-major 展平存储；shape 为对应的张量维度（含 batch）。
 * 无论模型输出是 float / int，这里统一转成 float 返回，方便上层观测拼装。
 */
struct NamedTensorOut
{
    std::vector<int64_t> shape;   ///< Tensor shape, e.g. {1, 21} or {1, 13, 4}
    std::vector<float> values;    ///< Flattened row-major data
};

/**
 * @brief Model interface base class
 *
 * Defines common interface for model loading and inference, supporting both Torch and ONNX backends
 */
class Model
{
public:
    virtual ~Model() = default;

    /**
     * @brief Load model file
     * @param model_path Model file path
     * @return Returns true if loading succeeds, false if it fails
     */
    virtual bool load(const std::string& model_path) = 0;

    /**
     * @brief Check if model is loaded
     * @return Returns true if loaded, false otherwise
     */
    virtual bool is_loaded() const = 0;

    /**
     * @brief Forward inference (single input, supports initializer list)
     * @param inputs Vector of input data vectors (usually single element)
     * @return Inference result vector
     */
    virtual std::vector<float> forward(const std::vector<std::vector<float>>& inputs) = 0;

    /**
     * @brief Multi-input / multi-output inference (named tensors).
     *
     * 用于像 mjlab tracking 这类带多个输入(obs + time_step)和多个输出
     * (actions + baked motion gather 输出)的图。调用方统一用 float 提供输入，
     * runtime 会按模型声明的 element type 自适应转换（如 time_step 的 int64）。
     *
     * @param inputs        name -> 展平的 float 输入数据
     * @param input_shapes  name -> 该输入的 shape（含 batch，如 {1,114} / {1,1}）
     * @param output_names  需要取回的输出名列表
     * @return name -> NamedTensorOut（shape + 展平 float 数据）
     *
     * 默认实现抛异常；仅 ONNXModel 支持。
     */
    virtual std::map<std::string, NamedTensorOut> forward_io(
        const std::map<std::string, std::vector<float>>& inputs,
        const std::map<std::string, std::vector<int64_t>>& input_shapes,
        const std::vector<std::string>& output_names)
    {
        (void)inputs; (void)input_shapes; (void)output_names;
        throw std::runtime_error("forward_io() not supported by this model backend");
    }

    /**
     * @brief Get model type string
     * @return Model type ("torch" or "onnx")
     */
    virtual std::string get_model_type() const = 0;
};

/**
 * @brief Torch model implementation class
 *
 * Model inference implementation based on PyTorch TorchScript
 */
class TorchModel : public Model
{
private:
    bool loaded_ = false;               ///< Whether model is loaded
    std::string model_path_;            ///< Model file path

#ifdef USE_TORCH
    torch::jit::script::Module model_; ///< TorchScript model object
#endif

public:
    TorchModel();
    ~TorchModel();

    bool load(const std::string& model_path) override;
    bool is_loaded() const override { return loaded_; }
    std::vector<float> forward(const std::vector<std::vector<float>>& inputs) override;
    std::string get_model_type() const override { return "torch"; }

private:
#ifdef USE_TORCH
    /**
     * @brief Convert vector data to Torch tensor
     * @param data Input data vector
     * @param shape Tensor shape
     * @return Torch tensor
     */
    torch::Tensor vector_to_torch(const std::vector<float>& data, const std::vector<int64_t>& shape);

    /**
     * @brief Convert Torch tensor to vector data
     * @param tensor Input tensor
     * @return Data vector
     */
    std::vector<float> torch_to_vector(const torch::Tensor& tensor);
#endif
};

/**
 * @brief ONNX model implementation class
 *
 * Model inference implementation based on ONNX Runtime
 */
class ONNXModel : public Model
{
private:
    bool loaded_ = false;               ///< Whether model is loaded
    std::string model_path_;            ///< Model file path

#ifdef USE_ONNX
    std::unique_ptr<Ort::Session> session_;                 ///< ONNX inference session
    std::unique_ptr<Ort::Env> env_;                         ///< ONNX runtime environment
    Ort::MemoryInfo memory_info_;                           ///< Memory information
    std::vector<std::string> input_node_names_;             ///< Input node names
    std::vector<std::string> output_node_names_;            ///< Output node names
    std::vector<std::vector<int64_t>> input_shapes_;        ///< Input shapes
    std::vector<std::vector<int64_t>> output_shapes_;       ///< Output shapes
    std::vector<ONNXTensorElementDataType> input_types_;    ///< Input element types (for int/float dispatch)
#endif

public:
    ONNXModel();
    ~ONNXModel();

    bool load(const std::string& model_path) override;
    bool is_loaded() const override { return loaded_; }
    std::vector<float> forward(const std::vector<std::vector<float>>& inputs) override;
    std::map<std::string, NamedTensorOut> forward_io(
        const std::map<std::string, std::vector<float>>& inputs,
        const std::map<std::string, std::vector<int64_t>>& input_shapes,
        const std::vector<std::string>& output_names) override;
    std::string get_model_type() const override { return "onnx"; }

private:
#ifdef USE_ONNX
    /**
     * @brief Setup input/output node information
     */
    void setup_input_output_info();

    /**
     * @brief Extract data from ONNX outputs
     * @param outputs ONNX inference outputs
     * @return Extracted data vector
     */
    std::vector<float> extract_output_data(const std::vector<Ort::Value>& outputs);
#endif
};

/**
 * @brief Model factory class
 *
 * Responsible for creating and loading different types of models
 */
class ModelFactory
{
public:
    /**
     * @brief Model type enumeration
     */
    enum class ModelType
    {
        TORCH,  ///< TorchScript model
        ONNX,   ///< ONNX model
        AUTO    ///< Automatically detect model type
    };

    /**
     * @brief Create model of specified type
     * @param type Model type
     * @return Model smart pointer
     */
    static std::unique_ptr<Model> create_model(ModelType type = ModelType::AUTO);

    /**
     * @brief Detect model type based on file path
     * @param model_path Model file path
     * @return Detected model type
     */
    static ModelType detect_model_type(const std::string& model_path);

    /**
     * @brief Load model file
     * @param model_path Model file path
     * @param type Model type (default: auto-detect)
     * @return Successfully loaded model smart pointer, returns nullptr on failure
     */
    static std::unique_ptr<Model> load_model(const std::string& model_path, ModelType type = ModelType::AUTO);
};

} // namespace InferenceRuntime

#endif // INFERENCE_RUNTIME_HPP
