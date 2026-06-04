/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference_runtime.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <cmath>

#ifdef USE_TORCH
#include <ATen/Parallel.h>
#endif

namespace InferenceRuntime
{

// ============================================================================
// TorchModel Implementation
// ============================================================================

TorchModel::TorchModel()
{
#ifdef USE_TORCH
    // Set threads before model load
    torch::set_num_threads(1);
#endif
}

TorchModel::~TorchModel()
{
}

bool TorchModel::load(const std::string& model_path)
{
    try
    {
#ifdef USE_TORCH
        // Load TorchScript model onto CPU explicitly.
        // This keeps deployment robust when the JIT archive was exported from CUDA (e.g. cuda:0 tensors).
        model_ = torch::jit::load(model_path, torch::kCPU);
        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Successfully loaded Torch model: " << model_path << std::endl;
        return true;
#else
        std::cout << LOGGER::WARNING << "Torch support not compiled. Please define USE_TORCH." << std::endl;
        loaded_ = false;
        return false;
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Failed to load Torch model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<float> TorchModel::forward(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

#ifdef USE_TORCH
    try
    {
        // Convert input vector to Torch tensor (use first input only)
        const auto& input = inputs[0];
        auto input_tensor = torch::tensor(input, torch::kFloat32).reshape({1, static_cast<int64_t>(input.size())});

        // Disable gradient computation before each forward pass
        torch::autograd::GradMode::set_enabled(false);

        // Ensure single-threaded execution (critical for performance!)
        torch::set_num_threads(1);

        // Execute forward inference
        auto output = model_.forward({input_tensor}).toTensor();

        // Convert output tensor to vector
        return torch_to_vector(output);
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Torch inference error: " << e.what() << std::endl;
        throw;
    }
#else
    throw std::runtime_error("Torch support not compiled");
#endif
}

#ifdef USE_TORCH
torch::Tensor TorchModel::vector_to_torch(const std::vector<float>& data, const std::vector<int64_t>& shape)
{
    // Use torch::tensor() + reshape() to match test program behavior
    auto tensor = torch::tensor(data, torch::kFloat32).reshape(shape);
    return tensor;
}

std::vector<float> TorchModel::torch_to_vector(const torch::Tensor& tensor)
{
    // Ensure tensor is contiguous and on CPU
    auto cpu_tensor = tensor.is_contiguous() ? tensor : tensor.contiguous();
    if (cpu_tensor.device().type() != torch::kCPU)
    {
        cpu_tensor = cpu_tensor.to(torch::kCPU);
    }

    // Get data pointer and size
    float* data_ptr = cpu_tensor.data_ptr<float>();
    int64_t num_elements = cpu_tensor.numel();

    // Copy data to vector
    return std::vector<float>(data_ptr, data_ptr + num_elements);
}
#endif

// ============================================================================
// ONNXModel Implementation
// ============================================================================

ONNXModel::ONNXModel()
#ifdef USE_ONNX
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
#ifdef USE_ONNX
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
#endif
}

ONNXModel::~ONNXModel()
{
#ifdef USE_ONNX
    session_.reset();
    env_.reset();
#endif
}

bool ONNXModel::load(const std::string& model_path)
{
    try
    {
#ifdef USE_ONNX
        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create inference session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

        // Setup input/output information
        setup_input_output_info();

        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Successfully loaded ONNX model: " << model_path << std::endl;
        return true;
#else
        std::cout << LOGGER::WARNING << "ONNX support not compiled. Please define USE_ONNX." << std::endl;
        loaded_ = false;
        return false;
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Failed to load ONNX model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<float> ONNXModel::forward(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

#ifdef USE_ONNX
    try
    {
        // Create memory info
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Get input (use first input only)
        const auto& input = inputs[0];
        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input.data()),
            input.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Prepare input/output names
        const char* input_names[] = {input_node_names_[0].c_str()};
        const char* output_names[] = {output_node_names_[0].c_str()};

        // Execute inference
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // Extract output data
        return extract_output_data(outputs);
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "ONNX inference error: " << e.what() << std::endl;
        throw;
    }
#else
    throw std::runtime_error("ONNX support not compiled");
#endif
}

#ifdef USE_ONNX
void ONNXModel::setup_input_output_info()
{
    // Get input node information
    size_t num_input_nodes = session_->GetInputCount();
    input_node_names_.reserve(num_input_nodes);
    input_shapes_.reserve(num_input_nodes);
    input_types_.reserve(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; ++i)
    {
        // Get input name
        auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        input_node_names_.push_back(std::string(input_name.get()));

        // Get input shape
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();

        // Record element type so forward_io() can dispatch int/float per input
        // (e.g. mjlab time_step is declared as int64 while obs is float32).
        input_types_.push_back(input_tensor_info.GetElementType());

        std::vector<int64_t> shape;
        for (auto dim : input_dims)
        {
            // Handle dynamic dimensions
            if (dim == -1)
            {
                shape.push_back(1);
            }
            else
            {
                shape.push_back(dim);
            }
        }
        input_shapes_.push_back(shape);
    }

    // Get output node information
    size_t num_output_nodes = session_->GetOutputCount();
    output_node_names_.reserve(num_output_nodes);
    output_shapes_.reserve(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; ++i)
    {
        // Get output name
        auto output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_node_names_.push_back(std::string(output_name.get()));

        // Get output shape
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();

        std::vector<int64_t> shape;
        for (auto dim : output_dims)
        {
            // Handle dynamic dimensions
            if (dim == -1)
            {
                shape.push_back(1);
            }
            else
            {
                shape.push_back(dim);
            }
        }
        output_shapes_.push_back(shape);
    }
}

std::vector<float> ONNXModel::extract_output_data(const std::vector<Ort::Value>& outputs)
{
    if (outputs.empty())
    {
        throw std::runtime_error("No outputs from ONNX model");
    }

    // Get first output tensor
    auto& output = outputs[0];
    float* output_data = const_cast<float*>(output.GetTensorData<float>());

    // Calculate total number of output elements
    auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();

    int64_t num_elements = 1;
    for (auto dim : output_shape)
    {
        if (dim > 0)
        {
            num_elements *= dim;
        }
    }

    // Copy output data to vector
    std::vector<float> result(output_data, output_data + num_elements);

    return result;
}

std::map<std::string, NamedTensorOut> ONNXModel::forward_io(
    const std::map<std::string, std::vector<float>>& inputs,
    const std::map<std::string, std::vector<int64_t>>& input_shapes,
    const std::vector<std::string>& output_names)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<const char*> in_names;
    std::vector<Ort::Value> in_values;
    // Backing stores must outlive session_->Run(); reserve so emplace_back never
    // reallocates and invalidates the data pointers held by Ort::Value.
    std::vector<std::vector<float>> float_store;
    std::vector<std::vector<int64_t>> int64_store;
    std::vector<std::vector<int32_t>> int32_store;
    float_store.reserve(inputs.size());
    int64_store.reserve(inputs.size());
    int32_store.reserve(inputs.size());

    for (const auto& kv : inputs)
    {
        const std::string& name = kv.first;
        const std::vector<float>& vals = kv.second;

        int idx = -1;
        for (size_t i = 0; i < input_node_names_.size(); ++i)
        {
            if (input_node_names_[i] == name) { idx = static_cast<int>(i); break; }
        }
        if (idx < 0)
        {
            throw std::runtime_error("forward_io: unknown ONNX input name '" + name + "'");
        }

        std::vector<int64_t> shape;
        auto sit = input_shapes.find(name);
        if (sit != input_shapes.end())
        {
            shape = sit->second;
        }
        else
        {
            shape = { 1, static_cast<int64_t>(vals.size()) };
        }

        in_names.push_back(input_node_names_[idx].c_str());

        const ONNXTensorElementDataType etype = input_types_[idx];
        if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        {
            int64_store.emplace_back();
            auto& buf = int64_store.back();
            buf.resize(vals.size());
            for (size_t i = 0; i < vals.size(); ++i) buf[i] = static_cast<int64_t>(std::llround(vals[i]));
            in_values.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, buf.data(), buf.size(), shape.data(), shape.size()));
        }
        else if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
        {
            int32_store.emplace_back();
            auto& buf = int32_store.back();
            buf.resize(vals.size());
            for (size_t i = 0; i < vals.size(); ++i) buf[i] = static_cast<int32_t>(std::llround(vals[i]));
            in_values.push_back(Ort::Value::CreateTensor<int32_t>(
                memory_info, buf.data(), buf.size(), shape.data(), shape.size()));
        }
        else
        {
            float_store.emplace_back(vals);
            auto& buf = float_store.back();
            in_values.push_back(Ort::Value::CreateTensor<float>(
                memory_info, buf.data(), buf.size(), shape.data(), shape.size()));
        }
    }

    std::vector<const char*> out_names;
    out_names.reserve(output_names.size());
    for (const std::string& on : output_names)
    {
        bool found = false;
        for (const auto& n : output_node_names_) { if (n == on) { found = true; break; } }
        if (!found)
        {
            throw std::runtime_error("forward_io: unknown ONNX output name '" + on + "'");
        }
        out_names.push_back(on.c_str());
    }

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        in_names.data(), in_values.data(), in_values.size(),
        out_names.data(), out_names.size());

    std::map<std::string, NamedTensorOut> result;
    for (size_t i = 0; i < output_names.size(); ++i)
    {
        auto info = outputs[i].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();
        int64_t n = 1;
        for (auto d : shape) { if (d > 0) n *= d; }

        NamedTensorOut td;
        td.shape = shape;
        td.values.resize(static_cast<size_t>(n));

        const ONNXTensorElementDataType et = info.GetElementType();
        if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        {
            const int64_t* p = outputs[i].GetTensorData<int64_t>();
            for (int64_t k = 0; k < n; ++k) td.values[k] = static_cast<float>(p[k]);
        }
        else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
        {
            const int32_t* p = outputs[i].GetTensorData<int32_t>();
            for (int64_t k = 0; k < n; ++k) td.values[k] = static_cast<float>(p[k]);
        }
        else
        {
            const float* p = outputs[i].GetTensorData<float>();
            std::copy(p, p + n, td.values.begin());
        }
        result[output_names[i]] = std::move(td);
    }

    return result;
}
#endif

// ============================================================================
// ModelFactory Implementation
// ============================================================================

std::unique_ptr<Model> ModelFactory::create_model(ModelType type)
{
    switch (type)
    {
        case ModelType::TORCH:
            return std::make_unique<TorchModel>();
        case ModelType::ONNX:
            return std::make_unique<ONNXModel>();
        default:
            return nullptr;
    }
}

ModelFactory::ModelType ModelFactory::detect_model_type(const std::string& model_path)
{
    // Extract file extension from path
    std::filesystem::path path(model_path);
    std::string extension = path.extension().string();

    // Convert to lowercase for case-insensitive comparison
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // Determine model type based on extension
    if (extension == ".pt" || extension == ".pth")
    {
        return ModelType::TORCH;
    }
    else if (extension == ".onnx")
    {
        return ModelType::ONNX;
    }
    else
    {
        throw std::runtime_error("Unknown model file extension: " + extension + ". Supported: .pt, .pth, .onnx");
    }
}

std::unique_ptr<Model> ModelFactory::load_model(const std::string& model_path, ModelType type)
{
    // If type is AUTO, automatically detect model type
    if (type == ModelType::AUTO)
    {
        type = detect_model_type(model_path);
    }

    // Create and load model
    auto model = create_model(type);
    if (model && model->load(model_path))
    {
        return model;
    }
    return nullptr;
}

} // namespace InferenceRuntime
