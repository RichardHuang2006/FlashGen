/**
 * flashgen_bindings.cpp — PyBind11 extension module: flashgen._C
 *
 * Exposes the C++ InferenceEngine and supporting types to Python.
 * This bridges the Python API (flashgen.engine.AsyncLLMEngine) with
 * the C++ runtime (InferenceEngine in csrc/runtime/pipeline.cpp).
 *
 * Threading model:
 *   The C++ engine runs on a background C++ thread. The streaming callback
 *   (InferenceRequest::Callback) is called from that thread. To safely call
 *   back into Python, we must:
 *     1. Acquire the Python GIL inside the callback lambda.
 *     2. Release it before returning to C++.
 *   This is done via py::gil_scoped_acquire inside the std::function wrapper.
 *
 * Memory ownership:
 *   Python passes raw GPU pointers (via tensor.data_ptr()) for weight tensors.
 *   The C++ engine does NOT own these — Python-side torch.Tensor objects must
 *   remain alive for the lifetime of the engine. The bindings document this.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "model_config.hpp"
#include "request.hpp"
#include "block_allocator.hpp"
#include "paged_kv_cache.cuh"
#include "prefix_cache.hpp"
#include "scheduler.hpp"
#include "transformer.hpp"
#include "pipeline.hpp"

namespace py = pybind11;

// ── Helper: convert Python dict of weight pointers → TransformerWeights ──────

static TransformerWeights dict_to_weights(
    const py::dict& d,
    int n_layers)
{
    TransformerWeights w;
    w.layers.resize(n_layers);

    auto get_ptr = [&](const char* key) -> float* {
        if (!d.contains(key)) return nullptr;
        py::object obj = d[key];
        if (obj.is_none()) return nullptr;
        return reinterpret_cast<float*>(obj.cast<int64_t>());
    };

    w.token_emb  = get_ptr("token_emb");
    w.pos_emb    = get_ptr("pos_emb");
    w.ln_f_gamma = get_ptr("ln_f_gamma");
    w.ln_f_beta  = get_ptr("ln_f_beta");
    w.lm_head    = get_ptr("lm_head");

    for (int i = 0; i < n_layers; ++i) {
        auto& lw = w.layers[i];
        auto layer_get = [&](const char* suffix) -> float* {
            std::string key = "layers." + std::to_string(i) + "." + suffix;
            return get_ptr(key.c_str());
        };

        lw.ln1_gamma = layer_get("ln1_gamma");
        lw.ln1_beta  = layer_get("ln1_beta");
        lw.Wq        = layer_get("Wq");
        lw.Wk        = layer_get("Wk");
        lw.Wv        = layer_get("Wv");
        lw.Wo        = layer_get("Wo");
        lw.bq        = layer_get("bq");
        lw.bk        = layer_get("bk");
        lw.bv        = layer_get("bv");
        lw.bo        = layer_get("bo");
        lw.ln2_gamma = layer_get("ln2_gamma");
        lw.ln2_beta  = layer_get("ln2_beta");
        lw.W1        = layer_get("W1");
        lw.W1_gate   = layer_get("W1_gate");  // LLaMA SwiGLU gate stream
        lw.b1        = layer_get("b1");
        lw.W2        = layer_get("W2");
        lw.b2        = layer_get("b2");
    }
    return w;
}


PYBIND11_MODULE(_C, m) {
    m.doc() = "FlashGen CUDA inference engine — C++ / PyBind11 extension";

    // ── KVQuantType ──────────────────────────────────────────────────────────
    py::enum_<KVQuantType>(m, "KVQuantType")
        .value("NONE",     KVQuantType::NONE)
        .value("INT8",     KVQuantType::INT8)
        .value("FP8_E4M3", KVQuantType::FP8_E4M3)
        .export_values();

    // ── ModelConfig ──────────────────────────────────────────────────────────
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("name",          &ModelConfig::name)
        .def_readwrite("d_model",       &ModelConfig::d_model)
        .def_readwrite("n_heads",       &ModelConfig::n_heads)
        .def_readwrite("n_kv_heads",    &ModelConfig::n_kv_heads)
        .def_readwrite("n_layers",      &ModelConfig::n_layers)
        .def_readwrite("d_ff",          &ModelConfig::d_ff)
        .def_readwrite("vocab_size",    &ModelConfig::vocab_size)
        .def_readwrite("max_seq_len",   &ModelConfig::max_seq_len)
        .def_readwrite("layer_norm_eps",&ModelConfig::layer_norm_eps)
        .def_readwrite("use_fp16",      &ModelConfig::use_fp16)
        .def_readwrite("kv_quant",      &ModelConfig::kv_quant)
        .def("head_dim",         &ModelConfig::head_dim)
        .def("actual_kv_heads",  &ModelConfig::actual_kv_heads)
        .def("gqa_group_size",   &ModelConfig::gqa_group_size)
        // Presets
        .def_static("gpt2",        &ModelConfig::gpt2)
        .def_static("gpt2_medium", &ModelConfig::gpt2_medium)
        .def_static("gpt2_xl",     &ModelConfig::gpt2_xl)
        .def_static("llama2_7b",   &ModelConfig::llama2_7b)
        .def("__repr__", [](const ModelConfig& c) {
            return "<ModelConfig name=" + c.name +
                   " d=" + std::to_string(c.d_model) +
                   " L=" + std::to_string(c.n_layers) + ">";
        });

    // ── CacheConfig ──────────────────────────────────────────────────────────
    py::class_<CacheConfig>(m, "CacheConfig")
        .def(py::init<>())
        .def_readwrite("block_size",             &CacheConfig::block_size)
        .def_readwrite("num_gpu_blocks",         &CacheConfig::num_gpu_blocks)
        .def_readwrite("gpu_memory_utilization", &CacheConfig::gpu_memory_utilization)
        .def_readwrite("kv_quant",               &CacheConfig::kv_quant)
        .def_readwrite("enable_prefix_caching",  &CacheConfig::enable_prefix_caching);

    // ── SchedulerConfig ──────────────────────────────────────────────────────
    py::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<>())
        .def_readwrite("max_num_seqs",           &SchedulerConfig::max_num_seqs)
        .def_readwrite("max_num_tokens",         &SchedulerConfig::max_num_tokens)
        .def_readwrite("max_prefill_tokens",     &SchedulerConfig::max_prefill_tokens)
        .def_readwrite("enable_chunked_prefill", &SchedulerConfig::enable_chunked_prefill);

    // ── EngineConfig ─────────────────────────────────────────────────────────
    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("model",       &EngineConfig::model)
        .def_readwrite("cache",       &EngineConfig::cache)
        .def_readwrite("scheduler",   &EngineConfig::scheduler)
        .def_readwrite("weight_path", &EngineConfig::weight_path)
        .def_readwrite("device_id",   &EngineConfig::device_id);

    // ── SamplingParams ───────────────────────────────────────────────────────
    py::class_<SamplingParams>(m, "SamplingParams")
        .def(py::init<>())
        .def_readwrite("temperature",        &SamplingParams::temperature)
        .def_readwrite("top_p",              &SamplingParams::top_p)
        .def_readwrite("top_k",              &SamplingParams::top_k)
        .def_readwrite("repetition_penalty", &SamplingParams::repetition_penalty)
        .def_readwrite("greedy",             &SamplingParams::greedy)
        .def_readwrite("max_tokens",         &SamplingParams::max_tokens)
        .def_readwrite("stop_token_ids",     &SamplingParams::stop_token_ids);

    // ── FinishReason ─────────────────────────────────────────────────────────
    py::enum_<FinishReason>(m, "FinishReason")
        .value("NONE",        FinishReason::NONE)
        .value("EOS_TOKEN",   FinishReason::EOS_TOKEN)
        .value("MAX_TOKENS",  FinishReason::MAX_TOKENS)
        .value("STOP_STRING", FinishReason::STOP_STRING)
        .value("ABORT",       FinishReason::ABORT)
        .export_values();

    // ── InferenceResponse ────────────────────────────────────────────────────
    py::class_<InferenceResponse>(m, "InferenceResponse")
        .def(py::init<>())
        .def_readwrite("request_id",       &InferenceResponse::request_id)
        .def_readwrite("output_token_ids", &InferenceResponse::output_token_ids)
        .def_readwrite("generated_text",   &InferenceResponse::generated_text)
        .def_readwrite("finish_reason",    &InferenceResponse::finish_reason)
        .def_readwrite("latency_ms",       &InferenceResponse::latency_ms)
        .def_readwrite("tpot_ms",          &InferenceResponse::tpot_ms)
        .def_readwrite("prompt_tokens",    &InferenceResponse::prompt_tokens)
        .def_readwrite("generated_tokens", &InferenceResponse::generated_tokens)
        .def_readwrite("success",          &InferenceResponse::success)
        .def_readwrite("error",            &InferenceResponse::error);

    // ── InferenceRequest ─────────────────────────────────────────────────────
    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def(py::init<>())
        .def_readwrite("request_id",      &InferenceRequest::request_id)
        .def_readwrite("prompt_token_ids",&InferenceRequest::prompt_token_ids)
        .def_readwrite("sampling",        &InferenceRequest::sampling)
        .def_readwrite("priority",        &InferenceRequest::priority)
        .def("set_callback", [](InferenceRequest& req, py::object py_cb) {
            // Wrap Python callable as C++ std::function with GIL management.
            // The callback is invoked from a C++ background thread, so we must
            // acquire the GIL before calling into Python.
            req.callback = [py_cb = std::move(py_cb)](const InferenceResponse& resp) {
                py::gil_scoped_acquire gil;
                py_cb(resp);
            };
        });

    // ── EngineStats ──────────────────────────────────────────────────────────
    py::class_<EngineStats>(m, "EngineStats")
        .def_property_readonly("requests_completed",
            [](const EngineStats& s) { return s.requests_completed.load(); })
        .def_property_readonly("tokens_generated",
            [](const EngineStats& s) { return s.tokens_generated.load(); })
        .def_property_readonly("prefill_tokens",
            [](const EngineStats& s) { return s.prefill_tokens.load(); })
        .def_property_readonly("iterations",
            [](const EngineStats& s) { return s.iterations.load(); })
        .def("avg_latency_ms",  &EngineStats::avg_latency_ms)
        .def("throughput_tps",  &EngineStats::throughput_tps);

    // ── InferenceEngine ──────────────────────────────────────────────────────
    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<const EngineConfig&>(),
             py::arg("config"),
             "Construct the engine. Does NOT start the engine loop.")

        // Lifecycle
        .def("start",    &InferenceEngine::start,
             "Start the background engine loop thread.")
        .def("shutdown", &InferenceEngine::shutdown,
             "Gracefully stop the engine loop.")
        .def("is_running", &InferenceEngine::is_running)

        // Request submission
        .def("submit", [](InferenceEngine& eng, InferenceRequest req) {
            return eng.submit(std::move(req));
        }, py::arg("request"),
           "Async: submit request with callback. Returns request ID.")
        .def("generate", [](InferenceEngine& eng, InferenceRequest req) {
            py::gil_scoped_release release;   // let C++ run without holding GIL
            return eng.generate(std::move(req));
        }, py::arg("request"),
           "Sync: submit and block until response is ready.")

        // Stats
        .def("stats", &InferenceEngine::stats,
             py::return_value_policy::reference_internal)

        // Benchmarking
        .def("benchmark_prefill", &InferenceEngine::benchmark_prefill,
             py::arg("seq_len"), py::arg("batch_size"), py::arg("num_runs"),
             "Measure prefill throughput (tokens/sec).")
        .def("benchmark_decode",  &InferenceEngine::benchmark_decode,
             py::arg("num_seqs"), py::arg("context_len"), py::arg("decode_steps"),
             "Measure decode throughput (tokens/sec).")

        // Weight loading from Python tensors
        .def("load_weights_from_dict",
            [](InferenceEngine& /*eng*/, const py::dict& /*weight_dict*/) {
                // NOTE: This requires exposing model_ from InferenceEngine.
                // Placeholder — implement once model_ accessor is public.
                throw std::runtime_error(
                    "load_weights_from_dict not yet implemented. "
                    "Use EngineConfig.weight_path for binary weights, or "
                    "use the 'pytorch' backend for HuggingFace model loading."
                );
            }, py::arg("weight_dict"),
            "Load weights from Python dict of {name: tensor.data_ptr()} entries.");
}
