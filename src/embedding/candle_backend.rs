//! Candle-based inference backend for embedding and reranking models.
//!
//! Pure-Rust ML runtime using candle with Metal GPU acceleration on macOS.
//! Provides [`BertEmbedder`] for sentence embeddings (BGE-small/base/large)
//! and [`CrossEncoderReranker`] for relevance scoring (BGE-reranker-base).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::xlm_roberta::{
    Config as XLMRobertaConfig, XLMRobertaForSequenceClassification,
};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

/// Paths to downloaded model files from HuggingFace Hub.
pub struct ModelFiles {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weights_path: PathBuf,
}

/// Download model files from HuggingFace Hub.
///
/// Uses `hf_hub::api::sync::Api` which caches at `~/.cache/huggingface/hub/`.
/// Designed to be called from `spawn_blocking` since it performs synchronous I/O.
pub fn download_model(repo_id: &str, _cache_dir: Option<&Path>) -> Result<ModelFiles> {
    let api = hf_hub::api::sync::Api::new().context("Failed to initialize HuggingFace Hub API")?;
    let repo = api.model(repo_id.to_string());

    let config_path = repo
        .get("config.json")
        .context("Failed to download config.json")?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;
    let weights_path = repo
        .get("model.safetensors")
        .context("Failed to download model.safetensors")?;

    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        weights_path,
    })
}

/// Select the best available compute device.
///
/// Tries Metal (macOS) or CUDA (Linux/Windows) if the corresponding feature
/// is enabled. Probes layer-norm support since BERT/RoBERTa require it —
/// falls back to CPU if the GPU backend lacks the kernel.
pub fn select_device() -> Device {
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            if probe_layer_norm(&device) {
                tracing::info!("Using Metal GPU for inference");
                return device;
            }
            tracing::warn!("Metal GPU available but layer-norm not supported, falling back to CPU");
        }
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            if probe_layer_norm(&device) {
                tracing::info!("Using CUDA GPU for inference");
                return device;
            }
            tracing::warn!("CUDA GPU available but layer-norm not supported, falling back to CPU");
        }
    }
    tracing::info!("Using CPU for inference");
    Device::Cpu
}

/// Probe whether a device supports layer-norm (required by BERT/RoBERTa).
fn probe_layer_norm(device: &Device) -> bool {
    (|| -> candle_core::Result<()> {
        let weight = Tensor::ones(4, DType::F32, device)?;
        let bias = Tensor::zeros(4, DType::F32, device)?;
        let ln = LayerNorm::new(weight, bias, 1e-5);
        let input = Tensor::randn(0f32, 1.0, (1, 4), device)?;
        let _ = ln.forward(&input)?;
        Ok(())
    })()
    .is_ok()
}

/// BERT-based text embedder using candle.
///
/// Wraps a `BertModel` for generating sentence embeddings via mean pooling
/// and L2 normalization. Compatible with BGE-small/base/large-en-v1.5 models.
pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl BertEmbedder {
    /// Load a BERT embedding model from downloaded files.
    pub fn new(files: &ModelFiles, device: Device) -> Result<Self> {
        let config_str =
            std::fs::read_to_string(&files.config_path).context("Failed to read model config")?;
        let config: BertConfig =
            serde_json::from_str(&config_str).context("Failed to parse BERT config")?;

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        // SAFETY: mmap'd safetensors file — safe as long as the file is not modified
        // while the model is in use.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&files.weights_path], DType::F32, &device)
                .context("Failed to load model weights")?
        };
        let model = BertModel::load(vb, &config).context("Failed to construct BERT model")?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Generate embeddings for a batch of texts.
    ///
    /// Applies mean pooling over token hidden states (masked by attention mask)
    /// followed by L2 normalization. Returns one embedding vector per input text.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(str_refs, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let input_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_ids().to_vec())
            .collect();
        let attention_mask: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().to_vec())
            .collect();
        let token_type_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_type_ids().to_vec())
            .collect();

        let input_ids = Tensor::from_vec(input_ids, (batch_size, max_len), &self.device)?;
        let attention_mask_t =
            Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device)?;
        let token_type_ids = Tensor::from_vec(token_type_ids, (batch_size, max_len), &self.device)?;

        // Forward pass -> [batch, seq_len, hidden_size]
        let output = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask_t))?;

        // Mean pooling: mask padding tokens, sum, divide by token count
        let mask_f32 = attention_mask_t.to_dtype(DType::F32)?.unsqueeze(2)?;
        let masked = output.broadcast_mul(&mask_f32)?;
        let summed = masked.sum(1)?;
        let counts = mask_f32.sum(1)?;
        let pooled = summed.broadcast_div(&counts)?;

        // L2 normalize
        let norms = pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norms)?;

        normalized
            .to_vec2::<f32>()
            .context("Failed to convert embeddings to Vec")
    }
}

/// Cross-encoder reranker using XLM-RoBERTa.
///
/// Scores query-document pairs for relevance using a sequence classification
/// head. Compatible with BGE-reranker-base and similar cross-encoder models.
pub struct CrossEncoderReranker {
    model: XLMRobertaForSequenceClassification,
    tokenizer: Tokenizer,
    device: Device,
}

impl CrossEncoderReranker {
    /// Load a cross-encoder reranker from downloaded files.
    pub fn new(files: &ModelFiles, device: Device) -> Result<Self> {
        let config_str = std::fs::read_to_string(&files.config_path)
            .context("Failed to read reranker config")?;
        let config: XLMRobertaConfig =
            serde_json::from_str(&config_str).context("Failed to parse XLM-RoBERTa config")?;

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load reranker tokenizer: {}", e))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        // SAFETY: mmap'd safetensors file — safe as long as the file is not modified
        // while the model is in use.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&files.weights_path], DType::F32, &device)
                .context("Failed to load reranker weights")?
        };
        let model = XLMRobertaForSequenceClassification::new(1, &config, vb)
            .context("Failed to construct XLM-RoBERTa model")?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Score query-document pairs for relevance.
    ///
    /// Applies sigmoid to raw logits, returning a 0..1 relevance score per pair.
    pub fn score_pairs(&self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        let pair_refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(q, d)| (q.as_str(), d.as_str()))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(pair_refs, true)
            .map_err(|e| anyhow::anyhow!("Reranker tokenization failed: {}", e))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let input_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_ids().to_vec())
            .collect();
        let attention_mask: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().to_vec())
            .collect();

        let input_ids = Tensor::from_vec(input_ids, (batch_size, max_len), &self.device)?;
        let attention_mask = Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device)?;
        // XLM-RoBERTa doesn't use token_type_ids — pass zeros
        let token_type_ids = input_ids.zeros_like()?;

        // Forward pass -> [batch, 1] logits
        let logits = self
            .model
            .forward(&input_ids, &attention_mask, &token_type_ids)?;

        // Sigmoid to get 0..1 relevance scores, then flatten
        let scores = candle_nn::ops::sigmoid(&logits)?;
        let scores = scores.flatten_all()?.to_vec1::<f32>()?;

        Ok(scores)
    }
}
