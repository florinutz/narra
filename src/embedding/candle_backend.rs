//! Candle-based inference backend for embedding, reranking, and classification models.
//!
//! Pure-Rust ML runtime using candle with Metal GPU acceleration on macOS.
//! Provides [`BertEmbedder`] for sentence embeddings (BGE-small/base/large),
//! [`CrossEncoderReranker`] for relevance scoring (BGE-reranker-base),
//! and [`SequenceClassifier`] for multi-label classification (GoEmotions).

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
    #[cfg(target_os = "macos")]
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

/// Natural Language Inference classifier using RoBERTa/XLM-RoBERTa.
///
/// Classifies premise-hypothesis pairs into entailment/neutral/contradiction.
/// Used for zero-shot theme classification: given a text and a candidate theme,
/// the entailment probability indicates how strongly the text matches the theme.
pub struct NliClassifier {
    model: XLMRobertaForSequenceClassification,
    tokenizer: Tokenizer,
    device: Device,
    entailment_idx: usize,
}

impl NliClassifier {
    /// Load an NLI classifier from downloaded model files.
    ///
    /// Parses `id2label` from config.json to determine the entailment label index.
    pub fn new(files: &ModelFiles, device: Device) -> Result<Self> {
        let config_str =
            std::fs::read_to_string(&files.config_path).context("Failed to read NLI config")?;
        let config: XLMRobertaConfig =
            serde_json::from_str(&config_str).context("Failed to parse NLI config")?;

        // Parse id2label to find the entailment index
        let config_json: serde_json::Value =
            serde_json::from_str(&config_str).context("Failed to parse config as JSON")?;
        let id2label = config_json
            .get("id2label")
            .and_then(|v| v.as_object())
            .context("config.json missing id2label mapping")?;

        let entailment_idx = id2label
            .iter()
            .find_map(|(k, v)| {
                let label = v.as_str()?;
                if label.eq_ignore_ascii_case("entailment") {
                    k.parse::<usize>().ok()
                } else {
                    None
                }
            })
            .context("id2label does not contain an 'entailment' label")?;

        let num_labels = id2label.len();

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load NLI tokenizer: {}", e))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        // SAFETY: mmap'd safetensors file — safe as long as the file is not modified
        // while the model is in use.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&files.weights_path], DType::F32, &device)
                .context("Failed to load NLI weights")?
        };
        let model = XLMRobertaForSequenceClassification::new(num_labels, &config, vb)
            .context("Failed to construct NLI model")?;

        Ok(Self {
            model,
            tokenizer,
            device,
            entailment_idx,
        })
    }

    /// Classify premise-hypothesis pairs.
    ///
    /// Returns the entailment probability (0..1) for each pair.
    /// Batches all pairs in a single forward pass for efficiency.
    pub fn classify_pairs(&self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        let pair_refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(pair_refs, true)
            .map_err(|e| anyhow::anyhow!("NLI tokenization failed: {}", e))?;

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
        // RoBERTa/XLM-RoBERTa doesn't use token_type_ids — pass zeros
        let token_type_ids = input_ids.zeros_like()?;

        // Forward pass -> [batch, num_labels] logits
        let logits = self
            .model
            .forward(&input_ids, &attention_mask, &token_type_ids)?;

        // Softmax over labels (NLI is single-label, not multi-label)
        let probs = candle_nn::ops::softmax(&logits, 1)?;
        let probs_vec = probs.to_vec2::<f32>()?;

        // Extract entailment probability for each pair
        let entailment_scores: Vec<f32> = probs_vec
            .into_iter()
            .map(|row| row.get(self.entailment_idx).copied().unwrap_or(0.0))
            .collect();

        Ok(entailment_scores)
    }
}

/// Token-level classifier using BERT for Named Entity Recognition.
///
/// Wraps a `BertModel` with a custom linear classification head for per-token
/// BIO tagging. Compatible with BERT-based NER models (e.g., dslim/bert-base-NER).
pub struct TokenClassifier {
    model: BertModel,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    tokenizer: Tokenizer,
    device: Device,
    num_labels: usize,
    labels: Vec<String>,
}

/// A recognized entity extracted by the token classifier.
#[derive(Debug, Clone)]
pub struct RecognizedEntity {
    /// Entity text as it appears in the input
    pub text: String,
    /// Entity type (PER, LOC, ORG, MISC)
    pub label: String,
    /// Average confidence score across tokens
    pub score: f32,
    /// Character start offset in input text
    pub start: usize,
    /// Character end offset in input text
    pub end: usize,
}

impl TokenClassifier {
    /// Load a token classifier from downloaded model files.
    ///
    /// Loads BERT model weights under the `bert.*` prefix and a separate
    /// `classifier.*` linear head for per-token label prediction.
    pub fn new(files: &ModelFiles, device: Device) -> Result<Self> {
        let config_str = std::fs::read_to_string(&files.config_path)
            .context("Failed to read token classifier config")?;
        let config: BertConfig =
            serde_json::from_str(&config_str).context("Failed to parse BERT config")?;

        // Parse id2label from config.json
        let config_json: serde_json::Value =
            serde_json::from_str(&config_str).context("Failed to parse config as JSON")?;
        let id2label = config_json
            .get("id2label")
            .and_then(|v| v.as_object())
            .context("config.json missing id2label mapping")?;

        let mut label_entries: Vec<(usize, String)> = id2label
            .iter()
            .filter_map(|(k, v)| {
                let idx: usize = k.parse().ok()?;
                let label = v.as_str()?.to_string();
                Some((idx, label))
            })
            .collect();
        label_entries.sort_by_key(|(idx, _)| *idx);
        let labels: Vec<String> = label_entries.into_iter().map(|(_, label)| label).collect();
        let num_labels = labels.len();

        if num_labels == 0 {
            anyhow::bail!("id2label is empty — cannot determine label count");
        }

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load NER tokenizer: {}", e))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        // SAFETY: mmap'd safetensors file — safe as long as the file is not modified.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&files.weights_path], DType::F32, &device)
                .context("Failed to load NER weights")?
        };

        // Load classifier head weights before BertModel consumes vb
        let classifier_weight = vb
            .pp("classifier")
            .get((num_labels, config.hidden_size), "weight")
            .context("Failed to load classifier.weight")?;
        let classifier_bias = vb
            .pp("classifier")
            .get(num_labels, "bias")
            .context("Failed to load classifier.bias")?;

        // NER models use `bert.*` prefix for the base model weights
        let model = BertModel::load(vb.pp("bert"), &config)
            .context("Failed to construct BERT model for NER")?;

        Ok(Self {
            model,
            classifier_weight,
            classifier_bias,
            tokenizer,
            device,
            num_labels,
            labels,
        })
    }

    /// Extract named entities from texts.
    ///
    /// Runs per-token classification, merges BIO tags into entity spans,
    /// and maps subword offsets back to original character positions.
    pub fn extract_entities(&self, texts: &[String]) -> Result<Vec<Vec<RecognizedEntity>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(str_refs, true)
            .map_err(|e| anyhow::anyhow!("NER tokenization failed: {}", e))?;

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

        // Forward pass through BERT -> [batch, seq_len, hidden_size]
        let hidden_states =
            self.model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask_t))?;

        // Apply classification head: hidden @ W^T + b -> [batch, seq_len, num_labels]
        let logits = hidden_states
            .matmul(&self.classifier_weight.t()?)?
            .broadcast_add(&self.classifier_bias)?;

        // Softmax per token -> probabilities
        let probs = candle_nn::ops::softmax(&logits, 2)?;
        let probs_3d = probs.to_vec3::<f32>()?;

        // Post-process: merge BIO tags into entity spans per text
        let mut all_entities = Vec::with_capacity(batch_size);

        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let offsets = encoding.get_offsets();
            let special_tokens = encoding.get_special_tokens_mask();
            let token_probs = &probs_3d[batch_idx];

            let entities =
                self.merge_bio_tags(token_probs, offsets, special_tokens, &texts[batch_idx]);
            all_entities.push(entities);
        }

        Ok(all_entities)
    }

    /// Merge BIO-tagged tokens into entity spans.
    fn merge_bio_tags(
        &self,
        token_probs: &[Vec<f32>],
        offsets: &[(usize, usize)],
        special_tokens: &[u32],
        original_text: &str,
    ) -> Vec<RecognizedEntity> {
        let mut entities: Vec<RecognizedEntity> = Vec::new();
        let mut current_entity: Option<(String, usize, usize, Vec<f32>)> = None; // (label, start, end, scores)

        for (tok_idx, probs) in token_probs.iter().enumerate() {
            // Skip special tokens ([CLS], [SEP], [PAD])
            if tok_idx >= special_tokens.len() || special_tokens[tok_idx] == 1 {
                // Flush any pending entity
                if let Some((label, start, end, scores)) = current_entity.take() {
                    let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
                    let text = original_text[start..end].to_string();
                    if !text.trim().is_empty() {
                        entities.push(RecognizedEntity {
                            text: text.trim().to_string(),
                            label,
                            score: avg_score,
                            start,
                            end,
                        });
                    }
                }
                continue;
            }

            // Skip padding tokens (offsets are (0,0))
            let (tok_start, tok_end) = offsets[tok_idx];
            if tok_start == 0 && tok_end == 0 && tok_idx > 0 {
                continue;
            }

            // Find best label for this token
            let (best_idx, best_prob) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            let tag = &self.labels[best_idx];

            if let Some(entity_type) = tag.strip_prefix("B-") {
                // Flush previous entity
                if let Some((label, start, end, scores)) = current_entity.take() {
                    let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
                    let text = original_text[start..end].to_string();
                    if !text.trim().is_empty() {
                        entities.push(RecognizedEntity {
                            text: text.trim().to_string(),
                            label,
                            score: avg_score,
                            start,
                            end,
                        });
                    }
                }
                // Start new entity
                current_entity = Some((
                    entity_type.to_string(),
                    tok_start,
                    tok_end,
                    vec![*best_prob],
                ));
            } else if let Some(entity_type) = tag.strip_prefix("I-") {
                // Continue existing entity of matching type
                if let Some((ref label, _, ref mut end, ref mut scores)) = current_entity {
                    if label == entity_type {
                        *end = tok_end;
                        scores.push(*best_prob);
                        continue;
                    }
                }
                // Orphan I-tag (no matching B-) — treat as B-
                if let Some((label, start, end, scores)) = current_entity.take() {
                    let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
                    let text = original_text[start..end].to_string();
                    if !text.trim().is_empty() {
                        entities.push(RecognizedEntity {
                            text: text.trim().to_string(),
                            label,
                            score: avg_score,
                            start,
                            end,
                        });
                    }
                }
                current_entity = Some((
                    entity_type.to_string(),
                    tok_start,
                    tok_end,
                    vec![*best_prob],
                ));
            } else {
                // O tag — flush any pending entity
                if let Some((label, start, end, scores)) = current_entity.take() {
                    let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
                    let text = original_text[start..end].to_string();
                    if !text.trim().is_empty() {
                        entities.push(RecognizedEntity {
                            text: text.trim().to_string(),
                            label,
                            score: avg_score,
                            start,
                            end,
                        });
                    }
                }
            }
        }

        // Flush final entity
        if let Some((label, start, end, scores)) = current_entity {
            let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
            let text = original_text[start..end].to_string();
            if !text.trim().is_empty() {
                entities.push(RecognizedEntity {
                    text: text.trim().to_string(),
                    label,
                    score: avg_score,
                    start,
                    end,
                });
            }
        }

        entities
    }

    /// Get the label names in index order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Get the number of classification labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
}

/// Multi-label sequence classifier using XLM-RoBERTa.
///
/// Classifies text into multiple labels with independent sigmoid activations.
/// Compatible with RoBERTa-based GoEmotions models (28 emotion labels).
pub struct SequenceClassifier {
    model: XLMRobertaForSequenceClassification,
    tokenizer: Tokenizer,
    device: Device,
    num_labels: usize,
    labels: Vec<String>,
}

impl SequenceClassifier {
    /// Load a sequence classifier from downloaded model files.
    ///
    /// Parses `id2label` from config.json to determine label names and count.
    pub fn new(files: &ModelFiles, device: Device) -> Result<Self> {
        let config_str = std::fs::read_to_string(&files.config_path)
            .context("Failed to read classifier config")?;
        let config: XLMRobertaConfig =
            serde_json::from_str(&config_str).context("Failed to parse XLM-RoBERTa config")?;

        // Parse id2label from config.json for label names
        let config_json: serde_json::Value =
            serde_json::from_str(&config_str).context("Failed to parse config as JSON")?;
        let id2label = config_json
            .get("id2label")
            .and_then(|v| v.as_object())
            .context("config.json missing id2label mapping")?;

        // Build ordered label list from id2label: {"0": "admiration", "1": "amusement", ...}
        let mut label_entries: Vec<(usize, String)> = id2label
            .iter()
            .filter_map(|(k, v)| {
                let idx: usize = k.parse().ok()?;
                let label = v.as_str()?.to_string();
                Some((idx, label))
            })
            .collect();
        label_entries.sort_by_key(|(idx, _)| *idx);
        let labels: Vec<String> = label_entries.into_iter().map(|(_, label)| label).collect();
        let num_labels = labels.len();

        if num_labels == 0 {
            anyhow::bail!("id2label is empty — cannot determine label count");
        }

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load classifier tokenizer: {}", e))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        // SAFETY: mmap'd safetensors file — safe as long as the file is not modified
        // while the model is in use.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&files.weights_path], DType::F32, &device)
                .context("Failed to load classifier weights")?
        };
        let model = XLMRobertaForSequenceClassification::new(num_labels, &config, vb)
            .context("Failed to construct classifier model")?;

        Ok(Self {
            model,
            tokenizer,
            device,
            num_labels,
            labels,
        })
    }

    /// Classify texts into multi-label scores.
    ///
    /// Returns one `Vec<(label, score)>` per input text, with all labels
    /// and their sigmoid-activated scores (0..1).
    pub fn classify(&self, texts: &[String]) -> Result<Vec<Vec<(String, f32)>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(str_refs, true)
            .map_err(|e| anyhow::anyhow!("Classifier tokenization failed: {}", e))?;

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

        // Forward pass -> [batch, num_labels] logits
        let logits = self
            .model
            .forward(&input_ids, &attention_mask, &token_type_ids)?;

        // Sigmoid for independent multi-label activations
        let scores = candle_nn::ops::sigmoid(&logits)?;
        let scores_vec = scores.to_vec2::<f32>()?;

        // Pair each score with its label name
        let results: Vec<Vec<(String, f32)>> = scores_vec
            .into_iter()
            .map(|row| {
                self.labels
                    .iter()
                    .zip(row)
                    .map(|(label, score)| (label.clone(), score))
                    .collect()
            })
            .collect();

        Ok(results)
    }

    /// Get the label names in index order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Get the number of classification labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
}
