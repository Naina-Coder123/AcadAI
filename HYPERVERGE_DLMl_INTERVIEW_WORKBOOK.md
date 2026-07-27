# HyperVerge DL/ML Research Intern — Interview Workbook

Interview: **Monday, 27 July 2026, 9:30 PM IST**  
Role: **DL/ML Research Intern (LLMs & VLMs)**

## How to use this workbook

Use the answers as **talking points**, not a memorized script. Be exact about AcadAI:

- You built a Python, Streamlit, FAISS, Sentence-Transformers and retrieval/evaluation system.
- The repository has TF-IDF and Sentence-Transformer benchmark paths, OCR-like noise experiments, FAISS/hybrid retrieval, reranking, critique, and grounding checks.
- Do **not** say you trained a PyTorch model, fine-tuned an LLM, built an OCR model, or deployed a VLM unless you personally did so outside this repository. Say how you would do it instead.

---

## 1. Opening, résumé, and AcadAI

**Interviewer:** Introduce yourself.

**Candidate:** I am interested in reliable ML systems for real-world document and multimodal problems. My main project, AcadAI, is an evaluation-driven RAG system for academic document understanding. It ingests academic PDFs, chunks and retrieves evidence using FAISS and hybrid retrieval, then produces grounded tutoring responses. I focused on measurable behavior: retrieval quality, OCR-like noise robustness, hallucination checks, and latency-quality trade-offs.

**Interviewer:** Explain AcadAI in one minute.

**Candidate:** A student uploads notes or PDFs and asks a question. AcadAI extracts text, creates document chunks with source metadata, retrieves relevant evidence through dense FAISS retrieval with lexical fallback and optional reranking, then routes the evidence to tutoring, critique, grounding, and memory components. The important point is that I evaluate retrieval separately with Precision@K, Recall@K, MRR, nDCG, latency, and OCR-noise experiments instead of treating a fluent demo answer as proof of quality.

**Interviewer:** Why is this relevant to HyperVerge?

**Candidate:** It maps to document AI: noisy documents, retrieval over extracted content, model evaluation, failure analysis, and accuracy-latency trade-offs. While AcadAI is primarily a RAG system rather than a trained vision model, it gave me a strong foundation in diagnosing document-pipeline failures and evaluating them systematically.

**Interviewer:** What was your specific contribution?

**Candidate:** I built and integrated the Python retrieval and evaluation workflow: PDF ingestion and chunking, FAISS-store loading, hybrid and reranked retrieval paths, grounding and critique checks, plus reproducible benchmark scripts and OCR-noise robustness tests. I would clearly distinguish my work from any library capability I used.

**Interviewer:** What is the hardest technical decision you made?

**Candidate:** Balancing semantic retrieval against exact terminology. Dense embeddings retrieve paraphrases but can confuse related academic subjects; lexical retrieval preserves exact terms but can miss paraphrases. I therefore treat hybrid retrieval, subject filtering, query expansion, and reranking as hypotheses to evaluate rather than universal fixes.

**Interviewer:** What would you improve next?

**Candidate:** I would collect a larger document-level held-out benchmark and compare TF-IDF with embedding models under clean and OCR-corrupted text. Then I would contrastively fine-tune a small sentence-transformer using academic query–positive–hard-negative triples, measuring quality, latency, and robustness before claiming improvement.

**Interviewer:** Your current benchmark is small and perfect. Why should we trust it?

**Candidate:** We should not overinterpret it. A perfect score on a small benchmark verifies the pipeline and baseline on those examples, not broad generalization. I would enlarge the dataset, use hard paraphrases and cross-subject distractors, split by document to prevent leakage, and report confidence intervals or repeated evaluations where feasible.

## 2. Python and software engineering

**Interviewer:** How do you organize an ML codebase?

**Candidate:** I separate ingestion, preprocessing, data definitions, model or retrieval logic, training/evaluation, configuration, and UI/deployment. Experiments must be runnable independently from the app so metrics can be reproduced. I log dataset version, seed, model name, parameters, and metrics with each run.

**Interviewer:** List versus generator—why does it matter?

**Candidate:** A list materializes all items in memory; a generator yields items lazily. For large image, PDF, or tokenized-text datasets, lazy iteration avoids RAM exhaustion, though it complicates random access and shuffling.

**Interviewer:** What are decorators useful for?

**Candidate:** They add behavior around functions without changing their core logic—for example caching expensive model loading, timing functions, retrying transient requests, logging arguments, or validating inputs. I would use them carefully because they can hide control flow during debugging.

**Interviewer:** What is a context manager?

**Candidate:** It guarantees cleanup around a resource, usually through `with`. Typical ML uses are opening files, database connections, locks, and inference contexts such as `torch.no_grad()`.

**Interviewer:** How would you speed up a slow Python data pipeline?

**Candidate:** Profile first. Then batch work, avoid repeated parsing or disk reads, cache embeddings and tokenization, use NumPy/PyTorch vectorized operations instead of Python loops, and use parallel data loading only after confirming it helps. I would monitor CPU, GPU, I/O, and memory separately.

**Interviewer:** How do you make experiments reproducible?

**Candidate:** Fix random seeds for Python, NumPy, and the framework; record package versions, hardware, dataset version, code commit, configuration, and preprocessing. For strict reproducibility, I would enable deterministic framework settings while documenting any speed trade-off.

**Interviewer:** How would you test an ML pipeline?

**Candidate:** Unit-test deterministic transformations and metric functions; integration-test a tiny end-to-end dataset; verify output shapes, dtypes, and ranges; and maintain a small regression set of known failures. Tests do not replace benchmark evaluation, but they catch broken pipelines early.

## 3. Mathematics and deep-learning fundamentals

**Interviewer:** What is gradient descent?

**Candidate:** It iteratively updates parameters in the negative-gradient direction to reduce a loss: `theta = theta - learning_rate * gradient(loss)`. Mini-batch SGD estimates that gradient from a batch, trading noisier updates for practical compute and often useful generalization.

**Interviewer:** Explain backpropagation.

**Candidate:** The forward pass produces predictions and loss. Backpropagation applies the chain rule from the loss backward through the computational graph to efficiently compute each parameter’s gradient. The optimizer then updates those parameters.

**Interviewer:** Why are nonlinear activations required?

**Candidate:** A composition of purely linear layers is still linear. Nonlinear activations allow the model to learn complex functions and decision boundaries. ReLU is simple and helps gradient flow; GELU is common in Transformers.

**Interviewer:** What are vanishing and exploding gradients?

**Candidate:** Repeated multiplication in backpropagation can make gradients shrink close to zero or grow very large. Remedies include good initialization, normalization, residual connections, suitable activations, gradient clipping, and careful learning-rate scheduling.

**Interviewer:** What is the bias–variance trade-off?

**Candidate:** High bias means the model is too constrained and underfits. High variance means it learns training-specific noise and overfits. I diagnose it with train/validation curves, then adjust capacity, data quality, regularization, augmentation, or training duration.

**Interviewer:** L1 versus L2 regularization?

**Candidate:** L1 penalizes absolute weights and can drive some weights to zero, encouraging sparsity. L2 penalizes squared weights and smoothly discourages large parameters. In deep learning, L2-like regularization is often implemented as weight decay; with adaptive optimizers, AdamW is preferred because it decouples decay from the gradient update.

**Interviewer:** Batch normalization versus layer normalization?

**Candidate:** Batch norm normalizes using batch statistics and is common in CNNs. Layer norm normalizes across features for each individual example and is independent of batch size, which makes it well suited to Transformers and variable sequence lengths.

**Interviewer:** What is dropout?

**Candidate:** During training it randomly masks activations, reducing co-adaptation and helping regularization. During evaluation it is disabled, so `model.train()` and `model.eval()` matter.

**Interviewer:** Cross-entropy loss versus MSE for classification?

**Candidate:** Cross-entropy compares predicted probability distributions to target labels and provides useful gradients for classification logits. MSE can be used but is usually a poorer fit for categorical likelihood modelling and often trains less effectively.

**Interviewer:** What is class imbalance and how do you handle it?

**Candidate:** A majority class can dominate accuracy, hiding poor minority-class performance. I use class-weighted loss, oversampling or targeted augmentation, suitable thresholds, and metrics such as precision, recall, F1, PR-AUC, per-class recall, and confusion matrices.

## 4. PyTorch questions

**Interviewer:** Describe a PyTorch training loop.

**Candidate:** Set `model.train()`, iterate over batches, move tensors to device, call `optimizer.zero_grad()`, forward pass, calculate loss, call `loss.backward()`, optionally clip gradients, then `optimizer.step()`. For validation, use `model.eval()` and `torch.no_grad()`.

**Interviewer:** Why call `zero_grad()`?

**Candidate:** PyTorch accumulates gradients by default. Without clearing them between optimizer updates, gradients from previous batches would be added unintentionally—except when deliberate gradient accumulation is being used.

**Interviewer:** `torch.no_grad()` versus `torch.inference_mode()`?

**Candidate:** Both disable gradient tracking. `inference_mode()` can reduce additional overhead and is suited to pure inference; `no_grad()` is more flexible where some tensors may later interact with autograd. In either case, I would use eval mode for dropout and batch norm.

**Interviewer:** What does a DataLoader do?

**Candidate:** It batches samples, can shuffle training data, and can use worker processes and pinned memory to improve transfer throughput. I tune worker count empirically because too many workers can hurt performance or exhaust resources.

**Interviewer:** Why use mixed precision?

**Candidate:** FP16 or BF16 reduces memory use and can speed compatible GPUs. FP16 may overflow or underflow, so AMP uses loss scaling. BF16 has a wider exponent range and is often more numerically robust when hardware supports it.

**Interviewer:** What is gradient accumulation?

**Candidate:** I run several micro-batches, accumulate gradients, then take one optimizer step. It approximates a larger effective batch size when GPU memory prevents loading that batch at once. The loss is typically scaled by the number of accumulation steps.

**Interviewer:** What causes CUDA out-of-memory, and what do you try?

**Candidate:** Large batch size, long sequences or high-resolution images, activation memory, optimizer state, fragmentation, and memory leaks. I reduce batch size or sequence length, use gradient accumulation/checkpointing, mixed precision, smaller models, proper cleanup, and inspect memory allocation rather than blindly clearing cache.

**Interviewer:** Why might a PyTorch model give different validation results every run?

**Candidate:** Random initialization, shuffled data, nondeterministic GPU kernels, dropout accidentally active, differing data splits, or unseeded augmentation. I inspect all of these before concluding that the model itself is unstable.

## 5. CNNs, vision, and document AI

**Interviewer:** Why are convolutions effective for images?

**Candidate:** They use local receptive fields and shared weights, exploiting spatial structure. Weight sharing makes them more parameter-efficient than a fully connected layer over pixels, and stacking layers builds progressively larger receptive fields.

**Interviewer:** What do stride, padding, and pooling do?

**Candidate:** Stride controls how far a kernel moves and can downsample. Padding preserves edge information and controls output size. Pooling summarizes local regions and adds some translation tolerance, though modern architectures often use strided convolutions or adaptive pooling.

**Interviewer:** What is data augmentation? Give document examples.

**Candidate:** Augmentation creates label-preserving variation to improve generalization. For documents: mild rotation, blur, compression artifacts, brightness or contrast changes, perspective distortion, and resolution variation. I would not apply transformations that invalidate labels, such as aggressive cropping of key text.

**Interviewer:** How would you diagnose a vision classifier with high training accuracy but weak validation accuracy?

**Candidate:** Check data leakage and split strategy first, especially duplicate images or templates across sets. Then inspect labels and train/validation distributions, use augmentations appropriate to production data, regularization, early stopping, and slice metrics by class and image quality.

**Interviewer:** Explain IoU, mAP, and NMS.

**Candidate:** IoU is overlap divided by union between predicted and ground-truth boxes. Average Precision summarizes the precision-recall curve for a class; mAP averages AP across classes and sometimes IoU thresholds. NMS removes duplicate overlapping predictions by keeping the highest-scoring box and suppressing similar ones.

**Interviewer:** OCR versus document understanding?

**Candidate:** OCR converts visual text into characters. Document understanding also requires layout, reading order, tables, key-value relationships, figures, and semantic meaning. OCR is an important component but not a complete document-AI solution.

**Interviewer:** Design a scanned-PDF pipeline.

**Candidate:** Detect whether pages are born-digital or scanned. For scans, preprocess with quality checks, deskewing, denoising and contrast normalization; run OCR; preserve page and bounding-box metadata; detect layout regions; make structure-aware chunks; then retrieve or extract with page-level citations. I would evaluate each stage and the end-to-end task.

**Interviewer:** How do OCR errors affect AcadAI?

**Candidate:** Character substitutions, broken spacing, and missing text can distort lexical matching and embeddings, so relevant chunks may fall in rank. AcadAI includes OCR-like corruption experiments to measure retrieval robustness; the next step is testing on real scanned documents and comparing cleanup, hybrid retrieval, and robust embeddings.

## 6. Transformers and modern neural networks

**Interviewer:** Explain scaled dot-product attention.

**Candidate:** Tokens become query, key, and value vectors. Attention computes query-key similarity, divides by `sqrt(d_k)` to avoid overly sharp softmax scores, normalizes with softmax, and uses the scores to mix value vectors: `softmax(QK^T / sqrt(d_k))V`.

**Interviewer:** Why multi-head attention?

**Candidate:** Multiple heads can represent different relationships simultaneously—local dependencies, long-range references, syntax, or domain patterns. Their outputs are concatenated and projected back to the model dimension.

**Interviewer:** Why positional encodings?

**Candidate:** Attention alone has no inherent order. Positional methods tell the model where tokens occur. Learned absolute embeddings, sinusoidal encodings, and RoPE are common choices; RoPE helps encode relative position and is widely used in modern decoder LLMs.

**Interviewer:** Encoder-only vs decoder-only vs encoder-decoder?

**Candidate:** Encoder-only models such as BERT use bidirectional context and suit embeddings/classification. Decoder-only models use causal masking to generate tokens. Encoder-decoder models encode an input then generate output and suit sequence-to-sequence tasks such as translation or summarization.

**Interviewer:** Why are residual connections important?

**Candidate:** They provide an identity path for information and gradients, making deep networks easier to optimize. They are central to ResNets and Transformer blocks.

**Interviewer:** What is causal masking?

**Candidate:** It blocks a decoder token from attending to future tokens, so next-token prediction does not leak the answer during training. At inference, the model generates autoregressively.

**Interviewer:** What is the attention complexity problem?

**Candidate:** Standard full attention scales quadratically with sequence length in memory and compute. Practical approaches include chunking plus retrieval, sparse/window attention, efficient attention variants, KV caching, and long-context architectures.

**Interviewer:** What is a token, and why does tokenization matter?

**Candidate:** A token is a unit processed by the model, often a subword. Tokenization affects vocabulary coverage, sequence length, multilingual behavior, cost, and how well code, numbers, or OCR-corrupted text are represented.

**Interviewer:** What is KV caching?

**Candidate:** During autoregressive generation, past keys and values are cached so the model does not recompute them for every new token. It reduces generation latency but increases memory use with context length and batch size.

## 7. Embeddings, RAG, and AcadAI follow-ups

**Interviewer:** What is an embedding?

**Candidate:** It is a dense vector representation where related inputs are positioned near each other in a learned vector space. In AcadAI, query and chunk embeddings enable semantic similarity search.

**Interviewer:** Cosine similarity versus dot product?

**Candidate:** Cosine similarity compares the angle between vectors, removing magnitude effects. Dot product incorporates both direction and magnitude. If vectors are L2-normalized, their dot product equals cosine similarity; index metric and preprocessing must be consistent.

**Interviewer:** Explain FAISS and index choice.

**Candidate:** FAISS provides efficient vector similarity search. For smaller datasets, exact flat search can be simple and accurate. For larger datasets, approximate methods such as IVF or HNSW trade a small amount of recall for speed and memory efficiency. I would benchmark recall and latency using realistic query distributions.

**Interviewer:** Why must the query embedding model match the FAISS index?

**Candidate:** The index vectors occupy the representation space and dimension of the embedding model used during indexing. A different model may have incompatible dimensions or a semantically incomparable space, producing errors or meaningless nearest neighbors.

**Interviewer:** Dense retrieval versus BM25/TF-IDF?

**Candidate:** Lexical retrieval is strong for exact entities, identifiers and rare technical terms. Dense retrieval captures paraphrases and semantic similarity but can confuse related topics. Hybrid retrieval is valuable when the task contains both kinds of signals.

**Interviewer:** What is reranking?

**Candidate:** First-stage retrieval returns a broad candidate set quickly. A more expensive cross-encoder then scores the query and each candidate jointly to reorder a smaller list. This can improve precision but adds latency.

**Interviewer:** How do chunk size and overlap affect RAG?

**Candidate:** Small chunks are precise but may lack context; large chunks preserve context but dilute relevance and consume context window. Overlap prevents information from being split at boundaries but creates redundancy. I would tune these experimentally by document type and task.

**Interviewer:** What is query expansion, and what can go wrong?

**Candidate:** Query expansion adds related terms or reformulates an underspecified query to improve recall. It can also inject incorrect assumptions or drift from user intent, so I use it conservatively and validate it on a benchmark.

**Interviewer:** How do you evaluate RAG end-to-end?

**Candidate:** Separate retrieval evaluation from generation evaluation. Retrieval needs evidence relevance and ranking metrics; generation needs correctness, completeness, groundedness, citation accuracy, safety, latency, and human review. A correct answer from model memory can mask bad retrieval, so the separation matters.

**Interviewer:** How would you reduce hallucination?

**Candidate:** Improve retrieval and chunk context, require evidence citations, constrain the answer to supplied evidence where appropriate, run claim-level support checks, and enable abstention or clarification when evidence is weak. I would measure groundedness, not merely rely on prompts.

## 8. LLM fine-tuning, PEFT, and alignment

**Interviewer:** When do you use RAG instead of fine-tuning?

**Candidate:** Use RAG when knowledge changes often, source attribution matters, or private documents must be consulted at inference. Fine-tuning is better for stable behavior such as output format, domain style, task execution, or structured extraction. They can also be combined.

**Interviewer:** Explain supervised fine-tuning.

**Candidate:** Starting from a pretrained model, train it on curated prompt-response examples using next-token loss. Quality, diversity, correct formatting, data de-duplication, and held-out evaluation are usually more important than simply increasing the number of examples.

**Interviewer:** What is LoRA?

**Candidate:** Low-Rank Adaptation freezes original weights and learns small low-rank matrices that approximate the update to selected layers. It reduces trainable parameters, checkpoint size, GPU memory, and training cost.

**Interviewer:** What is QLoRA?

**Candidate:** QLoRA stores the base model in quantized form, often 4-bit, while training LoRA adapters in higher precision. It makes fine-tuning larger open-weight models possible on more limited hardware, while requiring care around quantization and stability.

**Interviewer:** Which layers would you target with LoRA?

**Candidate:** Attention projections—often query and value—are common starting points, and feed-forward layers may also help. I would not assume a universal setting: I would run ablations over modules, rank, alpha, dropout, learning rate, and data size.

**Interviewer:** What are catastrophic forgetting and overfitting in fine-tuning?

**Candidate:** Excessive or narrow fine-tuning can degrade general capabilities or cause the model to memorize training patterns. I use high-quality diversified data, conservative hyperparameters, held-out general and task-specific evaluations, parameter-efficient methods, and early stopping.

**Interviewer:** What is instruction tuning?

**Candidate:** It fine-tunes models to follow diverse natural-language instructions and produce helpful responses. It improves interaction behavior, but does not guarantee fresh factual knowledge or factual grounding.

**Interviewer:** Explain RLHF, DPO, and reward hacking.

**Candidate:** RLHF uses preference comparisons to train a reward model and then optimizes the policy, commonly with PPO. DPO directly learns preferred behavior from preference pairs without an explicit RL loop. Reward hacking occurs when a model exploits imperfections in the reward signal—for example sounding confident or verbose rather than being correct—so preference data and evaluation must be carefully designed.

**Interviewer:** How would you build preference data for document QA?

**Candidate:** For the same question and evidence, create answer pairs where the preferred answer is accurate, complete, cited, and appropriately uncertain. Rejected answers should include plausible but unsupported claims, missing evidence, bad citations, and irrelevant content. Annotators need a precise rubric and adversarial examples.

## 9. Vision-language models and multimodality

**Interviewer:** What is a VLM?

**Candidate:** A Vision Language Model jointly uses visual and language information for tasks such as image question answering, captioning, visual grounding, chart or document understanding, and multimodal conversation.

**Interviewer:** How can image and text representations be aligned?

**Candidate:** A common approach uses separate image and text encoders trained contrastively: matching image-text pairs are pulled together in embedding space while nonmatching pairs are pushed apart. Other systems feed visual tokens through a projector into an LLM for generative reasoning.

**Interviewer:** What is CLIP-style contrastive learning?

**Candidate:** Given a batch of correct image-text pairs, it maximizes similarity for matching pairs and minimizes it for other pairs using a contrastive objective. This learns a shared embedding space useful for zero-shot classification and retrieval.

**Interviewer:** OCR-plus-LLM versus VLM for documents?

**Candidate:** OCR-plus-LLM is modular, interpretable, and often effective for text-heavy pages, but OCR and layout errors propagate. A VLM can directly use visual layout, charts, tables and non-textual cues, but may be more costly and harder to evaluate. The best choice depends on accuracy, latency, privacy, and document diversity.

**Interviewer:** How would you evaluate a document VLM?

**Candidate:** Use task-specific measures such as exact match/F1 for QA, ANLS for text extraction, key-value and table accuracy, grounding/citation quality, and human review of long answers. Slice results by scan quality, language, layout complexity, document type, handwriting, and question type.

**Interviewer:** What are common VLM failure modes?

**Candidate:** Hallucinating visual details, weak small-text recognition, spatial or counting errors, sensitivity to resolution and crop, poor handling of tables/charts, language bias, and incorrect confidence. I would build a failure taxonomy and evaluate by slices rather than only report one average number.

**Interviewer:** How would you improve VLM performance on low-quality documents?

**Candidate:** Start with data: representative low-quality scans, annotations and augmentations. Test image preprocessing, higher-resolution/tiled inference, OCR/layout-aware hybrid pipelines, domain adaptation or parameter-efficient fine-tuning, and confidence-aware routing. Every change should be compared on a fixed held-out robustness set.

## 10. Debugging, evaluation, and ML system design

**Interviewer:** Training loss is not decreasing. What is your debugging order?

**Candidate:** First overfit a tiny clean subset. If that fails, inspect labels, input scaling, tensor shapes and dtypes, model-output/loss compatibility, learning rate, gradient norms, parameter updates, and train mode. Then validate the data loader and augmentations before changing architecture.

**Interviewer:** Loss becomes NaN. What do you inspect?

**Candidate:** Input and labels for NaN/Inf; unstable `log`, division, exponentials or softmax; excessive learning rate and exploding gradients; mixed-precision overflow; and invalid operations in custom loss. I isolate the first failing batch/layer, inspect activation and gradient ranges, then apply a targeted fix such as stable operations, clipping, lower LR, or AMP scaling.

**Interviewer:** What is data leakage?

**Candidate:** Information from validation/test data leaks into training or model-selection decisions, inflating scores. Examples include duplicate documents across splits, fitting normalization on all data, chunks from one source in train and test, or tuning repeatedly on the test set. Split by the real unit of generalization.

**Interviewer:** Accuracy is high but the product fails. Why?

**Candidate:** Aggregate accuracy can hide class imbalance, distribution shift, costly error types, calibration problems, threshold choices, or critical slices such as blurry documents. I use task-appropriate metrics, confusion matrices, calibration checks, slice analysis, and error cost.

**Interviewer:** Precision versus recall—when would you favor each?

**Candidate:** Favor precision when false positives are costly, such as automatically approving a document. Favor recall when missing a true case is costly, such as fraud screening or safety review. Threshold selection should follow the business/error cost, not a generic preference.

**Interviewer:** What is calibration?

**Candidate:** A calibrated model’s predicted confidence matches empirical correctness: predictions at 80% confidence should be correct roughly 80% of the time. Calibration supports safe thresholds and human-review routing. Methods include temperature scaling, evaluated with reliability diagrams and expected calibration error.

**Interviewer:** How do you choose a baseline?

**Candidate:** Start with the simplest meaningful baseline that can be reproduced quickly—rule-based, TF-IDF, linear model, or pretrained model without fine-tuning. A baseline gives a quality/latency reference and prevents claiming progress without evidence.

**Interviewer:** Design an experiment to compare two retrieval models.

**Candidate:** Fix the dataset, document-level split, queries, relevance labels, preprocessing, hardware, candidate count, and evaluation metrics. Run both models under the same conditions and report P@K, Recall@K, MRR, nDCG, latency percentiles, memory, errors, and slice-level results. Inspect disagreements qualitatively and avoid tuning on the final test set.

**Interviewer:** What would you log in production?

**Candidate:** Privacy-safe request metadata, model/index/version, latency by component, errors, resource use, confidence or retrieval scores, user feedback, and sampled human-reviewed outcomes. I would avoid logging raw sensitive documents unnecessarily and define retention/access controls.

## 11. Behavioural and closing questions

**Interviewer:** Tell me about a failure.

**Candidate:** In AcadAI, short underspecified questions can retrieve generic or wrong-subject chunks. Rather than hide it, I framed it as a failure hypothesis: compare dense, lexical and hybrid retrieval; inspect top-k subjects; test query expansion and metadata filters; then define a top-k acceptance criterion. That taught me to make failures measurable.

**Interviewer:** What do you do when you do not know an answer?

**Candidate:** I state what I know, identify the assumption or missing fact, and describe how I would verify it—documentation, a minimal experiment, logs, or a paper. I avoid confidently inventing an answer; that is particularly important in ML because many choices are empirical.

**Interviewer:** Why should we hire you if you have not yet trained a large VLM?

**Candidate:** I bring strong Python and evaluation-oriented thinking, plus a working document-AI project where I have dealt with retrieval, noisy text, grounding and failure analysis. I understand the core DL/Transformer concepts and am ready to deepen hands-on PyTorch, fine-tuning and multimodal work. I learn through baselines, controlled experiments and careful debugging rather than treating models as black-box APIs.

**Interviewer:** What will you learn in the first month?

**Candidate:** I would first understand the task, data, labels, current baseline, evaluation protocol, failure slices, and production constraints. Then I would reproduce a baseline, identify one measurable improvement, and document results and trade-offs. That lets me contribute quickly without making unsupported architectural changes.

**Interviewer:** Do you have questions for us?

**Candidate:** Yes. How does the team evaluate document and multimodal models beyond aggregate accuracy—do you track slices such as image quality, language, layout, and document type? Also, for interns, how much of the work is model training/fine-tuning versus building evaluation and failure-analysis infrastructure?

---

## Rapid-fire facts to know

| Topic | Short answer |
|---|---|
| ReLU | `max(0, x)`; simple, but can create dead neurons. |
| GELU | Smooth activation widely used in Transformers. |
| AdamW | Adam with decoupled weight decay. |
| F1 | Harmonic mean of precision and recall. |
| ROC-AUC caveat | Can look overly optimistic on severe class imbalance; PR-AUC may be more informative. |
| MRR | Mean of reciprocal rank of first relevant result. |
| nDCG | Ranking metric that discounts lower positions and supports graded relevance. |
| LoRA | Train low-rank adapters while freezing base weights. |
| RAG | Retrieve external evidence at inference, then generate using it. |
| Embedding | Dense vector for semantic comparison. |
| Cross-encoder | Jointly scores query-document pair; accurate but slower. |
| VLM | Model connecting image and language representations. |
| Overfit tiny batch | Essential fast test for pipeline/model correctness. |

## Questions to avoid answering dishonestly

- “Which PyTorch training run did you personally complete?” — describe only actual work; otherwise explain the intended training loop and say you are extending hands-on experience.
- “What LoRA rank did you use?” — do not invent one. Say you have studied the method and would tune rank empirically.
- “Which OCR engine/VLM did AcadAI deploy?” — AcadAI is OCR-ready and contains synthetic OCR-noise robustness evaluation; it does not by itself prove a deployed OCR or VLM pipeline.
- “What benchmark improvement did your embedding fine-tuning achieve?” — do not claim one unless you ran and recorded it.

## Last-minute checklist

1. Practice the one-minute AcadAI pitch and the honest limitation statement.
2. Be able to derive self-attention conceptually and explain a PyTorch training loop.
3. Know the difference between retrieval metrics and generation metrics.
4. Prepare one debugging story, one failure story, and one question for the interviewer.
5. If asked a deep implementation detail you have not used: reason from first principles, state uncertainty, and describe the experiment you would run.
