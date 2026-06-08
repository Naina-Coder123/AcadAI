# AcadAI System Showcase

## Overview

AcadAI is a Research-Backed Multi-Agent Educational Intelligence Platform that combines:

- Retrieval-Augmented Generation (RAG)
- Multi-Agent Reasoning
- Personalized Tutoring
- Hallucination Detection
- Conversation Memory
- Learning Roadmaps
- Viva Preparation
- Revision Generation
- Retrieval Evaluation

The system is designed to provide grounded, explainable, and personalized educational assistance using student-uploaded academic resources.

---

# System Architecture

Student Query
      │
      ▼
Router Agent
      │
      ▼
FAISS Retrieval Engine
      │
      ▼
Reasoning Agent
      │
      ▼
Tutor Agent
      │
      ▼
Critic Agent
      │
      ▼
Grounding Engine
      │
      ▼
Final Answer

---

# Screenshot 1: Ask Module

Purpose:
Generate grounded educational answers using uploaded notes.

Features Demonstrated:

- Retrieval-Augmented Generation
- Source Grounding
- Critic Evaluation
- Hallucination Detection
- Evidence Tracking

Research Support:

- Lewis et al. (2020) RAG
- Maynez et al. (2020) Hallucination Detection

Observed Metrics:

- Grounding Score: 93.3%
- Accuracy Score: 10/10
- Relevance Score: 9/10
- Overall Quality: 8/10

Impact:

Provides trustworthy educational answers supported by retrieved evidence.

---

# Screenshot 2: Retrieval Evaluation Dashboard

Purpose:

Evaluate retrieval quality before generation.

Features Demonstrated:

- Hit Rate Analysis
- Subject Classification
- Retrieval Accuracy
- Source Traceability
- Retrieval Benchmarking

Observed Results:

Hit Rate: 100%

Queries Tested: 12

Subjects:

- DBMS
- OS
- DSA
- CN
- ML
- SE

Metrics:

Precision@1 = 1.00

Recall@4 = 1.00

MRR = 1.00

nDCG@4 = 0.9277

F1@4 = 0.7937

Impact:

Demonstrates that relevant educational content is consistently retrieved before answer generation.

---

# Screenshot 3: Revision Suite

Purpose:

Generate exam-oriented revision material.

Features:

- Night-before Exam Notes
- Flashcards
- Likely Exam Questions
- Key Definitions
- Quick Revision Content

Research Support:

- Intelligent Tutoring Systems
- AutoTutor
- Formative Feedback Models

Impact:

Reduces revision time while improving knowledge retention.

---

# Screenshot 4: Personalized Learning Roadmap

Purpose:

Generate adaptive learning plans.

Inputs:

- Student Name
- Branch
- Semester
- Learning Goal
- Preferred Explanation Level

Output:

Day-wise Study Roadmap

Example:

Day 1:
Study DBMS topic
Create notes
Solve questions

Day 2:
Practice normalization

Day 3:
Practice SQL

...

Research Support:

- Knowledge Tracing
- Personalized Learning Systems

Impact:

Transforms static learning into adaptive learning.

---

# Screenshot 5: Conversation Memory Agent

Purpose:

Maintain educational context across interactions.

Stored Information:

- Previous Questions
- Grounding Scores
- Learning Topics
- Weak Areas

Research Support:

Park et al. (2023)

Generative Agents

Impact:

Allows contextual follow-up conversations.

Example:

Student:
"What is deadlock?"

Follow-up:
"Explain prevention techniques."

The system understands previous context.

---

# Screenshot 6: Critic Evaluation Framework

Purpose:

Evaluate generated responses.

Dimensions:

1. Relevance
2. Accuracy
3. Completeness
4. Clarity

Generated Metrics:

- Overall Quality
- Response Time
- Evidence Used
- Review Requirement

Impact:

Provides explainability rarely available in educational chatbots.

---

# Screenshot 7: Grounding Engine

Purpose:

Detect hallucinations.

Grounding Metrics:

- Supported Statements
- Evidence Coverage
- Citation Coverage
- Grounding Score

Example:

Grounding Score = 93.3%

Supported Statements = 14 / 15

Impact:

Improves trustworthiness and factual reliability.

---

# Core Technologies

Frontend

- Streamlit

Backend

- Python

Retrieval

- FAISS
- BGE-Large Embeddings
- Hybrid Retrieval

Generation

- Mistral LLM

Reranking

- Cross Encoder

Architecture

- Multi-Agent Pipeline

Corpus Size

- 12,263+ Educational Chunks

---

# Research Contributions

AcadAI integrates ideas from:

[1] Retrieval-Augmented Generation (Lewis et al., 2020)

[2] Hallucination Detection (Maynez et al., 2020)

[3] Multi-Agent Systems (Xi et al., 2023)

[4] MetaGPT (Hong et al., 2024)

[5] Cognitive Self-Collaboration (Wang et al., 2024)

[6] Intelligent Tutoring Systems (Graesser et al., 2017)

[7] Generative Agents (Park et al., 2023)

[8] Dense Passage Retrieval (Karpukhin et al., 2020)

---

# Real-World Applications

Students

- Personalized Learning
- Exam Preparation
- Revision Support
- Viva Training

Faculty

- Academic Assistance
- Question Generation
- Course Support

Universities

- Institutional AI Tutor
- Digital Learning Assistant

EdTech Platforms

- Course-Specific AI Mentor

---

# Project Impact

Retrieval Recall: 100%

First Relevant Hit: 100%

MRR: 1.00

nDCG: 0.9277

Corpus Size: 12,263+ Chunks

Architecture: Multi-Agent RAG

Grounding: Evidence-Based

Personalization: Memory Driven

Educational Design: Tutor Agent

Hallucination Resistance: Grounding Engine

Scalability: FAISS Powered
