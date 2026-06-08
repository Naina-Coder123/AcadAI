# 🎓 AcadAI  
## Multi-Agent System for Personalized Academic Learning & Assistance

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Mistral](https://img.shields.io/badge/Mistral-LLM-orange?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-purple?style=for-the-badge)
![Multi-Agent](https://img.shields.io/badge/Multi--Agent-AI%20Workflow-black?style=for-the-badge)

---

## 📌 Project Summary

**AcadAI** is a multi-agent AI learning platform that helps students learn from their own academic material. It combines **Large Language Models**, **Retrieval-Augmented Generation**, **FAISS vector search**, **conversation memory**, **viva/quiz evaluation**, and **hallucination detection** to produce syllabus-aligned, evidence-grounded, exam-oriented academic answers.

Unlike a generic chatbot, AcadAI retrieves relevant content from uploaded notes, evaluates answer quality through a critic loop, verifies grounding against evidence, and adapts learning support using student memory and weak-topic tracking.

---

## 🚀 Why AcadAI?

Most AI tools are powerful but not optimized for college learning.

They often:

- Generate generic answers
- Ignore syllabus and lecture notes
- Hallucinate unsupported facts
- Lack exam-oriented structure
- Do not track student progress
- Do not provide viva-style evaluation

AcadAI solves these issues by using a **multi-agent academic workflow** where every answer is retrieved, reasoned, taught, reviewed, verified, and stored for personalization.

---

## 🎯 Objectives

- Build a personalized AI academic assistant for B.Tech students
- Ground answers using syllabus, notes, PDFs, and scanned documents
- Implement a multi-agent pipeline with Router, Reasoning, Tutor, Critic, and Memory agents
- Reduce hallucination using evidence verification and grounding score
- Support viva preparation, quiz evaluation, flashcards, revision notes, and roadmaps
- Provide a professional AI-LMS experience using Streamlit

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📚 Notes-Based Q&A | Answers questions using uploaded academic notes |
| 🔍 RAG Retrieval | Retrieves relevant chunks using FAISS vector search |
| 🤖 Multi-Agent Pipeline | Router, Reasoning, Tutor, Critic, Grounding, and Memory agents |
| 🧠 Conversation Memory | Supports follow-up questions using previous context |
| 🛡 Hallucination Detection | Checks whether answer claims are supported by evidence |
| 🎤 Viva / Quiz Mode | Generates viva questions and evaluates student answers |
| 📈 Weak Topic Tracker | Tracks weak topics based on quiz and answer quality |
| 📝 Revision Suite | Generates revision notes, flashcards, likely exam questions, and roadmaps |
| 📊 Evaluation Dashboard | Shows relevance, completeness, accuracy, clarity, and grounding |
| ⚡ Lightweight Retrieval | Uses local FAISS + hybrid reranking for practical laptop use |

---

# 🏗 System Architecture

## 1. Complete Block-Level Architecture

```mermaid
flowchart TD

    subgraph UI["🖥️ User Interface Layer"]
        A1["Student Dashboard"]
        A2["PDF Upload"]
        A3["Ask with Memory"]
        A4["Viva Studio"]
        A5["Revision Suite"]
        A6["Evaluation Console"]
    end

    subgraph Input["📥 Input Processing Layer"]
        B1["Student Query"]
        B2["Conversation Context"]
        B3["Uploaded Notes / PDFs"]
        B4["Preprocessing"]
    end

    subgraph RouterBlock["🧭 Router Block"]
        C1["Router Agent"]
        C2{"Route Decision"}
        C3["RAG Route"]
        C4["Direct LLM Route"]
        C5["Web Fallback Route"]
    end

    subgraph Retrieval["🔍 Retrieval Block"]
        D1["Query Expansion"]
        D2["Subject Detection"]
        D3["Embedding Model"]
        D4["FAISS Vector Search"]
        D5["Hybrid Reranking"]
        D6["Keyword Fallback"]
        D7["Top Evidence Chunks"]
        D8["Parent / Adjacent Context Expansion"]
    end

    subgraph ReasoningBlock["🧠 Reasoning Block"]
        E1["Reasoning Agent"]
        E2["Key Concept Extraction"]
        E3["Solution Plan"]
        E4["Difficulty Estimation"]
    end

    subgraph TutorBlock["👨‍🏫 Tutor Block"]
        F1["Tutor Agent"]
        F2["Step-by-Step Explanation"]
        F3["Examples"]
        F4["Exam-Oriented Answer"]
        F5["Draft Answer"]
    end

    subgraph CriticBlock["🧪 Critic Block"]
        G1["Critic Agent"]
        G2["Relevance Check"]
        G3["Completeness Check"]
        G4["Accuracy Check"]
        G5["Clarity Check"]
        G6{"Quality Passed?"}
    end

    subgraph GroundingBlock["🛡️ Grounding Block"]
        H1["Grounding Detector"]
        H2["Sentence-Level Claim Check"]
        H3["Evidence Match"]
        H4["Grounding Score"]
        H5{"Grounded Enough?"}
    end

    subgraph MemoryBlock["🧠 Memory & Personalization Block"]
        I1["Conversation Memory Agent"]
        I2["Student Profile"]
        I3["Weak Topic Tracker"]
        I4["Quiz Performance Store"]
        I5["Adaptive Difficulty"]
    end

    subgraph OutputBlock["📤 Output Layer"]
        J1["Final Answer"]
        J2["Citations"]
        J3["Scores"]
        J4["Recommendations"]
    end

    A1 --> B1
    A2 --> B3
    A3 --> B2
    B1 --> B4
    B2 --> B4
    B3 --> B4

    B4 --> C1
    C1 --> C2
    C2 --> C3
    C2 --> C4
    C2 --> C5

    C3 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> D5
    D5 --> D6
    D6 --> D7
    D7 --> D8

    D8 --> E1
    C4 --> E1
    C5 --> E1

    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> F1

    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> F5

    F5 --> G1
    G1 --> G2
    G2 --> G3
    G3 --> G4
    G4 --> G5
    G5 --> G6

    G6 -->|No: send improvement feedback| F1
    G6 -->|Yes| H1

    H1 --> H2
    H2 --> H3
    H3 --> H4
    H4 --> H5

    H5 -->|No: unsupported claims found| F1
    H5 -->|Yes| J1

    J1 --> J2
    J2 --> J3
    J3 --> J4

    J1 --> I1
    I1 --> I2
    I1 --> I3
    I1 --> I4
    I4 --> I5
```

---

## 2. Multi-Agent Refinement Loop

The **Critic Agent does not simply approve or reject the answer**. If the answer is incomplete, unclear, unsupported, or inaccurate, the Critic sends feedback back to the Tutor Agent. The Tutor then refines the answer and submits it again for evaluation.

```mermaid
flowchart LR

    subgraph Reasoning["🧠 Reasoning Agent"]
        A1["Identify Concepts"]
        A2["Build Answer Plan"]
        A3["Estimate Difficulty"]
    end

    subgraph Tutor["👨‍🏫 Tutor Agent"]
        B1["Generate Draft Answer"]
        B2["Add Examples"]
        B3["Add Exam Tips"]
        B4["Add Citations"]
    end

    subgraph Critic["🧪 Critic Agent"]
        C1["Check Relevance"]
        C2["Check Completeness"]
        C3["Check Accuracy"]
        C4["Check Clarity"]
        C5{"Satisfactory?"}
    end

    subgraph Feedback["🔁 Feedback Loop"]
        D1["Missing Concepts"]
        D2["Weak Evidence"]
        D3["Poor Clarity"]
        D4["Low Accuracy"]
    end

    subgraph Approval["✅ Approved Output"]
        E1["Refined Answer"]
        E2["Send to Grounding Detector"]
    end

    A1 --> A2 --> A3 --> B1
    B1 --> B2 --> B3 --> B4 --> C1
    C1 --> C2 --> C3 --> C4 --> C5

    C5 -->|No| D1
    C5 -->|No| D2
    C5 -->|No| D3
    C5 -->|No| D4

    D1 --> B1
    D2 --> B1
    D3 --> B1
    D4 --> B1

    C5 -->|Yes| E1 --> E2
```

---

## 3. RAG Retrieval Pipeline

AcadAI uses a hybrid retrieval system. It does not rely only on vector similarity. It combines semantic search, lexical matching, keyword overlap, subject boosting, and fallback retrieval.

```mermaid
flowchart TD

    subgraph QueryBlock["📌 Query Understanding"]
        A["Student Query"]
        B["Query Cleaning"]
        C["Academic Query Expansion"]
        D["Subject Detection"]
    end

    subgraph VectorBlock["🔢 Vector Retrieval"]
        E["Embedding Model"]
        F["Query Embedding"]
        G["FAISS Index"]
        H["Top Candidate Chunks"]
    end

    subgraph RerankBlock["⚖️ Hybrid Reranking"]
        I["Dense Similarity Score"]
        J["TF-IDF Lexical Score"]
        K["Keyword Overlap Score"]
        L["Subject / Source Boost"]
        M["Final Hybrid Score"]
    end

    subgraph FallbackBlock["🧩 Fallback Retrieval"]
        N["Weak Result Detection"]
        O["Keyword Fallback Search"]
        P["Merge + Deduplicate"]
    end

    subgraph EvidenceBlock["📚 Evidence Builder"]
        Q["Top K Evidence Chunks"]
        R["Neighbor Context Expansion"]
        S["Evidence Pack for Tutor"]
    end

    A --> B --> C --> D
    D --> E --> F --> G --> H

    H --> I
    H --> J
    H --> K
    H --> L

    I --> M
    J --> M
    K --> M
    L --> M

    M --> N
    N -->|Strong Results| Q
    N -->|Weak Results| O --> P --> Q

    Q --> R --> S
```

---

## 4. Grounding & Hallucination Detection Flow

The grounding layer checks whether the generated answer is supported by retrieved evidence. If unsupported claims are found, the answer goes back to the Tutor Agent for refinement.

```mermaid
flowchart TD

    subgraph AnswerBlock["📝 Tutor Output"]
        A["Draft / Refined Answer"]
        B["Sentence Splitter"]
        C["Individual Claims"]
    end

    subgraph EvidenceBlock["📚 Evidence Corpus"]
        D["Retrieved Chunks"]
        E["Evidence Text Builder"]
        F["Citation References"]
    end

    subgraph VerificationBlock["🛡️ Verification Engine"]
        G["Claim-Level Support Check"]
        H["Keyword Overlap"]
        I["Evidence Hit Ratio"]
        J["Citation Match"]
        K["Unsupported Claim Detection"]
    end

    subgraph ScoreBlock["📊 Grounding Score"]
        L["Supported Claims"]
        M["Unsupported Claims"]
        N["Final Grounding Score"]
        O{"Score >= Threshold?"}
    end

    subgraph LoopBlock["🔁 Refinement Loop"]
        P["Send Unsupported Claims to Critic"]
        Q["Critic Feedback"]
        R["Tutor Refines Answer"]
    end

    subgraph FinalBlock["✅ Final Output"]
        S["Approved Answer"]
        T["Grounding Score Display"]
        U["Citations Display"]
    end

    A --> B --> C
    D --> E
    F --> J

    C --> G
    E --> G
    G --> H
    G --> I
    G --> J
    H --> K
    I --> K
    J --> K

    K --> L
    K --> M
    L --> N
    M --> N
    N --> O

    O -->|No| P --> Q --> R --> A
    O -->|Yes| S --> T --> U
```

---

## 5. Memory & Personalization Architecture

AcadAI stores learning context so that future answers become more personalized.

```mermaid
flowchart TD

    subgraph Interaction["👩‍🎓 Student Interaction"]
        A["Question Asked"]
        B["Answer Generated"]
        C["Quiz Attempt"]
        D["Revision Request"]
    end

    subgraph Memory["🧠 Conversation Memory Agent"]
        E["Recent Chat History"]
        F["Topic History"]
        G["Previous Answers"]
        H["Grounding History"]
    end

    subgraph Profile["👤 Student Profile"]
        I["Name / Semester / Branch"]
        J["Preferred Difficulty"]
        K["Learning Goal"]
        L["Weak Topics"]
    end

    subgraph Analytics["📊 Learning Analytics"]
        M["Quiz Scores"]
        N["Repeated Mistakes"]
        O["Low Grounding Topics"]
        P["Weak Topic Counter"]
    end

    subgraph Personalization["🎯 Personalized Output"]
        Q["Context-Aware Follow-Up"]
        R["Adaptive Viva Difficulty"]
        S["Personalized Roadmap"]
        T["Revision Recommendations"]
    end

    A --> E
    B --> G
    C --> M
    D --> T

    E --> F
    F --> G
    G --> H

    I --> Q
    J --> R
    K --> S
    L --> T

    M --> N
    N --> P
    O --> P
    P --> L

    E --> Q
    M --> R
    L --> S
    L --> T
```

---

## 6. Viva / Quiz Mode Architecture

The Viva Studio generates questions from retrieved evidence, evaluates student answers, tracks weak topics, and adapts difficulty.

```mermaid
flowchart TD

    subgraph TopicInput["🎯 Topic Input"]
        A["Student Selects Topic"]
        B["Difficulty Level"]
        C["Conversation Context"]
    end

    subgraph EvidenceRetrieval["🔍 Evidence Retrieval"]
        D["Retrieve Topic Chunks"]
        E["Rerank Evidence"]
        F["Build Topic Context"]
    end

    subgraph QuestionGeneration["🎤 Viva Question Generator"]
        G["Generate Conceptual Questions"]
        H["Generate Applied Questions"]
        I["Generate Tricky Follow-Ups"]
        J["Final Question Set"]
    end

    subgraph Evaluation["🧪 Viva Critic Agent"]
        K["Student Answer"]
        L["Score out of 10"]
        M["Strengths"]
        N["Missing Points"]
        O["Corrected Answer"]
    end

    subgraph Adaptation["📈 Adaptive Learning"]
        P["Update Quiz Attempt"]
        Q["Update Weak Topic Tracker"]
        R{"Average Score"}
        S["Increase Difficulty"]
        T["Maintain Difficulty"]
        U["Reduce Difficulty"]
    end

    A --> D
    B --> G
    C --> F

    D --> E --> F
    F --> G
    F --> H
    F --> I

    G --> J
    H --> J
    I --> J

    J --> K
    K --> L
    L --> M
    L --> N
    N --> O

    L --> P
    N --> Q
    P --> R
    R -->|High| S
    R -->|Medium| T
    R -->|Low| U
```

---

## 7. End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    participant Student
    participant UI as Streamlit UI
    participant Router as Router Agent
    participant Retriever as RAG Retriever
    participant Reasoner as Reasoning Agent
    participant Tutor as Tutor Agent
    participant Critic as Critic Agent
    participant Grounding as Grounding Detector
    participant Memory as Memory Agent

    Student->>UI: Ask academic question
    UI->>Memory: Fetch recent conversation context
    UI->>Router: Send query + memory context

    Router->>Retriever: Select RAG route
    Retriever->>Retriever: Expand query
    Retriever->>Retriever: Detect subject
    Retriever->>Retriever: Search FAISS
    Retriever->>Retriever: Hybrid rerank
    Retriever-->>Reasoner: Return evidence chunks

    Reasoner->>Reasoner: Extract concepts and solution plan
    Reasoner-->>Tutor: Send plan + evidence

    Tutor->>Tutor: Generate draft answer
    Tutor-->>Critic: Submit draft answer

    Critic->>Critic: Evaluate relevance, completeness, accuracy, clarity

    alt Answer is weak
        Critic-->>Tutor: Send feedback for refinement
        Tutor->>Tutor: Improve answer
        Tutor-->>Critic: Resubmit refined answer
    else Answer is acceptable
        Critic-->>Grounding: Send approved answer
    end

    Grounding->>Grounding: Check claim support from evidence

    alt Unsupported claims found
        Grounding-->>Tutor: Send unsupported claims for correction
        Tutor->>Tutor: Refine answer using evidence
        Tutor-->>Grounding: Resubmit grounded answer
    else Grounding passed
        Grounding-->>UI: Return final grounded answer
    end

    UI->>Memory: Store query, answer, subject, grounding score
    UI-->>Student: Display answer, citations, metrics, and recommendations
```

---

## 8. Full AI-LMS Feature Map

```mermaid
flowchart TD

    subgraph AcadAI["🎓 AcadAI AI-LMS Platform"]
        A["Ask with Memory"]
        B["Viva Studio"]
        C["Revision Suite"]
        D["Evaluation Console"]
        E["Learning Memory"]
    end

    subgraph AskModule["💬 Ask with Memory"]
        A1["Question Answering"]
        A2["Context-Aware Follow-Ups"]
        A3["Evidence Citations"]
        A4["Grounding Score"]
    end

    subgraph VivaModule["🎤 Viva Studio"]
        B1["Question Generation"]
        B2["Answer Evaluation"]
        B3["Adaptive Difficulty"]
        B4["Weak Topic Update"]
    end

    subgraph RevisionModule["📚 Revision Suite"]
        C1["Revision Notes"]
        C2["Flashcards"]
        C3["Likely Exam Questions"]
        C4["Study Roadmaps"]
    end

    subgraph EvalModule["📊 Evaluation Console"]
        D1["Relevance Score"]
        D2["Completeness Score"]
        D3["Accuracy Score"]
        D4["Clarity Score"]
        D5["Grounding Score"]
    end

    subgraph MemoryModule["🧠 Learning Memory"]
        E1["Chat History"]
        E2["Weak Topics"]
        E3["Quiz Attempts"]
        E4["Student Profile"]
    end

    A --> A1 --> A2 --> A3 --> A4
    B --> B1 --> B2 --> B3 --> B4
    C --> C1 --> C2 --> C3 --> C4
    D --> D1 --> D2 --> D3 --> D4 --> D5
    E --> E1 --> E2 --> E3 --> E4
```

---

# ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Language | Python |
| LLM | Mistral API |
| Vector Store | FAISS |
| Embeddings | Sentence Transformers |
| Retrieval | Dense + Lexical + Keyword Hybrid Retrieval |
| Reranking | Lightweight Cross Encoder |
| PDF Parsing | PyPDF |
| Web Fallback | DuckDuckGo / Wikipedia fallback |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Environment | `.env` configuration |

---

# 📂Project Structure

```text
AcadAI/
├── acadai_app.py
├── AcadAI_FAISS_STORE/
│   ├── index.faiss
│   └── index.pkl
├── assets/
│   ├── workflow.png
│   ├── architecture.png
│   └── screenshots/
├── requirements.txt
├── .env.example
├── README.md
└── docs/
    ├── architecture.md
    ├── retrieval.md
    └── agents.md
```

---

# 🚀 Installation

```bash
git clone https://github.com/your-username/AcadAI.git
cd AcadAI
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=mistral-large-latest
FAISS_STORE_DIR=./AcadAI_FAISS_STORE
```

Run the app:

```bash
streamlit run acadai_app.py
```

---

# 📊 Evaluation Metrics

AcadAI evaluates every answer using:

| Metric | Purpose |
|---|---|
| Relevance | Checks whether answer matches the question |
| Completeness | Checks whether answer covers required concepts |
| Accuracy | Checks factual correctness |
| Clarity | Checks explanation quality |
| Grounding Score | Checks support from retrieved evidence |
| Evidence Used | Number of retrieved chunks used |
| Response Time | Time taken by the pipeline |

---

# 🔮 Future Enhancements

- OCR pipeline for handwritten/scanned notes
- Diagram understanding
- Voice-based viva mode
- Subject-wise knowledge graph
- LangGraph-based agent orchestration
- Multi-modal learning with images and diagrams
- YouTube lecture grounding
- Offline local LLM support
- Student dashboard with long-term progress analytics

---

# ⭐ Project Vision

AcadAI aims to become a personalized academic intelligence platform that helps students learn from their own material, prepare for exams, practice viva, revise efficiently, and trust AI-generated answers through evidence-grounded learning.
