# Proposed Chapter Revisions for Master's Thesis (AI Focus)

This document contains rewritten and expanded sections from the original outline (see `Master_thesis.docx`), which more accurately reflect the current state of artificial intelligence (LLM, RAG) and its real-world impacts on the labor market based on your research. You can directly edit these texts or copy them into your Word document.

---

## 2.2 Skills as a Component of Human Capital (Addition of AI Context)

In the context of the rapid development of artificial intelligence today, Becker's division of skills can be observed in a completely new dynamic. The ability to use generative AI tools for personal productivity, such as writing effective prompts (prompt engineering) for language models like ChatGPT or GitHub Copilot, is emerging in the labor market as a new form of **general human capital**. These skills are entirely agnostic to the working environment, and employees fully transfer them between different employers. This explains the reluctance of firms to actively invest in this type of training; they expect the worker to acquire these skills independently or to purchase a software license and immediately benefit from it (the so-called _AI-Adopters_ professions).

In contrast, the creation and management of complex enterprise LLM architectures (e.g., Retrieval-Augmented Generation over closed internal company documents, or the deployment and fine-tuning of local open-source models for data security reasons) represents modern **firm-specific human capital**. This knowledge is firmly tied to the specific data infrastructure and domain knowledge of the enterprise (_AI-Natives_ / engineers). At the same time, this creates an enormous wage premium because replacing an employee who understands the internal AI and data structure of the company is highly costly for the firm.

---

## 2.4 Technological Changes and the Labor Market (Integrating LLM and Polarization)

The latest shifts induced by generative artificial intelligence (Large Language Models) challenge the original assumptions of the ALM model (Autor, Levy & Murnane, 2003). While previous waves of digitalization and computing software predominantly replaced routine tasks, modern generative AI demonstrates the capability to perform intermediate cognitive and analytical tasks—activities previously considered safe from automation (e.g., writing code, summarizing complex documents, or creating marketing content).

This development further accelerates the labor market polarization predicted by Goos & Manning (2007) and the so-called "hollowing out" from the perspective of engineering skills. AI is currently driving down the demand for "standard" junior positions (traditional coders or copywriters), as their activities can be effectively handled by an AI assistant. Simultaneously, however, there is a skyrocketing demand for a small but scarce group of _AI Integrators_ and architects who can build these agents and deploy them into production. These shifts, skewed in favor of the most technically capable (Skill-Biased Technological Change), are experiencing their historical peak in the IT market, which predicts the emergence of massive wage premiums between the ordinary "user" and the true "creator" of artificial intelligence.

---

## 3.1 Artificial Intelligence (Shortening History and Shifting Focus to GenAI)

Efforts to create artificial intelligence and its early concepts (symbolic AI) date back to the 1950s, with figures like Alan Turing or John McCarthy. However, the true turning point for the labor market did not occur in the realm of theoretical research, but several decades later with the emergence of the Transformer architecture (2017) and the subsequent rise of Generative AI and Large Language Models (LLMs) at the end of 2022 in the form of the ChatGPT product.

While classic Machine Learning, which dominated the market until roughly 2020, was predominantly analytical and predictive (e.g., data clustering, image classification, price estimation, or sales forecasting), **Generative AI** brought a fundamental shift in utility. Models today demonstrate a semantic understanding of unstructured text and can generate new, coherent content (code, problem resolutions, texts, images) at the level of an interacting agent. This characteristic triggered skyrocketing investments and the creation of entirely new technologies (the so-called LLM ecosystem), leading to an acute need for companies to hire human capital capable of controlling these modern networks and integrating them into corporate products. Therefore, knowledge of neural networks based on the Transformer architecture today represents a distinct segment of the labor market compared to traditional data analysts.

---

## 3.4 AI Skills – Taxonomy and Categorization (A Real-World Reflection of the Market Based on Data)

When evaluating job postings and identifying the actual proportion of tasks (the so-called task-based approach), dividing positions simply into "IT without AI" and "IT with AI" proves to be grossly inadequate. Based on a bottom-up approach to the categorization of AI roles—where an advertisement is evaluated according to explicit hard skills—it is necessary to divide AI skills in today's labor market into at least three qualitatively distinct categories:

**1. Core AI (Model Development and Research)**
These are the creators of the foundational models. These are predominantly deeply expert and scientific roles, requiring university specialization (Masters / PhD) in mathematics or computer science.
_Key skills (signals from postings):_ PyTorch, TensorFlow, LLM model development, architectural modifications of neural networks (Transformers), computational optimization (CUDA).

**2. Applied AI / AI Integration (Integration and Orchestration)**  
This is the fastest-growing segment in the current IT labor market. These roles do not train new models from scratch for millions of dollars. The task of these engineers is to integrate APIs (e.g., from OpenAI or Anthropic) into their own company's products and build so-called AI infrastructure to provide value to end users.
_Key skills (signals from postings):_ RAG pipelines (Retrieval-Augmented Generation), vector databases (Pinecone, Chroma, Milvus), deployment via LangChain, MLOps, and the fine-tuning of open-source models.

**3. AI-Adopters (AI Users for Productivity Enhancement)**
These are not artificial intelligence engineers in the true sense, although the postings for these positions often carry the label or buzzword "AI". These workers consume AI outputs to fulfill routine or non-routine tasks in their regular profession.
_Key skills (signals from postings):_ Maintaining awareness of ChatGPT, GitHub Copilot, using tools like Midjourney or Jasper, "Prompt Engineering," and applying prompts via chat interfaces.

Analysis clearly demonstrates that these three categories face radically different wage premiums, demand curves, and risks of substitution by automation. Ignoring this taxonomy leads to a distortion of macroeconomic statistics regarding the demand for AI.

---

## 3.5 Trends in Demand, the "AI Washing" Phenomenon, and 3.8 Impacts of AI

With the growing media enthusiasm ("hype") surrounding artificial intelligence after 2023, a new demand trend can be observed, which can be termed **"AI Washing"**. This phenomenon describes a situation where organizations intentionally inflate job titles or the introductory paragraphs of postings with the keyword "AI" to appear innovative on the labor market, attract top candidates (the signaling effect), or appease investors. However, a closer semantic analysis of the content (tasks and hard skills) in these postings often reveals that the job profile requires no competencies from _Core AI_ or _Applied AI_, and the working expectations are limited to an _AI-Adopter_ level (e.g., writing better texts with the help of a chatbot).

This discrepancy causes a methodological problem during classic keyword matching. If the labor market is examined merely by searching for the acronym "AI" in descriptions, there is a massive dilution of the extracted data by so-called false-positive non-technical roles. The true, deep demand for IT engineers capable of managing machine learning models is hidden by the market precisely within these clouds of marketing-altered advertisements. It is therefore of fundamental importance to apply advanced filtration to the data via language models, which can contextually exclude "users" from engineering creators.
