#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive skills dictionary for deterministic extraction.

This module contains hardskills and softskills dictionaries used for
deterministic extraction from job postings. Ambiguous short terms (c, r, js, go)
are intentionally excluded - these are handled by LLM with context.

The dictionaries include:
- Canonical skill names (the normalized form)
- Variant mappings (alternative spellings that map to canonical names)
"""

from __future__ import annotations

# =============================================================================
# VARIANT MAPPINGS: Alternative spellings -> Canonical names
# =============================================================================

HARDSKILL_VARIANTS: dict[str, str] = {
    # JavaScript variants
    "javascript": "javascript",
    "java script": "javascript",
    "ecmascript": "javascript",
    "es6": "javascript",
    "es6+": "javascript",
    "javascript (es6+)": "javascript",
    
    # TypeScript variants
    "typescript": "typescript",
    "type script": "typescript",
    
    # Python variants
    "python": "python",
    "python3": "python",
    "python 3": "python",
    
    # Java variants (NOT "j" alone)
    "java": "java",
    "java se": "java",
    "java ee": "java ee",
    "j2ee": "java ee",
    "jakarta ee": "java ee",
    
    # .NET variants
    ".net": "dotnet",
    ".net core": "dotnet",
    ".net (core)": "dotnet",
    ".net (c#)": "dotnet",
    ".net framework": "dotnet",
    "dotnet": "dotnet",
    "dot net": "dotnet",
    "microsoft .net": "dotnet",
    "asp.net": "asp.net",
    "asp .net": "asp.net",
    "ado.net": "ado.net",
    
    # React variants
    "react": "react",
    "react.js": "react",
    "reactjs": "react",
    "react js": "react",
    "react native": "react native",
    "reactnative": "react native",
    
    # Angular variants
    "angular": "angular",
    "angularjs": "angular",
    "angular.js": "angular",
    "angular js": "angular",
    
    # Vue variants
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "vue js": "vue",
    "vuex": "vuex",
    
    # Node.js variants
    "node.js": "node.js",
    "nodejs": "node.js",
    "node js": "node.js",
    "node": "node.js",
    
    # Express variants
    "express": "express",
    "express.js": "express",
    "expressjs": "express",
    
    # Next.js variants
    "next.js": "next.js",
    "nextjs": "next.js",
    "next js": "next.js",
    
    # Nuxt variants
    "nuxt": "nuxt",
    "nuxt.js": "nuxt",
    "nuxtjs": "nuxt",
    
    # Spring variants
    "spring": "spring",
    "spring boot": "spring boot",
    "springboot": "spring boot",
    "spring framework": "spring",
    
    # Django/Flask
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "fast api": "fastapi",
    
    # SQL variants
    "sql": "sql",
    "mysql": "mysql",
    "my sql": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "psql": "postgresql",
    "mssql": "sql server",
    "ms sql": "sql server",
    "sql server": "sql server",
    "microsoft sql server": "sql server",
    "sqlite": "sqlite",
    "plsql": "pl/sql",
    "pl/sql": "pl/sql",
    "tsql": "t-sql",
    "t-sql": "t-sql",
    
    # NoSQL databases
    "nosql": "nosql",
    "mongodb": "mongodb",
    "mongo": "mongodb",
    "mongo db": "mongodb",
    "redis": "redis",
    "cassandra": "cassandra",
    "couchdb": "couchdb",
    "dynamodb": "dynamodb",
    "dynamo db": "dynamodb",
    "elasticsearch": "elasticsearch",
    "elastic search": "elasticsearch",
    "neo4j": "neo4j",
    
    # Cloud platforms
    "aws": "aws",
    "amazon web services": "aws",
    "azure": "azure",
    "microsoft azure": "azure",
    "gcp": "gcp",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    
    # Container/orchestration
    "docker": "docker",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "helm": "helm",
    "openshift": "openshift",
    
    # IaC
    "terraform": "terraform",
    "ansible": "ansible",
    "puppet": "puppet",
    "chef": "chef",
    "cloudformation": "cloudformation",
    
    # CI/CD
    "ci/cd": "ci/cd",
    "cicd": "ci/cd",
    "jenkins": "jenkins",
    "github actions": "github actions",
    "gitlab ci": "gitlab ci",
    "gitlab-ci": "gitlab ci",
    "circleci": "circleci",
    "circle ci": "circleci",
    "travis ci": "travis ci",
    "travisci": "travis ci",
    "azure devops": "azure devops",
    "microsoft azure dev ops": "azure devops",
    "azure pipelines": "azure pipelines",
    "bamboo": "bamboo",
    "teamcity": "teamcity",
    
    # Version control
    "git": "git",
    "github": "github",
    "gitlab": "gitlab",
    "bitbucket": "bitbucket",
    "svn": "svn",
    "subversion": "svn",
    
    # Testing frameworks
    "junit": "junit",
    "j unit": "junit",
    "mockito": "mockito",
    "jest": "jest",
    "jasmine": "jasmine",
    "karma": "karma",
    "mocha": "mocha",
    "chai": "chai",
    "pytest": "pytest",
    "py test": "pytest",
    "unittest": "unittest",
    "selenium": "selenium",
    "cypress": "cypress",
    "playwright": "playwright",
    "puppeteer": "puppeteer",
    "testng": "testng",
    "nunit": "nunit",
    "xunit": "xunit",
    "rspec": "rspec",
    "cucumber": "cucumber",
    "postman": "postman",
    
    # State management
    "redux": "redux",
    "ngrx": "ngrx",
    "rxjs": "rxjs",
    "rx.js": "rxjs",
    "mobx": "mobx",
    "zustand": "zustand",
    "recoil": "recoil",
    
    # APIs
    "rest": "restful api",
    "restful": "restful api",
    "restful api": "restful api",
    "restful apis": "restful api",
    "rest apis": "restful api",
    "rest api": "restful api",
    "graphql": "graphql",
    "graph ql": "graphql",
    "grpc": "grpc",
    "websocket": "websocket",
    "websockets": "websocket",
    "web socket": "websocket",
    "soap": "soap",
    "openapi": "openapi",
    "swagger": "swagger",
    
    # Architecture
    "microservices": "microservices",
    "micro services": "microservices",
    "micro-services": "microservices",
    "monolith": "monolith",
    "serverless": "serverless",
    "event-driven": "event-driven",
    "event driven": "event-driven",
    "soa": "soa",
    "service-oriented architecture": "soa",
    
    # Data/ML frameworks
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "tensorflow": "tensorflow",
    "tensor flow": "tensorflow",
    "pytorch": "pytorch",
    "py torch": "pytorch",
    "keras": "keras",
    "spark": "spark",
    "apache spark": "spark",
    "pyspark": "pyspark",
    "hadoop": "hadoop",
    "airflow": "airflow",
    "apache airflow": "airflow",
    "mlflow": "mlflow",
    "kubeflow": "kubeflow",
    
    # CSS/Styling
    "css": "css",
    "css3": "css",
    "sass": "sass",
    "scss": "scss",
    "less": "less",
    "tailwind": "tailwind",
    "tailwindcss": "tailwind",
    "tailwind css": "tailwind",
    "bootstrap": "bootstrap",
    "material ui": "material ui",
    "material-ui": "material ui",
    "mui": "material ui",
    "styled-components": "styled-components",
    "styled components": "styled-components",
    
    # HTML
    "html": "html",
    "html5": "html",
    
    # Build tools
    "webpack": "webpack",
    "vite": "vite",
    "rollup": "rollup",
    "parcel": "parcel",
    "babel": "babel",
    "gulp": "gulp",
    "grunt": "grunt",
    "maven": "maven",
    "gradle": "gradle",
    "npm": "npm",
    "yarn": "yarn",
    "pnpm": "pnpm",
    
    # OS/Platforms
    "linux": "linux",
    "ubuntu": "ubuntu",
    "centos": "centos",
    "redhat": "redhat",
    "red hat": "redhat",
    "rhel": "redhat",
    "debian": "debian",
    "unix": "unix",
    "windows": "windows",
    "windows server": "windows server",
    "macos": "macos",
    "mac os": "macos",
    "ios": "ios",
    "android": "android",
    
    # Shell/Scripting
    "bash": "bash",
    "shell": "shell",
    "shell scripting": "shell scripting",
    "powershell": "powershell",
    "power shell": "powershell",
    "zsh": "zsh",
    
    # Message queues
    "kafka": "kafka",
    "apache kafka": "kafka",

    "rabbit mq": "rabbitmq",
    "activemq": "activemq",
    "sqs": "sqs",
    "amazon sqs": "sqs",
    "sns": "sns",
    "amazon sns": "sns",
    "redis pub/sub": "redis pub/sub",
    
    # Monitoring/Logging
    "prometheus": "prometheus",
    "grafana": "grafana",
    "datadog": "datadog",
    "new relic": "new relic",
    "newrelic": "new relic",
    "splunk": "splunk",
    "elk": "elk stack",
    "elk stack": "elk stack",
    "logstash": "logstash",
    "kibana": "kibana",
    "cloudwatch": "cloudwatch",
    
    # Other languages (clear names only, NOT single letters)
    "c++": "c++",
    "cpp": "c++",
    "c#": "c#",
    "csharp": "c#",
    "c sharp": "c#",
    "rust": "rust",
    "golang": "golang",
    "kotlin": "kotlin",
    "scala": "scala",
    "swift": "swift",
    "objective-c": "objective-c",
    "objective c": "objective-c",
    "objc": "objective-c",
    "ruby": "ruby",
    "php": "php",
    "perl": "perl",
    "lua": "lua",
    "haskell": "haskell",
    "erlang": "erlang",
    "elixir": "elixir",
    "clojure": "clojure",
    "f#": "f#",
    "fsharp": "f#",
    "dart": "dart",
    "groovy": "groovy",
    "cobol": "cobol",
    "fortran": "fortran",
    "assembly": "assembly",
    "vba": "vba",
    "visual basic": "visual basic",
    "vb.net": "vb.net",
    "matlab": "matlab",
    "sas": "sas",
    "stata": "stata",
    "spss": "spss",
    
    # Frameworks continued
    "rails": "rails",
    "ruby on rails": "rails",
    "laravel": "laravel",
    "symfony": "symfony",
    "codeigniter": "codeigniter",
    "cakephp": "cakephp",
    "nestjs": "nest.js",
    "nest.js": "nest.js",
    "koa": "koa",
    "hapi": "hapi",
    "svelte": "svelte",
    "ember": "ember",
    "ember.js": "ember",
    "backbone": "backbone",
    "backbone.js": "backbone",
    "jquery": "jquery",
    "j query": "jquery",
    "gatsby": "gatsby",
    "remix": "remix",
    "astro": "astro",
    
    # Mobile
    "flutter": "flutter",
    "xamarin": "xamarin",
    "ionic": "ionic",
    "cordova": "cordova",
    "phonegap": "phonegap",
    "swift ui": "swiftui",
    "swiftui": "swiftui",
    "jetpack compose": "jetpack compose",
    
    # Other tools/technologies
    "jira": "jira",
    "confluence": "confluence",
    "slack": "slack",
    "trello": "trello",
    "asana": "asana",
    "notion": "notion",
    "figma": "figma",
    "sketch": "sketch",
    "adobe xd": "adobe xd",
    "invision": "invision",
    "zeplin": "zeplin",
    "nginx": "nginx",
    "apache": "apache",
    "tomcat": "tomcat",
    "iis": "iis",
    "haproxy": "haproxy",
    "kong": "kong",
    "api gateway": "api gateway",

    "json web token": "jwt",
    "saml": "saml",
    "ldap": "ldap",
    "active directory": "active directory",
    "ssl": "ssl",
    "tls": "tls",
    "https": "https",
    "http": "http",
    
    # Agile/Methodologies
    "agile": "agile",
    "scrum": "scrum",
    "kanban": "kanban",
    "devops": "devops",
    "devsecops": "devsecops",
    "tdd": "tdd",
    "test driven development": "tdd",
    "bdd": "bdd",
    "behavior driven development": "bdd",
    "waterfall": "waterfall",
    "lean": "lean",
    "six sigma": "six sigma",
    
    # Data formats
    "json": "json",
    "xml": "xml",
    "yaml": "yaml",
    "yml": "yaml",
    "csv": "csv",
    "protobuf": "protobuf",
    "avro": "avro",
    "parquet": "parquet",
    
    # AI/ML specific
    "machine learning": "machine learning",
    "deep learning": "deep learning",
    "neural networks": "neural networks",
    "neural network": "neural networks",
    "nlp": "nlp",
    "natural language processing": "nlp",
    "computer vision": "computer vision",
    "cv": "computer vision",
    "llm": "llm",
    "large language models": "llm",
    "gpt": "gpt",
    "chatgpt": "chatgpt",
    "openai": "openai",
    "langchain": "langchain",
    "hugging face": "huggingface",
    "huggingface": "huggingface",
    "transformers": "transformers",
    
    # BI Tools
    "tableau": "tableau",
    "power bi": "power bi",
    "powerbi": "power bi",
    "looker": "looker",
    "metabase": "metabase",
    "qlik": "qlik",
    "qlikview": "qlikview",
    "qliksense": "qliksense",
    "sisense": "sisense",
    "excel": "excel",
    "microsoft excel": "excel",
    
    # ETL/Data
    "etl": "etl",
    "elt": "elt",
    "dbt": "dbt",
    "data build tool": "dbt",
    "talend": "talend",
    "informatica": "informatica",
    "ssis": "ssis",
    "fivetran": "fivetran",
    "airbyte": "airbyte",
    "stitch": "stitch",
    "snowflake": "snowflake",
    "databricks": "databricks",
    "redshift": "redshift",
    "bigquery": "bigquery",
    "big query": "bigquery",
    
    # Other common skills

    "oop": "oop",
    "object oriented": "oop",
    "object-oriented": "oop",
    "design patterns": "design patterns",
    "solid": "solid",
    "solid principles": "solid",
    "clean code": "clean code",
    "clean architecture": "clean architecture",
    "ddd": "ddd",
    "domain driven design": "ddd",
    "domain-driven design": "ddd",
    "cqrs": "cqrs",
    "event sourcing": "event sourcing",
    "debugging": "debugging",

    "performance tuning": "performance tuning",
    "optimization": "optimization",

    "high availability": "high availability",
    "load balancing": "load balancing",
    "caching": "caching",
    "unit testing": "unit testing",
    "integration testing": "integration testing",
    "e2e testing": "e2e testing",
    "end-to-end testing": "e2e testing",
    "code review": "code review",
    "pair programming": "pair programming",
    "continuous integration": "continuous integration",
    "continuous deployment": "continuous deployment",
    "continuous delivery": "continuous delivery",
    "infrastructure as code": "infrastructure as code",
    "iac": "infrastructure as code",
    "container orchestration": "container orchestration",
    "web development": "web development",

    "backend development": "backend development",
    "back-end development": "backend development",
    "back end development": "backend development",
    "full stack": "full stack",
    "full-stack": "full stack",
    "fullstack": "full stack",
    "mobile development": "mobile development",
    "cross-platform": "cross-platform",
    "responsive design": "responsive design",
    "ui/ux": "ui/ux",
    "ui": "ui",
    "ux": "ux",
    "user interface": "ui",
    "user experience": "ux",
    "accessibility": "accessibility",
    "a11y": "accessibility",
    "seo": "seo",
    "search engine optimization": "seo",
    "cybersecurity": "cybersecurity",
    "cyber security": "cybersecurity",
    "information security": "information security",
    "infosec": "information security",
    "penetration testing": "penetration testing",
    "pen testing": "penetration testing",
    "vulnerability assessment": "vulnerability assessment",
    "encryption": "encryption",
    "firewall": "firewall",
    "vpn": "vpn",
    "siem": "siem",
    "networking": "networking",
    "tcp/ip": "tcp/ip",
    "dns": "dns",
    "dhcp": "dhcp",
    "routing": "routing",
    "switching": "switching",
    "load balancer": "load balancing",
    "cdn": "cdn",
    "content delivery network": "cdn",
    "storage": "storage",

    "gcs": "gcs",
    "google cloud storage": "gcs",
    "virtualization": "virtualization",
    "vmware": "vmware",
    "hyper-v": "hyper-v",
    "hyperv": "hyper-v",
    "virtual machines": "virtual machines",
    "vms": "virtual machines",
    "ec2": "ec2",
    "lambda": "lambda",
    "aws lambda": "lambda",
    "azure functions": "azure functions",
    "google cloud functions": "cloud functions",
    "cloud functions": "cloud functions",
    "ecs": "ecs",
    "eks": "eks",
    "aks": "aks",
    "gke": "gke",
    "fargate": "fargate",
    "rds": "rds",
    "aurora": "aurora",
    "s3": "s3",
    "cloudfront": "cloudfront",
    "route53": "route53",
    "route 53": "route53",
    "iam": "iam",
    "vpc": "vpc",
    "api": "api",
    "sdk": "sdk",
    "oauth": "oauth",
    "jwt": "jwt",
    "sso": "sso",
    "single sign-on": "sso",
    "webscraping": "web scraping",
    "web scraping": "web scraping",
    "beautifulsoup": "beautifulsoup",
    "scrapy": "scrapy",
    "requests": "requests",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "celery": "celery",
    "redis queue": "redis queue",
    "rq": "redis queue",
    "rabbitmq": "rabbitmq",
    
    # =========================================================================
    # ADDITIONAL SKILLS FROM DETECTION FAMILIES A-H
    # =========================================================================
    
    # --- Family A: Programming Languages & Frameworks (additions) ---
    "r programming": "r",
    "r language": "r",
    "rlang": "r",
    # Note: single "r" is ambiguous (handled by LLM)
    
    # --- Family B: Data, Cloud, and Orchestration ---
    # Cloud infrastructure concepts
    "cloud computing": "cloud computing",
    "cloud infrastructure": "cloud infrastructure",
    "cloud architecture": "cloud architecture",
    "cloud services": "cloud services",
    "cloud native": "cloud native",
    "cloud-native": "cloud native",
    "multi-cloud": "multi-cloud",
    "multi cloud": "multi-cloud",
    "hybrid cloud": "hybrid cloud",
    
    # Data engineering
    "data engineering": "data engineering",
    "data pipelines": "data pipelines",
    "data pipeline": "data pipelines",
    "data warehouses": "data warehouses",
    "data warehouse": "data warehouses",
    "data warehousing": "data warehouses",
    "data modeling": "data modeling",
    "data modelling": "data modeling",
    "data governance": "data governance",
    "data architecture": "data architecture",
    "data quality": "data quality",
    "data management": "data management",
    "data lake": "data lake",
    "data lakes": "data lake",
    "data lakehouse": "data lakehouse",
    "data mesh": "data mesh",
    "data catalog": "data catalog",
    "data lineage": "data lineage",
    "master data management": "master data management",
    "mdm": "master data management",
    "data integration": "data integration",
    "data migration": "data migration",
    "data transformation": "data transformation",
    "batch processing": "batch processing",
    "stream processing": "stream processing",
    "real-time data": "real-time data",
    "real time data": "real-time data",
    
    # Orchestration tools
    "orchestration": "orchestration",
    "workflow orchestration": "workflow orchestration",
    "data orchestration": "data orchestration",
    "prefect": "prefect",
    "dagster": "dagster",
    "luigi": "luigi",
    "argo": "argo",
    "argo workflows": "argo workflows",
    "step functions": "step functions",
    "aws step functions": "step functions",
    "mage": "mage",
    
    # MLOps
    "mlops": "mlops",
    "ml ops": "mlops",
    "aiops": "aiops",
    "ai ops": "aiops",
    "feature store": "feature store",
    "model serving": "model serving",
    "model deployment": "model deployment",
    "model monitoring": "model monitoring",
    "experiment tracking": "experiment tracking",
    "sagemaker": "sagemaker",
    "aws sagemaker": "sagemaker",
    "vertex ai": "vertex ai",
    "google vertex ai": "vertex ai",
    "azure ml": "azure ml",
    "azure machine learning": "azure ml",
    
    # --- Family C: Integration & Data Exchange ---
    "web services": "web services",
    "web api": "web api",
    "web apis": "web api",
    "http protocol": "http",
    "https protocol": "https",
    "api design": "api design",
    "api development": "api development",
    "api integration": "api integration",
    "api management": "api management",
    "api security": "api security",
    "microservice architecture": "microservices",
    "service mesh": "service mesh",
    "istio": "istio",
    "envoy": "envoy",
    "linkerd": "linkerd",
    
    # --- Family D: Security & Infrastructure ---
    # Cybersecurity
    "network security": "network security",
    "application security": "application security",
    "appsec": "application security",
    "cloud security": "cloud security",
    "endpoint security": "endpoint security",
    "security architecture": "security architecture",
    "security operations": "security operations",
    "secops": "security operations",
    "soc": "soc",
    "soc operations": "soc",
    "security operations center": "soc",
    
    # Incident response & forensics
    "incident response": "incident response",
    "ir": "incident response",
    "forensics": "forensics",
    "digital forensics": "digital forensics",
    "malware analysis": "malware analysis",
    "threat detection": "threat detection",
    "threat hunting": "threat hunting",
    "threat intelligence": "threat intelligence",
    "vulnerability management": "vulnerability management",
    "vulnerability scanning": "vulnerability scanning",
    "patch management": "patch management",
    "risk assessment": "risk assessment",
    "security assessment": "security assessment",
    
    # Security tools & technologies
    "firewall management": "firewall management",
    "ids": "ids",
    "intrusion detection": "ids",
    "ips": "ips",
    "intrusion prevention": "ips",
    "edr": "edr",
    "endpoint detection": "edr",
    "xdr": "xdr",
    "extended detection": "xdr",
    "soar": "soar",
    "security orchestration": "soar",
    "dlp": "dlp",
    "data loss prevention": "dlp",
    "pam": "pam",
    "privileged access management": "pam",
    "identity management": "identity management",
    "identity and access management": "iam",
    "zero trust": "zero trust",
    "zero trust architecture": "zero trust",
    
    # Security frameworks & standards
    "nist": "nist",
    "nist framework": "nist",
    "nist cybersecurity": "nist",
    "iso 27001": "iso27001",
    "iso27001": "iso27001",
    "iso 27002": "iso27002",
    "iso27002": "iso27002",
    "soc 2": "soc 2",
    "soc2": "soc 2",
    "pci dss": "pci dss",
    "pci-dss": "pci dss",
    "hipaa": "hipaa",
    "gdpr": "gdpr",
    "fedramp": "fedramp",
    "cis benchmarks": "cis benchmarks",
    "owasp": "owasp",
    "owasp top 10": "owasp",
    "mitre att&ck": "mitre att&ck",
    "mitre attack": "mitre att&ck",
    
    # Certifications (Security)
    "cissp": "cissp",
    "cism": "cism",
    "cisa": "cisa",
    "giac": "giac",
    "ceh": "ceh",
    "certified ethical hacker": "ceh",
    "oscp": "oscp",
    "comptia security+": "comptia security+",
    "comptia security plus": "comptia security+",
    "security+": "comptia security+",
    "comptia network+": "comptia network+",
    "network+": "comptia network+",
    "comptia a+": "comptia a+",
    "ccna": "ccna",
    "ccnp": "ccnp",
    "ccie": "ccie",
    "ccna": "ccna",
    "ccnp": "ccnp",
    "ccie": "ccie",
    
    # --- Family I: Generative AI & LLMs ---
    "generative ai": "generative ai",
    "genai": "generative ai",
    "llm": "llm",
    "large language model": "llm",
    "gpt": "gpt",
    "chatgpt": "chatgpt",
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "claude",
    "gemini": "gemini",
    "bard": "gemini",
    "google bard": "gemini",
    "vertex ai": "vertex ai",
    "langchain": "langchain",
    "llamaindex": "llamaindex",
    "llama index": "llamaindex",
    "huggingface": "huggingface",
    "hugging face": "huggingface",
    "transformers": "transformers",
    "bert": "bert",
    "pinecone": "pinecone",
    "weaviate": "weaviate",
    "chromadb": "chromadb",
    "milvus": "milvus",
    "faiss": "faiss",
    "vector database": "vector database",
    "vector db": "vector database",
    "rag": "rag",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "prompt engineering": "prompt engineering",
    "fine-tuning": "fine-tuning",
    "fine tuning": "fine-tuning",
    "lora": "lora",
    "qlora": "qlora",
    "stable diffusion": "stable diffusion",
    "midjourney": "midjourney",
    "dalle": "dalle",
    "dall-e": "dalle",
    "whisper": "whisper",
    "cohere": "cohere",
    "mistral": "mistral",
    "llama": "llama",
    "ollama": "ollama",
    
    # --- Family E: Software Engineering Domains & Tools ---
    "software development": "software engineering",
    "software engineering": "software engineering",
    "system integration": "system integration",
    "systems integration": "system integration",
    "software architecture": "software architecture",
    "system design": "system design",
    "systems design": "system design",
    "solution architecture": "solution architecture",
    "enterprise architecture": "enterprise architecture",
    "technical architecture": "technical architecture",
    
    # Testing & QA
    "qa": "qa",
    "quality assurance": "qa",
    "qa automation": "test automation",
    "test automation": "test automation",
    "automated testing": "test automation",
    "manual testing": "manual testing",
    "regression testing": "regression testing",
    "functional testing": "functional testing",
    "performance testing": "performance testing",
    "load testing": "load testing",
    "stress testing": "stress testing",
    "security testing": "security testing",
    "api testing": "api testing",
    "mobile testing": "mobile testing",
    "test planning": "test planning",
    "test strategy": "test strategy",
    "test cases": "test cases",
    "test management": "test management",
    "jmeter": "jmeter",
    "apache jmeter": "jmeter",
    "gatling": "gatling",
    "k6": "k6",
    "locust": "locust",
    "appium": "appium",
    "robot framework": "robot framework",
    "testcafe": "testcafe",
    "sonarqube": "sonarqube",
    "sonar": "sonarqube",
    "code quality": "code quality",
    "static analysis": "static analysis",
    "code coverage": "code coverage",
    
    # Version control & collaboration
    "version control": "version control",
    "source control": "version control",
    "branching strategies": "branching strategies",
    "gitflow": "gitflow",
    "trunk-based development": "trunk-based development",
    "trunk based development": "trunk-based development",
    
    # OOP & Design
    "object-oriented programming": "oop",
    "object oriented programming": "oop",
    "object-oriented design": "object-oriented design",
    "object oriented design": "object-oriented design",
    "object-oriented analysis and design": "ooad",
    "object oriented analysis and design": "ooad",
    "ooad": "ooad",
    "uml": "uml",
    "unified modeling language": "uml",
    
    # IDEs & Development Tools
    "visual studio code": "visual studio code",
    "vscode": "visual studio code",
    "vs code": "visual studio code",
    "visual studio": "visual studio",
    "intellij": "intellij",
    "intellij idea": "intellij",
    "pycharm": "pycharm",
    "webstorm": "webstorm",
    "eclipse": "eclipse",
    "xcode": "xcode",
    "android studio": "android studio",
    "rider": "rider",
    "sublime text": "sublime text",
    "vim": "vim",
    "neovim": "neovim",
    "emacs": "emacs",
    
    # --- Family F: Analytics & Visualization ---
    "data analysis": "data analysis",
    "data analytics": "data analytics",
    "business analytics": "business analytics",
    "data visualization": "data visualization",
    "data viz": "data visualization",
    "business intelligence": "business intelligence",
    "bi tools": "business intelligence",
    "bi development": "business intelligence",
    "reporting": "reporting",
    "report development": "reporting",
    "dashboards": "dashboards",
    "dashboard development": "dashboards",
    "data science": "data science",
    "predictive analytics": "predictive analytics",
    "prescriptive analytics": "prescriptive analytics",
    "descriptive analytics": "descriptive analytics",
    "statistical analysis": "statistical analysis",
    "statistics": "statistics",
    "statistical modeling": "statistical modeling",
    "a/b testing": "a/b testing",
    "ab testing": "a/b testing",
    "experimentation": "experimentation",
    "hypothesis testing": "hypothesis testing",
    
    # BI Tools additions
    "dax": "dax",
    "power query": "power query",
    "m language": "m language",
    "calculated columns": "calculated columns",
    "measures": "measures",
    "data studio": "data studio",
    "google data studio": "data studio",
    "looker studio": "looker studio",
    "superset": "superset",
    "apache superset": "superset",
    "redash": "redash",
    "mode analytics": "mode analytics",
    "amplitude": "amplitude",
    "mixpanel": "mixpanel",
    "google analytics": "google analytics",
    "ga4": "ga4",
    "adobe analytics": "adobe analytics",
    "segment": "segment",
    
    # --- Family G: Certifications & Standards (Cloud) ---
    "aws certified": "aws certified",
    "aws certification": "aws certified",
    "aws solutions architect": "aws solutions architect",
    "aws developer": "aws developer",
    "aws sysops": "aws sysops",
    "aws devops engineer": "aws devops engineer",
    "aws data engineer": "aws data engineer",
    "aws machine learning": "aws machine learning specialty",
    "azure administrator": "azure administrator",
    "azure developer": "azure developer",
    "azure solutions architect": "azure solutions architect",
    "azure devops engineer": "azure devops engineer",
    "azure data engineer": "azure data engineer",
    "az-900": "az-900",
    "az-104": "az-104",
    "az-204": "az-204",
    "az-305": "az-305",
    "az-400": "az-400",
    "gcp certified": "gcp certified",
    "google cloud certified": "gcp certified",
    "gcp professional": "gcp professional",
    "gcp associate": "gcp associate",
    "cloud practitioner": "cloud practitioner",
    "aws cloud practitioner": "cloud practitioner",
    "cka": "cka",
    "certified kubernetes administrator": "cka",
    "ckad": "ckad",
    "certified kubernetes application developer": "ckad",
    "cks": "cks",
    "certified kubernetes security specialist": "cks",
    "terraform certified": "terraform certified",
    "hashicorp certified": "hashicorp certified",
    "pmp": "pmp",
    "project management professional": "pmp",
    "certified scrum master": "csm",
    "csm": "csm",
    "safe": "safe",
    "scaled agile": "safe",
    "togaf": "togaf",
    "itil": "itil",
    
    # --- Family H: Tools, Libraries, UI Frameworks ---
    # Testing libraries
    "testing library": "testing library",
    "react testing library": "react testing library",
    "vue testing library": "vue testing library",
    "enzyme": "enzyme",
    "vitest": "vitest",
    "storybook": "storybook",
    
    # UI frameworks additions
    "ant design": "ant design",
    "antd": "ant design",
    "chakra ui": "chakra ui",
    "chakra": "chakra ui",
    "radix": "radix",
    "radix ui": "radix ui",
    "headless ui": "headless ui",
    "shadcn": "shadcn",
    "shadcn/ui": "shadcn",
    "daisyui": "daisyui",
    "daisy ui": "daisyui",
    "bulma": "bulma",
    "foundation": "foundation",
    "semantic ui": "semantic ui",
    "prime react": "primereact",
    "primereact": "primereact",
    "primevue": "primevue",
    "vuetify": "vuetify",
    "quasar": "quasar",
    
    # Frontend development
    "frontend development": "frontend development",
    "front-end development": "frontend development",
    "front end development": "frontend development",
    "frontend": "frontend development",
    "front-end": "frontend development",
    "micro frontends": "micro frontends",
    "micro-frontends": "micro frontends",
    "microfrontends": "micro frontends",
    "module federation": "module federation",
    "single spa": "single-spa",
    "single-spa": "single-spa",
    
    # Accessibility
    "web accessibility": "accessibility",
    "wcag": "wcag",
    "wcag 2.0": "wcag",
    "wcag 2.1": "wcag",
    "wcag 2.2": "wcag",
    "aria": "aria",
    "wai-aria": "aria",
    "screen reader": "screen reader",
    
    # Animation & Graphics
    "css animations": "css animations",
    "web animations": "web animations",
    "gsap": "gsap",
    "greensock": "gsap",
    "framer motion": "framer motion",
    "lottie": "lottie",
    "three.js": "three.js",
    "threejs": "three.js",
    "webgl": "webgl",
    "canvas": "canvas",
    "svg": "svg",
    "d3.js": "d3.js",
    "d3": "d3.js",
    "chart.js": "chart.js",
    "chartjs": "chart.js",
    "highcharts": "highcharts",
    "plotly": "plotly",
    "echarts": "echarts",
    "recharts": "recharts",
    "nivo": "nivo",
    
    # State management additions
    "tanstack query": "tanstack query",
    "react query": "tanstack query",
    "swr": "swr",
    "jotai": "jotai",
    "xstate": "xstate",
    "pinia": "pinia",
    
    # Form handling
    "react hook form": "react hook form",
    "formik": "formik",
    "yup": "yup",
    "zod": "zod",
    
    # Package managers & bundlers additions
    "esbuild": "esbuild",
    "swc": "swc",
    "turbopack": "turbopack",
    "turborepo": "turborepo",
    "nx": "nx",
    "lerna": "lerna",
    "monorepo": "monorepo",
    "mono repo": "monorepo",
    
    # Documentation
    "swagger ui": "swagger ui",
    "redoc": "redoc",
    "readme": "readme",
    "jsdoc": "jsdoc",
    "typedoc": "typedoc",
    "sphinx": "sphinx",
    "mkdocs": "mkdocs",
}

# =============================================================================
# CANONICAL HARDSKILLS: The set of all normalized skill names (for lookup)
# =============================================================================

HARDSKILLS: set[str] = set(HARDSKILL_VARIANTS.values())

# =============================================================================
# SKILL FAMILIES: Categorization of skills by domain
# =============================================================================

# =============================================================================
# SKILL FAMILIES: Categorization of skills by domain
# =============================================================================

# 1. Programming Languages (Core)
CAT_LANGUAGES: set[str] = {
    "javascript", "typescript", "python", "java", "c++", "c#", "golang", "rust",
    "kotlin", "scala", "swift", "objective-c", "ruby", "php", "perl", "lua",
    "haskell", "erlang", "elixir", "clojure", "f#", "dart", "groovy", "cobol",
    "fortran", "assembly", "vba", "visual basic", "vb.net", "matlab", "sas",
    "stata", "spss", "r", "bash", "shell", "powershell", "zsh",
}

# 2. Frontend Development
CAT_FRONTEND: set[str] = {
    # Frameworks & Libs
    "react", "angular", "vue", "vuex", "svelte", "ember", "backbone", "jquery",
    "gatsby", "remix", "astro", "next.js", "nuxt", "hugo", "jekyll",
    # Styling
    "html", "css", "sass", "scss", "less", "tailwind", "bootstrap", "material ui",
    "styled-components", "bulma", "foundation", "semantic ui", "ant design",
    "chakra ui", "radix", "radix ui", "headless ui", "shadcn", "daisyui",
    "primereact", "primevue", "vuetify", "quasar",
    # Tools
    "webpack", "vite", "rollup", "parcel", "babel", "esbuild", "swc",
    "frontend development", "responsive design", "accessibility", "wcag", "aria",
    "ui", "ux", "ui/ux", "figma", "sketch", "adobe xd", "invision", "zeplin",
    "canvas", "svg", "webgl", "three.js", "d3.js", "chart.js", "highcharts",
    "plotly", "echarts", "recharts", "nivo", "gsap", "framer motion", "lottie",
    "micro frontends", "module federation", "single-spa",
    "state management", "redux", "ngrx", "rxjs", "mobx", "zustand", "recoil",
    "tanstack query", "swr", "jotai", "xstate", "pinia",
    "react hook form", "formik", "yup", "zod",
    "storybook", "testing library", "react testing library", "vue testing library", "enzyme",
    "jsp", "screen reader", "gulp", "grunt", "turbopack", "turborepo", "nx", "lerna", "monorepo", "mono repo", "pnpm", "yarn", "npm", "maven", "gradle",
    "jsdoc", "typedoc", "swagger ui", "redoc", "readme", "sphinx", "mkdocs",
    "sdk", "api", # Placeholders often used in frontend context or generic
    "web animations", "css animations", "seo",
}

# 3. Backend Development
CAT_BACKEND: set[str] = {
    # Frameworks
    "node.js", "express", "nest.js", "koa", "hapi",
    "spring", "spring boot", "java ee", "jakarta ee",
    "django", "flask", "fastapi", "rails", "laravel", "symfony", "codeigniter",
    "cakephp", "asp.net", "ado.net", "entity framework",
    # Concepts
    "backend development", "full stack", "api development", "api design",
    "api integration", "api management", "microservices", "serverless",
    "event-driven", "soa", "monolith", "cqrs", "event sourcing", "ddd",
    "clean architecture", "hexagonal architecture",
    "graphql", "grpc", "websocket", "soap", "rest", "restful api",
    "openapi", "swagger", "web api", "web services", "http", "https", "aiohttp", "httpx", "requests", "web scraping", "beautifulsoup", "scrapy",
    "celery", "rabbitmq", "activemq", "kafka", "redis queue", "rq",
    "nginx", "apache", "tomcat", "iis", "haproxy", "kong",
    "rest apis", "restful apis", "rest api", "graph ql", "web socket", "websockets", "micro-services", "micro services", "server less",
    "redis pub/sub",
    # .NET
    "dotnet", ".net", ".net core", ".net (core)", ".net (c#)", ".net framework", "dot net", "microsoft .net",
}

# 4. Mobile & Desktop Development
CAT_MOBILE_DESKTOP: set[str] = {
    "mobile development", "android", "ios", "react native", "flutter",
    "xamarin", "ionic", "cordova", "phonegap", "swiftui", "jetpack compose",
    "electron", "qt", "gtk", "wpf", "winforms", "macos", "windows",
}

# 5. Databases & Storage
CAT_DATABASES: set[str] = {
    # Relational
    "sql", "mysql", "postgresql", "sql server", "sqlite", "oracle", "mariadb",
    "pl/sql", "t-sql", "rds", "aurora", "microsoft sql server", "ms sql", "mssql", "postgres",
    # NoSQL
    "nosql", "mongodb", "cassandra", "couchdb", "dynamodb", "hbase",
    "neo4j", "redis", "memcached", "firestore", "firebase", "mongo", "mongo db", "dynamo db",
    # Vector
    "vector database", "pinecone", "weaviate", "chromadb", "milvus", "faiss", "vector db",
    # Search
    "elasticsearch", "solr", "algolia", "elastic search",
    # Storage
    "storage", "s3", "blob storage", "gcs", "minio", "google cloud storage",
}

# 6. Data Engineering & Big Data
CAT_DATA_ENGINEERING: set[str] = {
    "data engineering", "etl", "elt", "data pipelines", "data warehousing",
    "data modeling", "data governance", "data architecture", "data quality",
    "data catalog", "data lineage", "master data management", "data integration",
    "data migration", "batch processing", "stream processing", "real-time data",
    "spark", "pyspark", "hadoop", "hive", "mapreduce", "flink", "storm",
    "databricks", "snowflake", "redshift", "bigquery", "synapse",
    "airflow", "prefect", "dagster", "luigi", "nifi", "dbt", "talend",
    "informatica", "ssis", "fivetran", "airbyte", "stitch",
    "kafka", "kinesis", "pubsub", "data lake", "data lakes", "data lakehouse", "data mesh", "mdm",
    "avro", "parquet", "protobuf", "csv", "json", "xml", "yaml", "yml",
    "data transformation", "data pipeline", "data warehouse", "big query", "apache spark", "apache kafka", "apache airflow",
    "data modelling", "real time data", "data orchestration", "data build tool",
    "orchestration", "workflow orchestration", "data warehouses",
    "mage",
}

# 7. Data Science & Machine Learning
CAT_DATA_SCIENCE: set[str] = {
    "data science", "machine learning", "deep learning", "neural networks",
    "nlp", "computer vision", "predictive modeling", "statistical modeling",
    "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "statsmodels",
    "tensorflow", "keras", "pytorch", "xgboost", "lightgbm", "catboost",
    "opencv", "nltk", "spacy",
    "mlops", "aiops", "mlflow", "kubeflow", "feature store", "model serving",
    "model deployment", "model monitoring", "experiment tracking", "sagemaker", "vertex ai", "azure ml",
    "neural network", "natural language processing", "cv", "large language models", "py torch", "tensor flow", "aws sagemaker", "google vertex ai", "azure machine learning",
}

# 8. Generative AI & LLMs
CAT_GEN_AI: set[str] = {
    "generative ai", "llm", "large language model", "gpt", "chatgpt",
    "openai", "anthropic", "claude", "gemini", "cohere", "mistral",
    "llama", "ollama", "huggingface", "transformers", "bert",
    "stable diffusion", "midjourney", "dalle", "whisper",
    "rag", "retrieval augmented generation", "prompt engineering",
    "fine-tuning", "lora", "qlora", "langchain", "llamaindex", "genai",
    "retrieval-augmented generation", "fine tuning", "dall-e", "llama index", "hugging face",
}

# 9. Business Intelligence & Analytics
CAT_ANALYTICS: set[str] = {
    "data analytics", "business analytics", "business intelligence",
    "data visualization", "dashboards", "reporting",
    "power bi", "tableau", "looker", "looker studio", "qlik", "sisense",
    "superset", "metabase", "redash", "mode analytics", "amplitude",
    "mixpanel", "google analytics", "ga4", "adobe analytics", "segment",
    "excel", "dax", "power query",
    "statistics", "a/b testing", "experimentation", "statistical analysis",
    "qliksense", "qlikview", "data studio", "google data studio",
    "measures", "calculated columns", "m language",
    "predictive analytics", "prescriptive analytics", "descriptive analytics",
    "data viz", "bi tools", "bi development", "report development", "dashboard development", "ab testing", "hypothesis testing", "apache superset", "microsoft excel", "powerbi",
    "data analysis", # Missing
}

# 10. Cloud Computing (Core)
CAT_CLOUD: set[str] = {
    "cloud computing", "cloud infrastructure", "cloud architecture",
    "cloud native", "multi-cloud", "hybrid cloud",
    "aws", "azure", "gcp", "openstack", "heroku", "netlify", "vercel",
    "digitalocean", "linode",
    "ec2", "lambda", "ecs", "eks", "fargate", "vpc", "iam", "cloudformation",
    "azure functions", "cloud functions",
    "amazon web services", "microsoft azure", "google cloud", "google cloud platform",
    "step functions", "aws step functions", "multi cloud", "cloud-native", "aws lambda", "rhel",
    "sqs", "sns", "route53", "cloudfront", "amazon sqs", "amazon sns", "route 53",
    "gke", "aks", # Missing kubernetes services
    "cloud services",
}

# 11. DevOps & Containers
CAT_DEVOPS: set[str] = {
    "devops", "devsecops", "sre", "platform engineering",
    "ci/cd", "continuous integration", "continuous deployment",
    "jenkins", "github actions", "gitlab ci", "circleci", "travis ci",
    "azure devops", "bamboo", "teamcity", "argo", "flux",
    "docker", "kubernetes", "k8s", "helm", "openshift", "podman",
    "container orchestration", "service mesh", "istio", "linkerd",
    "terraform", "ansible", "puppet", "chef", "infrastructure as code", "iac",
    "pulumi", "vagrant",
    "prometheus", "grafana", "datadog", "new relic", "splunk",
    "elk stack", "logstash", "kibana", "cloudwatch", "envoy", "argo workflows",
    "gitlab-ci", "circle ci", "travisci", "microsoft azure dev ops", "azure pipelines",
    "newrelic", "elk", "continuous delivery",
    "virtual machines", "vmware", "hyper-v", "hyperv", "vms", "virtualization", # Infrastructure
}

# 12. Operating Systems & Hardware (Embedded/IoT)
CAT_OS_HARDWARE: set[str] = {
    "linux", "ubuntu", "debian", "centos", "redhat", "fedora", "alpine",
    "unix", "windows server",
    "embedded systems", "iot", "firmware", "microcontrollers", "arduino",
    "raspberry pi", "rtos", "embedded c", "system programming",
    "networking", "tcp/ip", "dns", "dhcp", "ip", # Basic networking often with OS
    "red hat", "mac os", "shell scripting", # Scripting
}

# 13. Networking & Protocols
CAT_NETWORKING: set[str] = {
    "networking", "network security", "tcp/ip", "dns", "http", "https",
    "routing", "switching", "load balancing", "cdn", "vpn", "firewall",
    "ssl", "tls", "wireshark",
    "api gateway", "proxy", "reverse proxy",
    "load balancer", "content delivery network", "http protocol", "https protocol",
}

# 14. Security, Identity & Compliance
CAT_SECURITY: set[str] = {
    "cybersecurity", "information security", "application security",
    "cloud security", "endpoint security", "network security",
    "identity management", "iam", "oauth", "oidc", "saml", "sso", "jwt",
    "active directory", "ldap", "okta", "auth0",
    "penetration testing", "vulnerability assessment", "threat modeling",
    "security operations", "soc", "siem", "splunk",
    "cryptography", "encryption", "pki",
    "compliance", "gdpr", "hipaa", "pci dss", "soc 2", "iso27001", "nist",
    "owasp",
    "cissp", "ceh", "oscp", "comptia security+",
    "security architecture", "threat detection", "soar", "patch management",
    "incident response", "forensics", "digital forensics", "malware analysis",
    "threat hunting", "threat intelligence", "vulnerability scanning", "vulnerability management",
    "risk assessment", "security assessment", "firewall management",
    "ids", "ips", "edr", "xdr", "dlp", "pam", "zero trust",
    "iso27002", "fedramp", "cis benchmarks", "mitre att&ck",
    "cism", "cisa", "giac", "comptia network+", "comptia a+",
    "cyber security", "infosec", "pen testing", "security operations center", "soc operations", "identity and access management", "single sign-on", "json web token", "nist framework", "nist cybersecurity", "owasp top 10", "mitre attack", "certified ethical hacker", "comptia security plus", "security+", "network+",
    "soc2", "pci-dss", "data loss prevention", "privileged access management", "zero trust architecture", "extended detection", "endpoint detection", "intrusion prevention", "intrusion detection",
    "api security",
}

# 15. Testing & QA
CAT_QA: set[str] = {
    "qa", "quality assurance", "software testing", "test automation",
    "automated testing", "manual testing", "performance testing", "load testing",
    "unit testing", "integration testing", "e2e testing", "regression testing",
    "selenium", "cypress", "playwright", "puppeteer", "appium",
    "jest", "mocha", "jasmine", "vitest", "pytest", "junit", "testng",
    "test planning", "test strategy", "cucumber", "gherkin", "postman",
    "k6", "jmeter", "locust", "gatling", "sonarqube", "test management",
    "code coverage", "static analysis", "code quality", "security testing", "api testing", "mobile testing",
    "test cases", "robot framework", "testcafe",
    "qa automation", "end-to-end testing", "chai", "py test", "j unit", "sonar", "stress testing", "functional testing",
    "karma", "unittest", "mockito", "xunit",
    "rspec", "nunit",
}

# 16. Software Architecture & Methodologies
CAT_ARCHITECTURE: set[str] = {
    "software architecture", "system design", "solution architecture",
    "enterprise architecture", "technical architecture",
    "design patterns", "oop", "functional programming", "solid", "dry",
    "agile", "scrum", "kanban", "waterfall", "safe", "lean",
    "tdd", "bdd", "pair programming", "code review",
    "sdlc", "version control", "branching strategies", "gitflow", "trunk-based development",
    "git", "github", "gitlab", "bitbucket", "svn",
    "jira", "confluence", "trello", "asana", "notion",
    "uml", "ooad", "object-oriented design", "domain-driven design",
    "performance tuning", "optimization", "scalability", "high availability", "caching", "debugging", "troubleshooting",
    "software development", "software engineering", "cross-platform", "six sigma", "clean code", "source control", "subversion", "object oriented programming", "object-oriented programming", "domain driven design", "domain-driven design", "solid principles", "test driven development", "behavior driven development", "trunk based development", "root cause analysis", "analytical skills", "analytical thinking", "critical thinking", "logical thinking", "problem solving", "problem-solving", "creative problem solving", "decision making", "decision-making", "strategic thinking", "business acumen", "object oriented design", "object oriented analysis and design", "object-oriented analysis and design", "unified modeling language", "systems design", "systems integration", "system integration",
    "web development",
}

# 17. Enterprise Platforms (ERP/CRM)
CAT_ENTERPRISE: set[str] = {
    "salesforce", "sap", "oracle", "dynamics 365", "servicenow",
    "workday", "netsuite", "peoplesoft", "siebel",
    "crm", "erp", "cms", "wordpress", "drupal", "magento", "shopify",
    "sharepoint", "data management", # Often enterprise context
    "slack",
}

# 18. Certifications (General)
CAT_CERTIFICATIONS: set[str] = {
    "aws certified", "azure certified", "gcp certified",
    "pmp", "csm", "itil", "togaf",
    "cka", "ckad", "cks",
    "cisco certified", "ccna", "ccnp", "ccie",
    "hashicorp certified", "terraform certified", "aws solutions architect", "aws developer", "aws sysops", "aws devops engineer", "aws data engineer", "aws machine learning specialty",
    "azure administrator", "azure developer", "azure solutions architect", "azure devops engineer", "azure data engineer",
    "az-900", "az-104", "az-204", "az-305", "az-400",
    "gcp professional", "gcp associate", "cloud practitioner",
    "aws certification", "project management professional", "certified scrum master", "scaled agile", "certified kubernetes administrator", "certified kubernetes application developer", "certified kubernetes security specialist", "aws cloud practitioner",
}

# 19. Other/Tools (Catch-all for IDEs, Editors, Soft Skills disguised as hard skills if any)
CAT_TOOLS_EDITORS: set[str] = {
    "visual studio code", "vscode", "vs code", "visual studio", "intellij", "intellij idea", "pycharm", "webstorm", "eclipse", "xcode", "android studio", "rider", "sublime text", "vim", "neovim", "emacs",
}
# Map this to "OS & Embedded" or a generic "Development Tools"? Let's put in Software Engineering or just generic "Development Tools"


# Combined mapping: canonical skill -> family name
SKILL_TO_FAMILY: dict[str, str] = {}

# Order matters slightly for overrides, but dictionary is singular key.
# We apply them in order.

_CATEGORIES = [
    (CAT_LANGUAGES, "Programming Languages"),
    (CAT_FRONTEND, "Frontend Development"),
    (CAT_BACKEND, "Backend Development"),
    (CAT_MOBILE_DESKTOP, "Mobile & Desktop"),
    (CAT_DATABASES, "Databases & Storage"),
    (CAT_DATA_ENGINEERING, "Data Engineering"),
    (CAT_DATA_SCIENCE, "Data Science & ML"),
    (CAT_GEN_AI, "Generative AI"),
    (CAT_ANALYTICS, "BI & Analytics"),
    (CAT_CLOUD, "Cloud Computing"),
    (CAT_DEVOPS, "DevOps & Containers"),
    (CAT_OS_HARDWARE, "OS & Embedded"),
    (CAT_NETWORKING, "Networking"),
    (CAT_SECURITY, "Security & Identity"),
    (CAT_QA, "Testing & QA"),
    (CAT_ARCHITECTURE, "Architecture & Methods"),
    (CAT_ENTERPRISE, "Enterprise Platforms"),
    (CAT_CERTIFICATIONS, "Certifications"),
    (CAT_TOOLS_EDITORS, "Tools & Editors"),
]

for skill_set, family_name in _CATEGORIES:
    for skill in skill_set:
        SKILL_TO_FAMILY[skill] = family_name


def get_skill_family(skill: str) -> str | None:
    """Get the family name for a canonical skill.
    
    Args:
        skill: The canonical skill name (lowercase).
        
    Returns:
        The family name (e.g., "Programming", "Security") or None if not found.
    """
    return SKILL_TO_FAMILY.get(skill.lower().strip())


def get_skill_families(skills: list[str]) -> dict[str, list[str]]:
    """Categorize a list of skills by their families.
    
    Args:
        skills: List of canonical skill names.
        
    Returns:
        Dictionary mapping family names to lists of skills in that family.
    """
    families: dict[str, list[str]] = {}
    for skill in skills:
        family = get_skill_family(skill)
        if family:
            if family not in families:
                families[family] = []
            families[family].append(skill)
    return families


def format_skills_by_family(skills: list[str]) -> str:
    """Format skills grouped by family as a string.
    
    Args:
        skills: List of canonical skill names.
        
    Returns:
        String like "Programming: python, java; Security: cissp, nist"
    """
    families = get_skill_families(skills)
    if not families:
        return ""
    
    parts = []
    # Sort families for consistent output
    for family in sorted(families.keys()):
        family_skills = sorted(families[family])
        parts.append(f"{family}: {', '.join(family_skills)}")
    
    return "; ".join(parts)


# =============================================================================
# SOFTSKILLS: Interpersonal/behavioral traits
# =============================================================================

SOFTSKILL_VARIANTS: dict[str, str] = {
    # Communication
    "communication": "communication",
    "communication skills": "communication",
    "verbal communication": "communication",
    "written communication": "communication",
    "oral communication": "communication",
    "presentation skills": "presentation skills",
    "public speaking": "public speaking",
    
    # Collaboration
    "collaboration": "collaboration",
    "collaborative": "collaboration",
    "teamwork": "teamwork",
    "team work": "teamwork",
    "team player": "teamwork",
    "cross-functional": "cross-functional collaboration",
    "cross functional": "cross-functional collaboration",
    
    # Leadership
    "leadership": "leadership",
    "mentoring": "mentoring",
    "mentorship": "mentoring",
    "coaching": "coaching",
    "team management": "team management",
    "people management": "people management",
    "project management": "project management",
    "stakeholder management": "stakeholder management",
    
    # Problem solving
    "problem solving": "problem solving",
    "problem-solving": "problem solving",
    "analytical skills": "analytical skills",
    "analytical thinking": "analytical skills",
    "critical thinking": "critical thinking",
    "logical thinking": "logical thinking",
    "troubleshooting": "troubleshooting",
    "root cause analysis": "root cause analysis",
    
    # Personal traits
    "attention to detail": "attention to detail",
    "detail-oriented": "attention to detail",
    "detail oriented": "attention to detail",
    "self-motivated": "self-motivated",
    "self motivated": "self-motivated",
    "proactive": "proactive",
    "initiative": "initiative",
    "takes initiative": "initiative",
    "independence": "independent",
    "independent": "independent",
    "self-starter": "self-starter",
    "self starter": "self-starter",
    "autonomous": "autonomous",
    "autonomy": "autonomous",
    
    # Adaptability
    "adaptability": "adaptability",
    "adaptable": "adaptability",
    "flexibility": "flexibility",
    "flexible": "flexibility",
    "fast learner": "fast learner",
    "quick learner": "fast learner",
    "continuous learning": "continuous learning",
    "growth mindset": "growth mindset",
    
    # Organization
    "time management": "time management",
    "organization": "organization",
    "organizational skills": "organization",
    "multitasking": "multitasking",
    "multi-tasking": "multitasking",
    "prioritization": "prioritization",
    "deadline-driven": "deadline-driven",
    
    # Creativity
    "creativity": "creativity",
    "creative": "creativity",
    "innovation": "innovation",
    "innovative": "innovation",
    "creative problem solving": "creative problem solving",
    
    # Interpersonal
    "interpersonal skills": "interpersonal skills",
    "relationship building": "relationship building",
    "customer service": "customer service",
    "client-facing": "client-facing",
    "client facing": "client-facing",
    "customer-focused": "customer-focused",
    "customer focused": "customer-focused",
    "empathy": "empathy",
    "emotional intelligence": "emotional intelligence",
    "negotiation": "negotiation",
    "conflict resolution": "conflict resolution",
    
    # Work ethic
    "work ethic": "work ethic",
    "accountability": "accountability",
    "responsibility": "responsibility",
    "ownership": "ownership",
    "reliability": "reliability",
    "dependability": "reliability",
    "integrity": "integrity",
    "professionalism": "professionalism",
    
    # Other
    "patience": "patience",
    "persistence": "persistence",
    "resilience": "resilience",
    "positive attitude": "positive attitude",
    "enthusiasm": "enthusiasm",
    "passion": "passion",
    "curiosity": "curiosity",
    "decision making": "decision making",
    "decision-making": "decision making",
    "strategic thinking": "strategic thinking",
    "business acumen": "business acumen",
}

SOFTSKILLS: set[str] = set(SOFTSKILL_VARIANTS.values())


def get_all_hardskill_patterns() -> list[str]:
    """Return all hardskill patterns sorted by length (longest first).
    
    Sorting by length ensures longer patterns are matched first,
    preventing partial matches (e.g., 'java' matching before 'javascript').
    """
    return sorted(HARDSKILL_VARIANTS.keys(), key=len, reverse=True)


def get_all_softskill_patterns() -> list[str]:
    """Return all softskill patterns sorted by length (longest first)."""
    return sorted(SOFTSKILL_VARIANTS.keys(), key=len, reverse=True)
