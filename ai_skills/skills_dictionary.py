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
    ".net framework": "dotnet",
    "dotnet": "dotnet",
    "dot net": "dotnet",
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
    "rest": "rest",
    "restful": "rest",
    "restful api": "restful api",
    "restful apis": "restful api",
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
    "css3": "css3",
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
    "html5": "html5",
    
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
    "rabbitmq": "rabbitmq",
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
    "oauth": "oauth",
    "oauth2": "oauth2",
    "jwt": "jwt",
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
    "api": "api",
    "apis": "api",
    "sdk": "sdk",
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
    "troubleshooting": "troubleshooting",
    "performance tuning": "performance tuning",
    "optimization": "optimization",
    "scalability": "scalability",
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
    "frontend development": "frontend development",
    "front-end development": "frontend development",
    "front end development": "frontend development",
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
    "s3": "s3",
    "amazon s3": "s3",
    "blob storage": "blob storage",
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
}

# =============================================================================
# CANONICAL HARDSKILLS: The set of all normalized skill names (for lookup)
# =============================================================================

HARDSKILLS: set[str] = set(HARDSKILL_VARIANTS.values())

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


def normalize_hardskill(skill: str) -> str | None:
    """Normalize a hardskill to its canonical form."""
    skill_lower = skill.lower().strip()
    return HARDSKILL_VARIANTS.get(skill_lower)


def normalize_softskill(skill: str) -> str | None:
    """Normalize a softskill to its canonical form."""
    skill_lower = skill.lower().strip()
    return SOFTSKILL_VARIANTS.get(skill_lower)
