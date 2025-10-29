# Setup Scripts and One-Time Configuration

This directory contains setup scripts, templates, and configuration files that are used for one-time setup tasks but are not part of the deployed Lambda code.

## Directory Structure

```
setup/
├── prompt_management/          # Bedrock Prompt Management setup
│   ├── create_prompts.py      # Script to create prompts via API
│   ├── summary_prompt_template.json
│   ├── case_check_prompt_template.json
│   └── README.md              # Detailed setup guide
└── README.md                  # This file
```

## What Goes Here

- **One-time setup scripts** - Scripts that create AWS resources (prompts, parameters, etc.)
- **Configuration templates** - JSON/YAML templates for AWS resources
- **Migration scripts** - Scripts for data migration or infrastructure changes
- **Development tools** - Helper scripts for local development setup

## What Does NOT Go Here

- **Lambda code** - Application code goes in `summariser/`
- **Tests** - Test files go in `tests/` or `evals/`
- **Deployment config** - `template.yaml` stays at root level
- **Runtime dependencies** - Anything needed at Lambda runtime stays in `summariser/`

## Usage

These files are:
- ✅ Committed to git (for team collaboration and documentation)
- ✅ Used during initial setup or configuration changes
- ❌ NOT deployed to Lambda
- ❌ NOT included in SAM build artifacts

## Prompt Management Setup

See [prompt_management/README.md](prompt_management/README.md) for detailed instructions on setting up Bedrock Prompt Management.

**Quick start:**
```bash
cd setup/prompt_management
python create_prompts.py --create all
```

This will create prompts in AWS and output ARNs to add to `template.yaml`.
