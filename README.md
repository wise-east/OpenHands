<a name="readme-top"></a>

<div align="center">
  <img src="https://raw.githubusercontent.com/OpenHands/docs/main/openhands/static/img/logo.png" alt="Logo" width="200">
  <h1 align="center" style="border-bottom: none">OpenHands: AI-Driven Development</h1>
</div>


<div align="center">
  <a href="https://github.com/OpenHands/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-20B2AA?style=for-the-badge" alt="MIT License"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=811504672#gid=811504672"><img src="https://img.shields.io/badge/SWEBench-77.6-00cc00?logoColor=FFE165&style=for-the-badge" alt="Benchmark Score"></a>
  <br/>
  <a href="https://docs.openhands.dev/sdk"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2511.03690"><img src="https://img.shields.io/badge/Paper-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Tech Report"></a>


  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=fr">français</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=zh">中文</a>

</div>

Welcome to OpenHands (formerly OpenDevin), a platform for software development agents powered by AI.

OpenHands agents can do anything a human developer can: modify code, run commands, browse the web,
call APIs, and yes—even copy code snippets from StackOverflow.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev), or [sign up for OpenHands Cloud](https://app.all-hands.dev) to get started.


> [!IMPORTANT]
> **Upcoming change**: We are renaming our GitHub Org from `All-Hands-AI` to `OpenHands` on October 20th, 2025.
> Check the [tracking issue](https://github.com/All-Hands-AI/OpenHands/issues/11376) for more information.


> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.

## ☁️ OpenHands Cloud
The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev),
which comes with $10 in free credits for new users.

## 💻 Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The easiest way to run OpenHands locally is using the CLI launcher with [uv](https://docs.astral.sh/uv/). This provides better isolation from your current project's virtual environment and is required for OpenHands' default MCP servers.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

**Launch OpenHands**:
```bash
# Launch the GUI server
uvx --python 3.12 openhands serve

# Or launch the CLI
uvx --python 3.12 openhands
```

You'll find OpenHands running at [http://localhost:3000](http://localhost:3000) (for GUI mode)!

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

You can also run OpenHands directly with Docker:

```bash
docker pull docker.openhands.dev/openhands/runtime:0.62-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.openhands.dev/openhands/runtime:0.62-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.openhands.dev/openhands/openhands:0.62
```

</details>

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)
> to secure your deployment by restricting network binding and implementing additional security measures.

### Getting Started

When you open the application, you'll be asked to choose an LLM provider and add an API key.
[Anthropic's Claude Sonnet 4.5](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-5-20250929`)
works best, but you have [many options](https://docs.all-hands.dev/usage/llms).

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for
system requirements and more information.

## 💡 Other ways to run OpenHands

> [!WARNING]
> OpenHands is meant to be run by a single user on their local workstation.
> It is not appropriate for multi-tenant deployments where multiple users share the same instance. There is no built-in authentication, isolation, or scalability.
>
> If you're interested in running OpenHands in a multi-tenant environment, check out the source-available, commercially-licensed
> [OpenHands Cloud Helm Chart](https://github.com/openHands/OpenHands-cloud)

🙌 Welcome to OpenHands, a [community](COMMUNITY.md) focused on AI-driven development. We’d love for you to [join us on Slack](https://dub.sh/openhands).

There are a few ways to work with OpenHands:

### OpenHands Software Agent SDK
The SDK is a composable Python library that contains all of our agentic tech. It's the engine that powers everything else below.

Define agents in code, then run them locally, or scale to 1000s of agents in the cloud.

[Check out the docs](https://docs.openhands.dev/sdk) or [view the source](https://github.com/OpenHands/software-agent-sdk/)

### OpenHands CLI
The CLI is the easiest way to start using OpenHands. The experience will be familiar to anyone who has worked
with e.g. Claude Code or Codex. You can power it with Claude, GPT, or any other LLM.

[Check out the docs](https://docs.openhands.dev/openhands/usage/run-openhands/cli-mode) or [view the source](https://github.com/OpenHands/OpenHands-CLI)

### OpenHands Local GUI
Use the Local GUI for running agents on your laptop. It comes with a REST API and a single-page React application.
The experience will be familiar to anyone who has used Devin or Jules.

[Check out the docs](https://docs.openhands.dev/openhands/usage/run-openhands/local-setup) or view the source in this repo.

### OpenHands Cloud
This is a deployment of OpenHands GUI, running on hosted infrastructure.

You can try it with a free $10 credit by [signing in with your GitHub or GitLab account](https://app.all-hands.dev).

OpenHands Cloud comes with source-available features and integrations:
- Integrations with Slack, Jira, and Linear
- Multi-user support
- RBAC and permissions
- Collaboration features (e.g., conversation sharing)

### OpenHands Enterprise
Large enterprises can work with us to self-host OpenHands Cloud in their own VPC, via Kubernetes.
OpenHands Enterprise can also work with the CLI and SDK above.

OpenHands Enterprise is source-available--you can see all the source code here in the enterprise/ directory,
but you'll need to purchase a license if you want to run it for more than one month.

Enterprise contracts also come with extended support and access to our research team.

Learn more at [openhands.dev/enterprise](https://openhands.dev/enterprise)

## 🧪 Running SWE-bench Evaluation

The repo includes `run_swebench_eval.py`, a self-contained script that generates patches with an OpenHands agent and evaluates them against [SWE-bench Verified](https://www.swebench.com/).

### Environment Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# Create a virtual environment
uv venv .venv --python 3.12

# Install all dependencies (core + runtimes + eval)
uv pip install --link-mode copy --python .venv/bin/python -e ".[third_party_runtimes,eval]"
```

### Required Environment Variables

Set these before running:

```bash
export OPENAI_API_KEY="..."          # or whichever LLM provider you use
export MODAL_TOKEN_ID="..."          # for Modal runtime
export MODAL_TOKEN_SECRET="..."
export DAYTONA_API_KEY="..."         # for Daytona runtime (if using --runtime daytona)
export DAYTONA_SERVER_URL="..."
export DAYTONA_TARGET="..."
```

### Usage

```bash
# Run evaluation on 50 instances (default) using Modal
.venv/bin/python run_swebench_eval.py -n 50

# Re-evaluate existing patches
.venv/bin/python run_swebench_eval.py eval --input-dir trajectories/<run_id>
```

Results (patches, trajectories, eval output) are saved to `trajectories/<timestamp>/`.

### Everything Else

Check out our [Product Roadmap](https://github.com/orgs/openhands/projects/1), and feel free to
[open up an issue](https://github.com/OpenHands/OpenHands/issues) if there's something you'd like to see!

You might also be interested in our [evaluation infrastructure](https://github.com/OpenHands/benchmarks), our [chrome extension](https://github.com/OpenHands/openhands-chrome-extension/), or our [Theory-of-Mind module](https://github.com/OpenHands/ToM-SWE).

All our work is available under the MIT license, except for the `enterprise/` directory in this repository (see the [enterprise license](enterprise/LICENSE) for details).
The core `openhands` and `agent-server` Docker images are fully MIT-licensed as well.

If you need help with anything, or just want to chat, [come find us on Slack](https://dub.sh/openhands).
