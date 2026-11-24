# Running From the Cloud

TTFS includes the network path from the machine running the benchmark to each STT service, so the same service measures differently from a laptop on home WiFi, a hotel connection, and a data center. Running from one cloud instance in a stated region keeps results comparable with each other and repeatable by someone else.

`scripts/cloud.sh` sets up an AWS EC2 instance and runs the benchmark on it. A run in the cloud is the usual `stt-benchmark` command with `scripts/cloud.sh bench` in front.

## What You Need

- The [AWS CLI](https://docs.aws.amazon.com/cli/), signed in to an account that may create EC2 instances (`aws sso login`, `aws login` or `aws configure`).
- An SSH key at `~/.ssh/id_ed25519.pub`, or another one named in `STT_CLOUD_KEY`.
- A working local setup: `.env` with your API keys, and `stt_benchmark_data/` with the audio samples and `results.db`.

## Setting Up

```bash
# Create the instance and install the benchmark on it (3-4 minutes)
scripts/cloud.sh launch

# Copy .env, the audio samples and results.db to it (about 3 GB the first time)
scripts/cloud.sh push
```

`launch` creates:

- one `c7i.4xlarge` instance (16 dedicated vCPUs) named `stt-benchmark`, in `us-west-2`
- a 50 GB disk
- a security group that allows SSH from your current IP address only
- a copy of your SSH public key

`push` copies your `.env`, and so your API keys, onto the instance's disk.

## Running

```bash
# Any stt-benchmark command, run on the instance
scripts/cloud.sh bench run --services deepgram -c 12
scripts/cloud.sh bench wer --services deepgram
scripts/cloud.sh bench report

# Stop the instance as soon as the run finishes
scripts/cloud.sh bench --stop run --services deepgram -c 12
```

`bench` first updates the instance to the latest commit of its branch, then runs the command inside `tmux`. If your connection drops the run carries on, and running `bench` again reattaches to it.

When you want the results locally, to regenerate the README table and charts:

```bash
scripts/cloud.sh pull
```

`pull` replaces your local `results.db` with the instance's, after saving the local one as `results.before-pull.<timestamp>.db`. Treat the instance's database as the one measurements are written to: runs made locally into the main database after a `push` are overwritten by the next `pull`.

## Stopping and Cost

```bash
scripts/cloud.sh status
scripts/cloud.sh stop        # keeps the disk
scripts/cloud.sh start
scripts/cloud.sh terminate   # deletes the instance and its disk
```

The instance stops itself after 30 minutes with no benchmark running and nobody connected over SSH. A stopped instance keeps its disk, with the benchmark, the database and your keys on it, and `start` brings it back.

Approximate `us-west-2` on-demand prices:

| | Rate | Charged |
|---|---|---|
| Instance, `c7i.4xlarge` | about $0.71 an hour | while running |
| Disk, 50 GB | about $4 a month | until terminated |

A 1,000-sample run at `-c 12` takes about 20 minutes. What the STT services and the WER judge charge is separate and the same as running locally.

## Settings

All optional, as environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `STT_CLOUD_REGION` | AWS region | `us-west-2` |
| `STT_CLOUD_TYPE` | Instance type | `c7i.4xlarge` |
| `STT_CLOUD_NAME` | Name of the instance, security group and key | `stt-benchmark` |
| `STT_CLOUD_KEY` | SSH public key to install | `~/.ssh/id_ed25519.pub` |
| `STT_CLOUD_BRANCH` | Branch to check out on the instance | `main` |

Use an instance type with dedicated CPUs. Burstable types (the `t` family) throttle the CPU, which delays the VAD and distorts timing.

## Choosing a Region

No region is neutral: each is closer to some services than to others, and some services answer more slowly from one of their regions than from another. Pick the region your agents run in, state it with any results you publish, and don't compare results measured from different places.

Before trusting numbers from a new instance or region, compare a few dozen samples at `--concurrency 1` with the same samples at your chosen level; see [Concurrency](cli.md#concurrency).
