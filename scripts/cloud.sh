#!/usr/bin/env bash
# Run the benchmark from an EC2 instance, so results don't depend on where
# your laptop is.
#
#   scripts/cloud.sh launch            create the instance and install the benchmark
#   scripts/cloud.sh push              copy .env, audio samples and results.db up
#   scripts/cloud.sh bench <args...>   run `stt-benchmark <args...>` on the instance
#   scripts/cloud.sh bench --stop ...  the same, and stop the instance when it finishes
#   scripts/cloud.sh pull              copy results.db back (the local one is backed up)
#   scripts/cloud.sh ssh [command]     open a shell, or run one command
#   scripts/cloud.sh status | stop | start | terminate
#
# Settings, all optional:
#   STT_CLOUD_REGION   AWS region              (default us-west-2)
#   STT_CLOUD_TYPE     instance type           (default c7i.4xlarge: 16 dedicated vCPUs)
#   STT_CLOUD_NAME     instance and key name   (default stt-benchmark)
#   STT_CLOUD_KEY      SSH public key          (default ~/.ssh/id_ed25519.pub)
#   STT_CLOUD_BRANCH   branch to check out     (default main)
#
# The instance stops itself after 30 minutes with no benchmark running and
# nobody connected, so a forgotten instance costs only its disk.
#
# Needs the AWS CLI, signed in to an account that may create EC2 instances.

set -euo pipefail

REGION="${STT_CLOUD_REGION:-us-west-2}"
TYPE="${STT_CLOUD_TYPE:-c7i.4xlarge}"
NAME="${STT_CLOUD_NAME:-stt-benchmark}"
KEY="${STT_CLOUD_KEY:-$HOME/.ssh/id_ed25519.pub}"
BRANCH="${STT_CLOUD_BRANCH:-main}"
REPO_URL="https://github.com/pipecat-ai/stt-benchmark.git"
REMOTE_DIR="stt-benchmark"
IDLE_MINUTES=30
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)

cd "$(dirname "$0")/.."

aws_ec2() { aws ec2 --region "$REGION" "$@"; }

# The instance's id and state, or nothing when there is none.
instance() {
    aws_ec2 describe-instances \
        --filters "Name=tag:Name,Values=$NAME" \
        "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' \
        --output text | head -1
}

instance_id() { instance | cut -f1; }

host() {
    local ip
    ip="$(instance | cut -f3)"
    if [[ -z "$ip" || "$ip" == "None" ]]; then
        echo "No running instance named $NAME in $REGION. Try: scripts/cloud.sh start" >&2
        exit 1
    fi
    echo "ubuntu@$ip"
}

remote() { ssh "${SSH_OPTS[@]}" "$(host)" "$@"; }

my_ip() { curl -fsS https://checkip.amazonaws.com; }

# Let this machine, and only this machine, reach the instance over SSH.
allow_my_ip() {
    local group="$1" ip
    ip="$(my_ip)"
    aws_ec2 authorize-security-group-ingress --group-id "$group" \
        --protocol tcp --port 22 --cidr "$ip/32" >/dev/null 2>&1 || true
}

security_group() {
    local group
    group="$(aws_ec2 describe-security-groups --filters "Name=group-name,Values=$NAME" \
        --query 'SecurityGroups[0].GroupId' --output text)"
    if [[ "$group" == "None" ]]; then
        group="$(aws_ec2 create-security-group --group-name "$NAME" \
            --description "SSH to the STT benchmark instance" \
            --query GroupId --output text)"
    fi
    allow_my_ip "$group"
    echo "$group"
}

cmd_launch() {
    if [[ -n "$(instance)" ]]; then
        echo "An instance named $NAME already exists in $REGION:"
        cmd_status
        exit 1
    fi
    if ! aws_ec2 describe-key-pairs --key-names "$NAME" >/dev/null 2>&1; then
        aws_ec2 import-key-pair --key-name "$NAME" --public-key-material "fileb://$KEY" >/dev/null
    fi
    local group ami id
    group="$(security_group)"
    ami="$(aws ssm get-parameter --region "$REGION" --query Parameter.Value --output text \
        --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id)"
    id="$(aws_ec2 run-instances --image-id "$ami" --instance-type "$TYPE" \
        --key-name "$NAME" --security-group-ids "$group" \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
        --query 'Instances[0].InstanceId' --output text)"
    echo "Launched $id ($TYPE in $REGION). Waiting for it to start..."
    aws_ec2 wait instance-status-ok --instance-ids "$id"
    cmd_setup
    echo
    echo "Ready. Next: scripts/cloud.sh push"
}

cmd_setup() {
    remote "bash -s" <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d $REMOTE_DIR ] || git clone -q $REPO_URL $REMOTE_DIR
cd $REMOTE_DIR
git fetch -q origin && git checkout -q $BRANCH && git pull -q --ff-only
uv sync -q
mkdir -p stt_benchmark_data

# Stop the instance when it has been idle for $IDLE_MINUTES minutes: no
# benchmark running and no SSH connection. A shutdown from inside stops an
# EC2 instance and keeps its disk.
sudo tee /usr/local/bin/stt-idle-stop >/dev/null <<'IDLE'
#!/bin/sh
stamp=/run/stt-last-active
[ -f "\$stamp" ] || touch "\$stamp"
if pgrep -f '^sshd: ubuntu' >/dev/null || pgrep -f '\.venv/bin/stt-benchmark' >/dev/null; then
    touch "\$stamp"
    exit 0
fi
age=\$(( \$(date +%s) - \$(stat -c %Y "\$stamp") ))
[ "\$age" -ge $((IDLE_MINUTES * 60)) ] && shutdown -h now
exit 0
IDLE
sudo chmod +x /usr/local/bin/stt-idle-stop
echo '* * * * * root /usr/local/bin/stt-idle-stop' | sudo tee /etc/cron.d/stt-idle-stop >/dev/null
echo "Installed: \$(git log --oneline -1)"
EOF
}

cmd_push() {
    local target
    target="$(host)"
    scp "${SSH_OPTS[@]}" -q .env "$target:$REMOTE_DIR/.env"
    echo "Copying the audio samples and results.db (about 3 GB the first time)..."
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
        stt_benchmark_data/audio stt_benchmark_data/results.db \
        "$target:$REMOTE_DIR/stt_benchmark_data/"
    echo "Copied .env, the audio samples and results.db."
}

cmd_pull() {
    local target backup
    target="$(host)"
    backup="stt_benchmark_data/results.before-pull.$(date +%Y%m%d_%H%M%S).db"
    cp stt_benchmark_data/results.db "$backup"
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
        "$target:$REMOTE_DIR/stt_benchmark_data/results.db" stt_benchmark_data/results.db
    echo "Copied results.db from the instance. The previous local copy is $backup"
}

# Run in tmux, so the run carries on if the connection drops; running the
# same command again reattaches. The instance may not know your terminal type.
cmd_bench() {
    local args after="echo; echo '[done: press enter]'; read"
    if [[ "${1:-}" == "--stop" ]]; then
        shift
        after="echo; echo '[done: stopping the instance in one minute]'; sudo shutdown -h +1"
    fi
    args="$(printf '%q ' "$@")"
    ssh -t "${SSH_OPTS[@]}" "$(host)" \
        "cd $REMOTE_DIR && git pull -q --ff-only && export PATH=\$HOME/.local/bin:\$PATH && \
         { infocmp >/dev/null 2>&1 || export TERM=xterm-256color; } && \
         tmux new-session -A -s bench \"uv run stt-benchmark $args; $after\""
}

cmd_ssh() {
    if [[ $# -eq 0 ]]; then
        ssh "${SSH_OPTS[@]}" "$(host)"
    else
        remote "$@"
    fi
}

cmd_status() {
    local line
    line="$(instance)"
    if [[ -z "$line" ]]; then
        echo "No instance named $NAME in $REGION."
    else
        echo "$line" | awk -v r="$REGION" '{print $1 "  " $2 "  " $3 "  (" r ")"}'
    fi
}

cmd_stop() {
    aws_ec2 stop-instances --instance-ids "$(instance_id)" >/dev/null
    echo "Stopping. The disk is kept; scripts/cloud.sh start brings it back."
}

cmd_start() {
    local id
    id="$(instance_id)"
    aws_ec2 start-instances --instance-ids "$id" >/dev/null
    aws_ec2 wait instance-status-ok --instance-ids "$id"
    # Your address may have changed since the instance was launched.
    security_group >/dev/null
    cmd_status
}

cmd_terminate() {
    local id
    id="$(instance_id)"
    read -r -p "Delete instance $id and everything on its disk? Type 'yes': " answer
    [[ "$answer" == "yes" ]] || exit 1
    aws_ec2 terminate-instances --instance-ids "$id" >/dev/null
    echo "Terminating."
}

command="${1:-}"
shift || true
case "$command" in
    launch | setup | push | pull | bench | ssh | status | stop | start | terminate) "cmd_$command" "$@" ;;
    *)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
