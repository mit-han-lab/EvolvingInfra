#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
# Build the runtime image from one committed revision and push it to Docker Hub.
#
#   bash docker/build_and_push.sh                      # origin/main
#   SOURCE_SHA=<commit or ref> bash docker/build_and_push.sh
#   CHECK_ONLY=1 bash docker/build_and_push.sh         # build context and credential scan only
#
# The context is `git archive` of the commit, so local changes, venvs and build outputs never enter
# the image. The tag is <sha12>-<arch>; deployments should pin the printed digest, not the tag.
# Build each architecture on a host of that architecture (amd64: H200/B300; arm64: GB300).
# Needs Docker and a `docker login` that can push to IMAGE_REPO.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_REPO="${IMAGE_REPO:-docker.io/weili0234/evolving-infra}"
SOURCE_SHA=$(git -C "$REPO" rev-parse --verify "${SOURCE_SHA:-origin/main}^{commit}")
case "$(uname -m)" in x86_64) ARCH=amd64 ;; aarch64|arm64) ARCH=arm64 ;; *) echo "unsupported arch" >&2; exit 2 ;; esac
TAG="${SOURCE_SHA:0:12}-${ARCH}"

work=$(mktemp -d "${TMPDIR:-/tmp}/evolving-infra-image.XXXXXX")
trap 'rm -rf "$work"' EXIT
git -C "$REPO" archive --format=tar "$SOURCE_SHA" | tar -x -C "$work"

# Refuse to build a context that contains credentials.
pattern='(hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{32,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{10,}|(ak|as)-[A-Za-z0-9]{20,})'
if hits=$(grep -rIlE "$pattern" "$work" 2>/dev/null); then
  echo "possible credentials in the build context:" >&2
  echo "${hits//$work\//}" >&2
  exit 3
fi
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  echo "context for $SOURCE_SHA passed the credential scan; not building (CHECK_ONLY=1)"
  exit 0
fi

# The ai-dynamo wheel needs AVX2. On an older build host, drop only its build-time import check;
# the image is the same, and the check must then run on a deployment host.
checks=full
if [[ "$ARCH" == amd64 ]] && ! grep -qw avx2 /proc/cpuinfo; then
  sed -i "s/python -c 'import dynamo.llm, /python -c 'import /" "$work/docker/Dockerfile"
  grep -q "import dynamo.llm" "$work/docker/Dockerfile" && { echo "could not drop the dynamo check" >&2; exit 4; }
  checks=no-dynamo-import
  echo "build host has no AVX2: skipping the build-time 'import dynamo.llm' check" >&2
fi

docker build -f "$work/docker/Dockerfile" \
  --label org.opencontainers.image.source=https://github.com/mit-han-lab/EvolvingInfra \
  --label evolving-infra.build-checks="$checks" \
  --label org.opencontainers.image.revision="$SOURCE_SHA" \
  -t "$IMAGE_REPO:$TAG" "$work"
docker push "$IMAGE_REPO:$TAG"
digest=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE_REPO:$TAG" \
  | grep -F "${IMAGE_REPO#docker.io/}@" | head -1)
echo "source $SOURCE_SHA"
echo "image  $IMAGE_REPO:$TAG"
echo "digest ${digest:-unknown}"
