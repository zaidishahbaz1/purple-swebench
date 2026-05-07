FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install Docker CLI so our agent can pull SWE-bench instance images and
# exec commands inside them to read actual source code before patching.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
    > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password agent
RUN groupadd -f docker && usermod -aG docker agent
RUN chmod 777 /var/run

USER agent
WORKDIR /home/agent

COPY --chown=agent pyproject.toml uv.lock README.md ./
COPY --chown=agent src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
