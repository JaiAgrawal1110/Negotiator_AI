#!/bin/bash
set -e

echo "=== NegotiateAI EC2 setup starting ==="

# 1. Update system packages
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    git

# 3. Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 4. Add Docker's apt repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Install Docker Engine + Docker Compose plugin
sudo apt-get update -y
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# 6. Let current user run docker without sudo
sudo usermod -aG docker $USER

# 7. Enable Docker to start on boot
sudo systemctl enable docker
sudo systemctl start docker

echo "=== Docker installed ==="
docker --version
docker compose version

echo "=== Setup complete ==="
echo "Run 'newgrp docker' (or log out/in) so this shell picks up docker group access."
echo "Next: clone your repo and run 'docker compose up -d --build'"
