# vLLM (Internal Mirror)

This repository is an internal mirror of [vLLM](https://github.com/vllm-project/vllm), adapted for our company's environment.

## Purpose

- **Internal Customization:** Adjust vLLM for our infrastructure, e.g., modify `Dockerfile` for our cluster.  
- **Experimental Features:** Add prototypes and experimental changes not in the official vLLM.

## Usage

Use the prebuilt Docker images hosted on GitHub Container Registry:

```bash
# Choose the image according to your environment (CUDA or ROCm)
docker pull ghcr.io/MotifTechnologies/vllm:cuda-latest  # for CUDA
docker pull ghcr.io/MotifTechnologies/vllm:rocm-latest  # for ROCm

# Run the container
docker run --rm -it ghcr.io/MotifTechnologies/vllm:<tag>
```

- For a full list of available images and tags, see [GHCR](https://github.com/MotifTechnologies/vllm/pkgs/container/vllm).

## Examples

### Skypilot multinode DP with external load balancing
```yaml
...
envs:
  PROXY_PORT: 8000
  MY_VLLM_PORT: 8080 # Can't use VLLM_PORT due to conflict with vLLM's internal usage

run: |
  if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    pkill haproxy
    python /app/vllm/examples/proxy/generate_haproxy_cfg.py \
      --proxy-port $PROXY_PORT \
      --vllm-port $MY_VLLM_PORT \
      --nodes "$SKYPILOT_NODE_IPS" \ # automatically provided by SkyPilot
      --output haproxy.cfg
    haproxy -f haproxy.cfg
  fi
  # Run a vLLM server on each node as a single-node instance
  vllm serve --port $MY_VLLM_PORT ...
```
