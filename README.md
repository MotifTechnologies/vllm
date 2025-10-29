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

For a full list of available images and tags, see [here](https://github.com/MotifTechnologies/vllm/pkgs/container/vllm).


## Examples

### multinode DP with external load balancing
```yaml
...
envs:
  PROXY_PORT: 8000
  VLLM_PORT: 8080
run: |
  if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    pkill haproxy
    python /app/vllm/examples/proxy/generate_haproxy_cfg.py \
      --proxy-port $PROXY_PORT \
      --vllm-port $VLLM_PORT \
      --nodes "$SKYPILOT_NODE_IPS" \
      --output haproxy.cfg
    haproxy -f haproxy.cfg
  fi
  vllm serve --port $VLLM_PORT ...
```
