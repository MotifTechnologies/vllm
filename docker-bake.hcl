group "cuda" {
  targets = ["vllm_cuda"]
}

group "rocm" {
  targets = ["vllm_rocm"]
}

target "vllm_cuda" {
  context = "."
  dockerfile = "docker/Dockerfile"
  tags = ["ghcr.io/motiftechnologies/vllm:cuda-latest"]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
  args = {
    max_jobs = "64"
    torch_cuda_arch_list = "9.0;10.0"
  }
}

target "rocm_base" {
  context = "."
  dockerfile = "docker/Dockerfile.rocm_base_motif"
  tags = ["ghcr.io/motiftechnologies/vllm:rocm_base-latest"]
  args = {
    max_jobs = "1"
  }
}

target "vllm_rocm" {
  context = "."
  dockerfile = "docker/Dockerfile.rocm"
  tags = ["ghcr.io/motiftechnologies/vllm:rocm-latest"]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
  args = {
    max_jobs = "1"
  }
  contexts = {
    rocm_base = "target:rocm_base"
  }
}
