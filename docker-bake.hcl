group "cuda" {
  targets = ["vllm_cuda", "vllm_cuda_builder"]
}

group "rocm" {
  targets = ["vllm_rocm"]
}

target "vllm_cuda" {
  context = "."
  dockerfile = "docker/Dockerfile"
  tags = ["ghcr.io/motiftechnologies/vllm"]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
  args = {
    max_jobs = "48"
    torch_cuda_arch_list = "9.0;10.0"
  }
}

target "vllm_cuda_builder" {
  inherits = ["vllm_cuda"]
  target   = "build"
  tags = ["ghcr.io/motiftechnologies/vllm-builder"]
}

target "rocm_base" {
  context = "."
  dockerfile = "docker/Dockerfile.rocm_base_motif"
  tags = ["ghcr.io/motiftechnologies/vllm-base"]
  args = {
    max_jobs = "1"
  }
}

target "vllm_rocm" {
  context = "."
  dockerfile = "docker/Dockerfile.rocm"
  tags = ["ghcr.io/motiftechnologies/vllm"]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
  args = {
    max_jobs = "1"
  }
  contexts = {
    rocm_base = "target:rocm_base"
  }
}
