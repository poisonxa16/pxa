# Local development override - automatically sets BUILD_NUMBER and BUILD_COMMIT
variable "BUILD_NUMBER" { default = "0" }
variable "BUILD_COMMIT" { default = "local-dev" }
variable "CUDA_VERSION" { default = "12.6.2" }

target "server" {
  inherits = ["settings"]
  dockerfile = "${VARIANT == "cpu" ? "./docker/pxa-cpu.Containerfile" : "./docker/pxa-cuda.Containerfile"}"
}

target "swap" {
  inherits = ["settings"]
  dockerfile = "${VARIANT == "cpu" ? "./docker/pxa-cpu.Containerfile" : "./docker/pxa-cuda.Containerfile"}"
}

target "full" {
  inherits = ["settings"]
  dockerfile = "${VARIANT == "cpu" ? "./docker/pxa-cpu.Containerfile" : "./docker/pxa-cuda.Containerfile"}"
}
