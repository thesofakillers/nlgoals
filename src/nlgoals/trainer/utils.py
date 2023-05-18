import enum

class Accelerator(str, enum.Enum):
    cpu = "cpu"
    gpu = "gpu"
    tpu = "tpu"
    ipu = "ipu"
    auto = "auto"
