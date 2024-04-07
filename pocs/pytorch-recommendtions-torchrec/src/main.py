import os
import torch
import torchrec
import torch.distributed as dist

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# Note - you will need a V100 or A100 to run tutorial as as!
# If using an older GPU (such as colab free K80),
# you will need to compile fbgemm with the appripriate CUDA architecture
# or run with "gloo" on CPUs
dist.init_process_group(backend="nccl")

ebc = torchrec.EmbeddingBagCollection(
    device="meta",
    tables=[
        torchrec.EmbeddingBagConfig(
            name="product_table",
            embedding_dim=64,
            num_embeddings=4096,
            feature_names=["product"],
            pooling=torchrec.PoolingType.SUM,
        ),
        torchrec.EmbeddingBagConfig(
            name="user_table",
            embedding_dim=64,
            num_embeddings=4096,
            feature_names=["user"],
            pooling=torchrec.PoolingType.SUM,
        )
    ]
)

model = torchrec.distributed.DistributedModelParallel(ebc, device=torch.device("cuda"))
print(model)
print(model.plan)

product_eb = torch.nn.EmbeddingBag(4096, 64)
product_eb(input=torch.tensor([101, 202, 303]), offsets=torch.tensor([0, 2, 2]))

mb = torchrec.KeyedJaggedTensor(
    keys = ["product", "user"],
    values = torch.tensor([101, 202, 303, 404, 505, 606]).cuda(),
    lengths = torch.tensor([2, 0, 1, 1, 1, 1], dtype=torch.int64).cuda(),
)

print(mb.to(torch.device("cpu")))

pooled_embeddings = model(mb)
print(pooled_embeddings)