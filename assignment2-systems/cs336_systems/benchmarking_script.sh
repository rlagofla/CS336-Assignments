uv run --frozen \
    nsys profile -o out/profile_2dot7B --pytorch=autograd-shapes-nvtx \
    python -m cs336_systems.benchmarking_script \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32