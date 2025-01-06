class temp_cache():
    iteration = 0
    ori_capa = 64
    cur_capa = 64
    update_freq = 10
    min_kv_capacity = 32
    decrease_rate = 0.8
    decoding_latency = 0
    max_GPU_memory = 0
    min_GPU_memory = 40960
    max_KV_cache_memory = 0
    min_KV_cache_memory = 40960
