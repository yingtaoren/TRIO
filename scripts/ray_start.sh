# # # source /opt/aps/workdir/input/jiechen/.venv/bin/activate
export CUDA_VISIBLE_DEVICES="0,1"
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate my_rag

ray stop
# 主节点启动
HEAD_NODE_IP="138.25.54.36"  # H100
ray start --head --node-ip-address ${HEAD_NODE_IP} --num-gpus 2 --port 8266 --dashboard-port 8267


