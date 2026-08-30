from config import *
from model.multi_agents import AugmentedDebateSystem
from flashrag.config import Config
from flashrag.utils import get_dataset


def TRIO(cfg, test_data):
    pipeline = AugmentedDebateSystem(cfg, 
                                  max_query_debate_rounds=cfg["max_query_debate_rounds"])
    result = pipeline.run(test_data, batch_size=500)


    return result

def main(cfg):
    all_splits = get_dataset(cfg)
    print(f"Loaded dataset splits: {list(all_splits.keys())}")
    test_data = all_splits["dev"]
    
    func_map = {
        "TRIO": TRIO,
    }
    
    func = func_map[cfg["method_name"]]
    func(cfg, test_data)

    

if __name__ == "__main__":
    
    cfg = init_cfg()

    main(cfg)

"""

conda activate drag &&python main.py --method_name "TRIO" \
               --gpu_id "0,1" \
               --dataset_name "musique" \
               --generator_model "config_llama3-8B-Instruct" \
               --max_query_debate_rounds 3 
"""