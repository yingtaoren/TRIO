import argparse
import json
import os

from flashrag.config import Config

def init_cfg():
    parser = argparse.ArgumentParser("Update configuration from command line arguments.")

    # 用于debug
    parser.add_argument(
        "--method_name",
        type=str,
        default="DRAG",
        help="Which method to run (default: DRAG)",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="CUDA GPU id (default: 0)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="2wiki",
        help="Dataset to use (default: 2wiki)",
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default="llama3-8B-instruct",
        help="Generator model name (default: llama3-8B-instruct)",
    )

    parser.add_argument("--config_path", type=str, help="Path to the config.json file")
    
    # Global Paths
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model")

    parser.add_argument("--save_dir", type=str, help="Output directory to save the responses",
                        default="output/")
    
    # Generator Settings
    parser.add_argument("--generator_batch_size", type=int, help="Batch size for the generator",
                        default=5)

    # Debate arguments
    parser.add_argument("--agents", type=int, help="Number of agents",
                        default=2)
    parser.add_argument("--rag_agents", type=int, help="Number of RAG agents for MAD-RAG",
                        default=0)
    parser.add_argument("--max_query_debate_rounds", type=int, help="Number of rounds",
                        default=3) 
    parser.add_argument("--max_answer_debate_rounds", type=int, help="Number of rounds",
                        default=0)
    parser.add_argument("--query_proponent_agent", type=int, help="Number of proponent agents for query stage",
                        default=1)
    parser.add_argument("--query_opponent_agent", type=int, help="Number of opponent agents for query stage",
                        default=1)
    parser.add_argument("--answer_proponent_agent", type=int, help="Number of proponent agents for answer stage",
                        default=1)
    parser.add_argument("--answer_opponent_agent", type=int, help="Number of opponent agents for answer stage",
                        default=1)
    parser.add_argument("--which_original", type=str, help="ablation study for using original model",
                        default="no_original")

    # parser.add_argument("--gpu_id", type=str, default="0")
    
    args = parser.parse_args()
    
    if args.method_name is None:
        raise ValueError("Please choose a method")
    
    if args.config_path is None:
    # warn if config file not found
        print("Warning: No config file provided. Using base configuration.")
        config = {}
        
    else:
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(f"Config file not found at {args.config_path}")
        
        with open(args.config_path, "r") as f:
            config = json.load(f)

    # Step 4: Update config dictionary with provided arguments
    for key, value in vars(args).items():
        if key != "config_path" and value is not None:  # Skip config_path and only update provided arguments
            config[key] = value

    config["save_note"] = args.method_name
    config["save_dir"] = os.path.join(config["save_dir"], config["generator_model"], config["save_note"])
    
    config =  Config("./config/base_config_TRIO.yaml", config)
    
    return config
