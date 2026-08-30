from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever
from flashrag.config import Config
from tqdm import tqdm
import torch
# from flashrag.prompt import PromptTemplate
from transformers import AutoTokenizer


class AugmentedDebateSystem(BasicPipeline):
    def __init__(self, config, prompt_template=None, 
                 max_query_debate_rounds=3,
                 generator=None, retriever=None,use_dual_models=True):

        super().__init__(config, prompt_template)
        self.config = config
        self.max_query_debate_rounds = max_query_debate_rounds
        self.use_dual_models = use_dual_models

        # Initialize generators
        if generator is None:
            if self.use_dual_models:
                print("Initializing dual model setup...")
                # query 模型
                # Print available GPU count and their indices
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    print(f"Number of available GPUs: {gpu_count}")
                    for i in range(gpu_count):
                        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                else:
                    print("No GPUs available.")
                query_config = dict(self.config.final_config) if hasattr(self.config, 'final_config') else dict(self.config)
                query_config["generator_model"] = 'query_writer_hf_Instruct'
                query_config["generator_model_path"] = '/scratch/Inference_materials/pretrained_model_Instruct/latest_query_agent_pro1'

                query_config['generator_max_input_len'] = 4096
                query_config["gpu_memory_utilization"] = 0.45
                query_config['generation_params']['max_tokens'] = 32
                
                # judge 模型
                judge_config = dict(self.config.final_config) if hasattr(self.config, 'final_config') else dict(self.config)
                judge_config["generator_model"] = 'judge_hf_Instruct'
                judge_config["generator_model_path"] = '/scratch/Inference_materials/pretrained_model_Instruct/llama-3-8b-combined3_Instruct'
                judge_config['generator_max_input_len'] = 4096
                judge_config["gpu_memory_utilization"] = 0.45
                judge_config['generation_params']['max_tokens'] = 32


                print('---------------------------------------')
                print(self.config.final_config['which_original'])
                print(self.config.final_config['generator_model_path'])

                self.judge_generator = get_generator(Config(config_dict=judge_config))
                self.query_generator = get_generator(Config(config_dict=query_config))

                self.generator = None
                model_path = query_config["generator_model_path"]

                self.tokenizer_length = AutoTokenizer.from_pretrained(model_path)

            else:
                print("Using single model setup...")
                self.generator = get_generator(config)
                self.query_generator = self.generator
                self.answer_generator = self.generator
                print(f"Single model loaded: {config['generator_model']}")
        else:
            self.generator = generator
            self.query_generator = generator
            self.answer_generator = generator
        

        self.retriever = get_retriever(config) if retriever is None else retriever
        print("Loading local retrieval service")

        self.agents_messages_answer_stage = dict()
        self.agents_messages_query_stage = dict()

        self.record = [] # 周期性记录数据

    def run(self, dataset, do_eval=True, batch_size=100):
        total = len(dataset)
        print(f"Total items in dataset: {total}")

        print(f"Using batch processing with batch size: {batch_size}")
        # Batch processing for query stage
        all_query_pools = self.query_stage_debate_batch(dataset.data, batch_size=batch_size)
        
        # Batch processing for answer stage
        print("Running batch answer generation...")
        self.answer_only_batch(dataset.data, all_query_pools, batch_size=batch_size)
        
        # Update outputs for all items
        for item in dataset:
            item_id = id(item)
            query_pool = all_query_pools.get(item_id, {})
            item.update_output("QueryStage_QueryPool", query_pool)

        if do_eval:
            dataset = self.evaluate(dataset)

        return dataset

    def batch_retrieve(self, queries, topk=None):
        """
        Batch retrieval interface that can switch between local and remote retrieval.
        
        Args:
            queries: List of query strings
            topk: Number of top results to return per query
            
        Returns:
            List of retrieval results, each corresponding to one query
        """
        if topk is None:
            topk = self.config["retrieval_topk"] if "retrieval_topk" in self.config else 3
            
        return self._local_batch_retrieve(queries, topk)
    
    def _local_batch_retrieve(self, queries, topk):
        """
        Use FlashRAG's local batch retrieval method.
        """
        print(f"Using local batch retrieval for {len(queries)} queries...")
        
        # Use FlashRAG's batch search if available
        if hasattr(self.retriever, 'batch_search'):
            results = self.retriever.batch_search(queries, num=topk)
        else:
            # Fallback to individual searches with progress bar
            results = []
            for query in tqdm(queries, desc="Retrieving", unit="query"):
                result = self.retriever.search(query, num=topk)
                results.append(result)
        
        return results

    
    def query_stage_debate_batch(self, items, batch_size=100):
        """
        Batch processing version of query_stage_debate.
        Process items in batches and continue retrieval rounds until all items 
        are determined to stop retrieving or maximum rounds are exhausted.
        """

        # Initialize result storage for all items
        all_query_pools = {}
        
        # Handle case where no debate rounds are needed
        if self.max_query_debate_rounds == 0:
            print("No debate rounds, performing batch retrieval...")
            queries = [item.question for item in items]
            
            # Batch retrieval for all queries at once
            print("Performing batch retrieval for initial queries...")
            batch_results = self.batch_retrieve(queries)
            
            for item, retrieval_results in zip(items, batch_results):
                query_pool = {item.question.strip(): retrieval_results}
                all_query_pools[id(item)] = query_pool
            
            return all_query_pools
        
        # Initialize active items (items that need further processing)
        active_items = list(items)
        
        # Process items in rounds
        for round_num in range(self.max_query_debate_rounds):
            if not active_items:
                break
                
            print(f"Query stage round {round_num + 1}/{self.max_query_debate_rounds}, processing {len(active_items)} items")
            
            # Process active items in batches
            items_to_continue = []
            
            for batch_start in tqdm(range(0, len(active_items), batch_size), 
                                  desc=f"Round {round_num + 1} Batches"):
                batch_end = min(batch_start + batch_size, len(active_items))
                batch_items = active_items[batch_start:batch_end]
                
                # Process current batch
                batch_results = self._process_batch_round(batch_items, round_num, all_query_pools)
                
                # Collect items that need to continue to next round
                for item, should_continue in batch_results:
                    if should_continue:
                        items_to_continue.append(item)
            
            print(f"Round {round_num + 1} completed. {len(items_to_continue)} items will continue to next round.")
            
            # Update active items for next round
            active_items = items_to_continue
        
        print(f"Query stage completed. Final query pools generated for {len(all_query_pools)} items.")
        return all_query_pools
    
    def _process_batch_round(self, batch_items, round_num, all_query_pools):
        """
        Process a batch of items for one debate round with batch inference.
        Returns list of (item, should_continue) tuples.
        """
        batch_results = []
        
        # Step 1: Initialize query pools for first round with batch retrieval
        if round_num == 0:
            # Collect all queries for batch retrieval
            queries = [item.question for item in batch_items]
            print(f"Performing batch retrieval for {len(queries)} queries in round {round_num + 1}...")
            
            # Use batch retrieval
            retrieval_results = self.batch_retrieve(queries)
            
            # Initialize query pools
            for item, results in zip(batch_items, retrieval_results):
                item_id = id(item)
                if item_id not in all_query_pools:
                    all_query_pools[item_id] = {}
                all_query_pools[item_id][item.question.strip()] = results
        
        # Step 2: Batch inference for each agent
        items_data = []
        for item in batch_items:
            item_id = id(item)
            query_pool = all_query_pools[item_id]
            items_data.append({
                'item': item,
                'item_id': item_id,
                'query_pool': query_pool,
                'input_query': item.question,
                'agents_messages': {}
            })
        
        # Process each agent type with batch inference
        print(f"Processing agent query for batch of {len(batch_items)} items...")
        agent_name = 'Query_Writer'
        # Prepare batch prompts
        batch_prompts = []
        for item_data in items_data:
            formated_query_pool = self.format_query_pool(item_data['query_pool'])
            round_message = [
                self._query_stage_system_message(agent_name),
                {"role": "user", "content": f"QUESTION: {item_data['input_query']}\nEXISTING_QUERIES_and_RETRIEVED_DOCUMENTS:\n{formated_query_pool}"}
            ]
            input_prompt = self.prompt_template.get_string(messages=round_message)
            batch_prompts.append(input_prompt)
            
        # Batch inference
        outputs = self.query_generator.generate(batch_prompts)
        
        # Store results
        for item_data, input_prompt, output in zip(items_data, batch_prompts, outputs):
            item = item_data['item']
            item.update_output(f"QueryStage_Round_{agent_name}_{round_num}_InputPrompt", input_prompt)
            item.update_output(f"QueryStage_Round_{agent_name}_{round_num}_Output", output)
            item_data['agents_messages'][agent_name] = [input_prompt, output]
        
        # Step 3: Record and moderator decisions with batch inference
        print(f"Processing moderator decisions for batch of {len(batch_items)} items...")
        
        # Prepare batch moderator prompts
        moderator_prompts = []
        for item_data in items_data:
            # Record for this item
            formated_query_pool = self.format_query_pool(item_data['query_pool'])

            self.record.append({
                "question": item_data['input_query'],
                "query_pool": self.format_query_pool(item_data['query_pool']),
            })
            # Moderator prompt
            moderator_message = [
                self._query_stage_moderator_message(
                item_data['agents_messages'], 
                item_data['input_query'], 
                item_data['query_pool']
            ),# 这里输出的是整个的systrm dict
            {"role": "user", "content": f"QUESTION: {item_data['input_query']}"}
            ]
            moderator_input_prompt = self.prompt_template.get_string(messages=moderator_message)
            moderator_prompts.append(moderator_input_prompt)
        
        # Batch moderator inference
        moderator_outputs = self.judge_generator.generate(moderator_prompts)
        
        # Step 4: Process moderator decisions and update query pools
        new_queries_batch = []
        new_queries_items = []
        
        for item_data, moderator_input_prompt, moderator_output in zip(items_data, moderator_prompts, moderator_outputs):
            item = item_data['item']
            item.update_output(f"QueryStage_Moderator_Round{round_num}_InputPrompt", moderator_input_prompt)
            item.update_output(f"QueryStage_Moderator_Round{round_num}_Output", moderator_output)
            
            # Determine if this item should continue
            should_continue = False
            # print('强制到最大轮次')
            # if False:
            if "Sufficient Information" in moderator_output and round_num > 0: # 强制至少添加一次检索
                # Proponent wins - stop for this item
                should_continue = False

            else:
                # Opponent wins - try to continue
                opponent_output = item_data['agents_messages']["Query_Writer"][1]
                new_query = self._handle_new_query(opponent_output)
                if new_query:
                    new_queries_batch.append(new_query)
                    new_queries_items.append((item_data, new_query))
                    should_continue = True
            
            batch_results.append((item, should_continue))
        
        # Step 5: Batch retrieval for new queries
        if new_queries_batch:
            print(f"Performing batch retrieval for {len(new_queries_batch)} new queries...")
            new_retrieval_results = self.batch_retrieve(new_queries_batch)
            
            # Update query pools with new results
            for (item_data, new_query), retrieval_results in zip(new_queries_items, new_retrieval_results):
                all_query_pools[item_data['item_id']][new_query] = retrieval_results
        
        return batch_results
    def _handle_new_query(self, query):
        # if "NEW_QUERY: Yes," in query:
        #     query = query.replace("NEW_QUERY\n: Yes,", "").strip()

        if "NEW_QUERY" in query:
            query = query.replace("NEW_QUERY", "").strip()
            if "Yes" in query:
                query = query.replace(":\nYes,", "").strip()
        return query


    def _extract_new_query(self, opponent_output):
        """
        Extract new query from opponent output.
        Returns None if no valid new query is found.
        """
        try:
            if "Query Optimization:" in opponent_output:
                optimization_instruction = opponent_output.split("Query Optimization:")[1].strip()
                if "->" in optimization_instruction:
                    new_query = optimization_instruction.split("->")[1].strip()
                else:
                    new_query = optimization_instruction
                return new_query
            elif "Query Expansion:" in opponent_output:
                new_query = opponent_output.split("Query Expansion:")[1].strip()
                return new_query
            else:
                return None
        except Exception as e:
            print(f"Error extracting new query: {e}")
            return None

    def answer_only_batch(self, items, all_query_pools, batch_size=100):
        """
        Batch processing for answer-only mode (no debate).
        """
        del(self.query_generator)
        del(self.judge_generator)
        print('delete two former model')

        # answer 模型加载
        answer_config= dict(self.config.final_config) if hasattr(self.config, 'final_config') else dict(self.config)
        answer_config["generator_model"] = 'hard_only_answer_instruct'
        answer_config["generator_model_path"] = '/scratch/Inference_materials/pretrained_model_Instruct/only_answer_8000_440_end_Instruct'

        answer_config["gpu_memory_utilization"] = 0.6
            
        self.answer_generator = get_generator(Config(config_dict=answer_config))

        print("Generating final answers without debate...")

        for batch_start in range(0, len(items), batch_size):
            batch_end = min(batch_start + batch_size, len(items))
            batch_items = items[batch_start:batch_end]
            
            batch_prompts = []
            for item in batch_items:
                item_id = id(item)
                query_pool = all_query_pools.get(item_id, {})

                message = [
                    self._answer_only_message(query_pool),
                    {"role": "user", "content": f"Question: {item.question}\n"}
                ]
                input_prompt = self.prompt_template.get_string(messages=message)
                batch_prompts.append(input_prompt)
              #tokens = self.tokenizer_length.encode(prompt, add_special_tokens=False)
                # text = tokenizer.decode(tokens)
            
            # Batch generate
            batch_outputs = self.answer_generator.generate(batch_prompts)
            
            # Store results
            for i, item in enumerate(batch_items):
                input_prompt = batch_prompts[i]
                output = batch_outputs[i]
                
                item.update_output("answer_input_prompt", input_prompt)
                item.update_output("pred", output)

    # +++ MODIFIED METHOD: For Proponent and Opponent +++
    def _query_stage_system_message(self, agent_name):
        system_message = {
            "role": "system",
            "content": """
You are a retrieval query generation agent.

Input:
QUESTION: …
EXISTING_QUERIES_and_RETRIEVED_DOCUMENTS: …

Task:
Generate exactly one NEW_QUERY that:
1) Directly helps answer the QUESTION or fill gaps not covered by EXISTING_QUERIES_and_RETRIEVED_DOCUMENTS;
2) Is different from all EXISTING_QUERIES.

Output rules:
- Only keywords; no sentences, stopwords, or connectors (e.g., and, or, of, the, about).
- Separate keywords with a single space;
- Maximum 8 keywords.

Output:
Only the NEW_QUERY text. No explanations or extra text.
                """
            }
        return system_message      
    
    def _query_stage_moderator_message(self, agents_messages, input_query, query_pool):
        agents_arguments = ""
        for agent in agents_messages:
            agents_arguments += f"{agent}: {agents_messages[agent][1]}\n"
            
        system_message = {
            "role": "system",
            "content": 
            # "The User asks a QUESTION, and the Assistant solves it. Answer the QUESTION based on the given EXISTING_QUERIES_and_RETRIEVED_DOCUMENTS. Output only the final answer. If the information is not available, respond with “Insufficient Information. DO NOT include explanations or additional text."
            f'Determine whether the EXISTING_QUERIES_and_RETRIEVED_DOCUMENTS provide sufficient information to answer the QUESTION correctly. Output only one of the following: "Sufficient Information" or "Insufficient Information". Do not include explanations or additional text.\n{self.format_query_pool(query_pool)}'
        }
        
        return system_message
        
    def _answer_only_message(self, query_pool):
        # ... (no change) ...
        if self.config["dataset_name"] == "StrategyQA":
            system_message = {"role": "system", 
                            "content": f"Answer the question based on the given document. Given two answer candidates, Yes and No, choose the best answer choice. Output only the final answer with no explanations or additional text.\n{self.format_query_pool(query_pool)}"}
        else:
            system_message = {"role": "system", 
                            "content": f"Answer the question based on the given document. Output only the final answer with no explanations or additional text.\n{self.format_query_pool(query_pool)}"}
        return system_message  
    
    def _format_reference(self, retrieval_result):
        # ... (no change) ...
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    
    def maintain_query_pool(self, query_pool, opponent_output):
        # ... (no change) ...
        try:
            if "Query Optimization:" in opponent_output:
                optimization_instruction = opponent_output.split("Query Optimization:")[1].strip()
                if "->" in optimization_instruction:
                    optimization_instruction = optimization_instruction.split("->")
                    original_query = optimization_instruction[0].strip()
                    new_query = optimization_instruction[1].strip()
                else:
                    original_query = optimization_instruction
                    new_query = optimization_instruction

                # query_pool.pop(self.find_most_similar_key(query_pool, original_query))

                retrieval_results = self.retriever.search(new_query)
                query_pool[new_query] = retrieval_results
            elif "Query Expansion:" in opponent_output:
                new_query = opponent_output.split("Query Expansion:")[1].strip()
                retrieval_results = self.retriever.search(new_query)
                query_pool[new_query] = retrieval_results
            else:
                return None
        except Exception as e:
            print(f"Error: {e}")
            print("\n")
            print(query_pool)
            print("\n")
            print(opponent_output)
            # It's better to return the original pool than to crash
            return query_pool
        
        return query_pool
    
    def format_query_pool(self, query_pool):
        # ... (no change) ...
        query_pool_str = ""
        for i, query in enumerate(query_pool):
            query_pool_str += f"Query {i+1}: {query}\nRetrieved Content:\n{self._format_reference(query_pool[query])}"
        
        return query_pool_str