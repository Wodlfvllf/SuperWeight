"""Model evaluation including zero-shot benchmarks and perplexity."""

import torch
import numpy as np
from typing import Dict, List, Optional
import logging
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluation of LLMs."""
    
    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
    @torch.no_grad()
    def evaluate_perplexity(
        self,
        dataset_name: str = "wikitext2",
        split: str = "test",
        seq_length: int = 2048,
        stride: int = 512
    ) -> float:
        """Compute perplexity on a dataset.
        
        Args:
            dataset_name: Name of dataset
            split: Dataset split
            seq_length: Sequence length for evaluation
            stride: Stride for sliding window
        
        Returns:
            Perplexity value
        """
        logger.info(f"Evaluating perplexity on {dataset_name}")
        
        # Load dataset
        if dataset_name == "wikitext2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            text = "\n\n".join(dataset["text"])
        elif dataset_name == "c4":
            dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            texts = []
            for i, example in enumerate(dataset):
                if i >= 256:  # Use subset for efficiency
                    break
                texts.append(example["text"])
            text = "\n\n".join(texts)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Tokenize
        encodings = self.tokenizer(text, return_tensors="pt")
        
        # Compute perplexity with sliding window
        nlls = []
        for begin_loc in tqdm(range(0, encodings.input_ids.size(1), stride)):
            end_loc = min(begin_loc + seq_length, encodings.input_ids.size(1))
            trg_len = end_loc - begin_loc
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
            
            nlls.append(neg_log_likelihood)
            
            if end_loc == encodings.input_ids.size(1):
                break
        
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        logger.info(f"{dataset_name} perplexity: {ppl.item():.2f}")
        
        return ppl.item()
    
    def evaluate_zero_shot(
        self,
        task_name: str,
        num_fewshot: int = 0
    ) -> Dict[str, float]:
        """Evaluate on zero-shot benchmark using lm-evaluation-harness.
        
        Args:
            task_name: Name of benchmark task
            num_fewshot: Number of few-shot examples
        
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            
            # Wrap model for lm-eval
            lm = HFLM(pretrained=self.model, tokenizer=self.tokenizer)
            
            # Run evaluation
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                batch_size=1
            )
            
            return results['results'][task_name]
            
        except ImportError:
            logger.error("lm-eval not installed. Install with: pip install lm-eval")
            return {}
        except Exception as e:
            logger.error(f"Error evaluating {task_name}: {e}")
            return {}
    
    def evaluate_all_benchmarks(
        self,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """Evaluate on all configured zero-shot benchmarks.
        
        Args:
            tasks: List of task names (uses config if None)
        
        Returns:
            Dictionary mapping task names to results
        """
        if tasks is None:
            tasks = self.config['evaluation']['tasks']
        
        results = {}
        for task in tasks:
            logger.info(f"Evaluating {task}...")
            task_results = self.evaluate_zero_shot(task)
            results[task] = task_results
        
        # Compute average accuracy
        accuracies = []
        for task, task_results in results.items():
            if 'acc' in task_results:
                accuracies.append(task_results['acc'])
            elif 'acc_norm' in task_results:
                accuracies.append(task_results['acc_norm'])
        
        if accuracies:
            results['average_accuracy'] = np.mean(accuracies)
            logger.info(f"Average accuracy: {results['average_accuracy']:.4f}")
        
        return results
    
    def compare_models(
        self,
        baseline_results: Dict,
        modified_results: Dict
    ) -> Dict:
        """Compare baseline and modified model results.
        
        Args:
            baseline_results: Results from baseline model
            modified_results: Results from modified model
        
        Returns:
            Comparison statistics
        """
        comparison = {
            'task_comparisons': {},
            'average_delta': 0.0
        }
        
        deltas = []
        for task in baseline_results:
            if task in modified_results and task != 'average_accuracy':
                baseline_acc = baseline_results[task].get('acc', 0)
                modified_acc = modified_results[task].get('acc', 0)
                delta = modified_acc - baseline_acc
                
                comparison['task_comparisons'][task] = {
                    'baseline': baseline_acc,
                    'modified': modified_acc,
                    'delta': delta,
                    'relative_delta': (delta / baseline_acc * 100) if baseline_acc > 0 else 0
                }
                deltas.append(delta)
        
        if deltas:
            comparison['average_delta'] = np.mean(deltas)
        
        return comparison
