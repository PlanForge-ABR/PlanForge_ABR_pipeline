#!/usr/bin/env python3
"""
Natural Language to State Conversion using DSPy - Multi-Domain Optimization
This script optimizes prompts using one example from each domain for better generalization,
while validation and processing remain domain-specific.

Key differences from nl2state.py:
- Prompt optimization uses one example from EACH domain (cross-domain learning)
- Optimized prompt is saved in src/ directory (not domain-specific folder)
- Validation and --N processing remain domain-specific
- Better generalization across different planning domains

Usage examples:
1. Train with cross-domain optimization and validate on specific domain:
python src/nl2state_2.py --train ./data/train --N 20 --N_train 1 --N_validation 10 --domain ferry --out nl2state_result.json --model gemini --model_name gemini-2.5-flash

   Output files:
   - src/ferry/nl2state_result.json (domain-specific results)
   - src/multi_domain_optimized_module.json (cross-domain optimized module)
   - src/multi_domain_optimized_module_prompt.txt (human-readable prompt)

2. Load pre-optimized cross-domain module:
python src/nl2state_2.py --train ./data/train --N 20 --domain ferry --out nl2state_result.json --model gemini --model_name gemini-2.5-flash --load_module multi_domain_optimized_module.json
"""

import os
import sys
import json
import random
import argparse
import dspy
import dotenv
import traceback
from typing import Dict, List, Tuple, Any
from dspy.teleprompt import BootstrapFewShot
from pathlib import Path

dotenv.load_dotenv()

# Import domain descriptions from utils
from src.utils import DOMAIN_DESCRIPTIONS, setup_logging
import logging


class NL2StateSignature(dspy.Signature):
    """Convert natural language description to formal state representation"""
    
    domain_description = dspy.InputField(desc="Description of the domain and its rules")
    context_and_question = dspy.InputField(desc="Current state context and question about goal state")
    initial_state = dspy.OutputField(desc="Initial state as structured predicates")
    goal_state = dspy.OutputField(desc="Goal state as structured predicates")


class NL2StateModule(dspy.Module):
    """DSPy module for natural language to state conversion"""
    
    def __init__(self, enable_thinking: bool = True):
        super().__init__()
        self.enable_thinking = enable_thinking
        if enable_thinking:
            self.generate = dspy.ChainOfThought(NL2StateSignature)
        else:
            self.generate = dspy.Predict(NL2StateSignature)
    
    def forward(self, domain_description: str, context_and_question: str):
        """Convert natural language to state representation"""
        result = self.generate(
            domain_description=domain_description,
            context_and_question=context_and_question,
            think=False
        )
        return result


def setup_model(model_type: str = "openai", model_name: str = "gpt-5.1") -> bool:
    """Setup DSPy language model configuration
    
    Returns:
        bool: True if DSPy thinking should be enabled, False if disabled (for thinking models)
    """
    gemini_api_key = os.getenv('GEMINI_API_KEY', 'GEMINI_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY', 'OPENAI_API_KEY')
    
    if model_type.lower() == "openai":
        if not openai_api_key:
            openai_api_key = os.getenv('OPENAI_API_KEY', 'OPENAI_API_KEY')
        lm = dspy.LM(f'openai/{model_name}', api_key=openai_api_key, temperature=1.0,max_tokens=64000, reasoning_effort="medium")
        print(f"✓ Configured GPT-5.1 model ({model_name})")
        logging.info(f"✓ Configured GPT-5.1 model ({model_name})")
        has_built_in_thinking = True  # GPT-5.1 does have built-in thinking mode

    elif model_type.lower() == "gemini":
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY', 'GEMINI_API_KEY')
        lm = dspy.LM(f'gemini/{model_name}', api_key=gemini_api_key, max_tokens=32000)
        print(f"✓ Configured Gemini model ({model_name})")
        logging.info(f"✓ Configured Gemini model ({model_name})")
        has_built_in_thinking = False

    elif model_type.lower() == "grok":
        xai_api_key = os.getenv('XAI_API_KEY')
        if not xai_api_key:
             raise ValueError("XAI_API_KEY not found in environment variables")
        xai_api_key = xai_api_key.strip()
        lm = dspy.LM(f'openai/{model_name}', api_key=xai_api_key, api_base="https://api.x.ai/v1", max_tokens=64000,cache=False)
        print(f"✓ Configured Grok model ({model_name})")
        logging.info(f"✓ Configured Grok model ({model_name})")
        has_built_in_thinking = False
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'gpt5.1' or 'gemini'")
    
    dspy.configure(lm=lm)
    return not has_built_in_thinking


def load_test_data(domain: str, src_path: str) -> List[Dict[str, Any]]:
    """Load test data from the nl2state_tests.json file"""
    test_file = os.path.join(src_path, domain, "nl2state_tests.json")
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    if domain in data:
        return data[domain]
    else:
        raise KeyError(f"Domain '{domain}' not found in test data")


def load_training_data(train_path: str, domain: str, n_examples: int) -> List[Dict[str, Any]]:
    """Load training data from the specified path"""
    train_file = os.path.join(train_path, f"{domain}-training-dev_08_22_2026.json")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    with open(train_file, 'r') as f:
        data = json.load(f)
    
    if len(data) > n_examples:
        data = random.sample(data, n_examples)
    
    return data


def load_multi_domain_test_data(src_path: str, domains: List[str] = None, n_per_domain: int = 1) -> List[Dict[str, Any]]:
    """Load test data from multiple domains for cross-domain optimization
    
    Args:
        src_path: Path to source directory
        domains: List of domains to load (if None, loads all available domains)
        n_per_domain: Number of examples to load per domain
        
    Returns:
        List of test examples with domain information
    """
    if domains is None:
        domains = list(DOMAIN_DESCRIPTIONS.keys())
    
    all_examples = []
    
    for domain in domains:
        try:
            test_data = load_test_data(domain, src_path)
            
            # Randomly sample n_per_domain examples
            if len(test_data) > n_per_domain:
                sampled_data = random.sample(test_data, n_per_domain)
            else:
                sampled_data = test_data[:n_per_domain]
            
            # Add domain information to each example
            for item in sampled_data:
                item['domain'] = domain
                all_examples.append(item)
            
            print(f"✓ Loaded {len(sampled_data)} examples from {domain}")
            logging.info(f"✓ Loaded {len(sampled_data)} examples from {domain}")
            
        except (FileNotFoundError, KeyError) as e:
            print(f"⚠️  Skipping {domain}: {e}")
            logging.warning(f"⚠️  Skipping {domain}: {e}")
            continue
    
    return all_examples


def convert_test_to_examples(test_data: List[Dict[str, Any]], domain: str = None) -> List[dspy.Example]:
    """Convert test data to DSPy examples
    
    Args:
        test_data: List of test data items (may include 'domain' field for multi-domain data)
        domain: Default domain if not specified in test_data items
    """
    examples = []
    
    for item in test_data:
        # Get domain from item or use provided default
        item_domain = item.get('domain', domain)
        
        if not item_domain:
            raise ValueError("Domain must be specified either in test data or as parameter")
        
        # Get domain description
        domain_desc = DOMAIN_DESCRIPTIONS.get(item_domain, f"This is a {item_domain} planning domain.")
        
        context_and_question = item.get('context_and_question', '')
        example_id = item.get('example_id', None)
        
        # Convert output states to JSON strings
        initial_state = json.dumps(item['output']['initial_state']['state'])
        goal_state = json.dumps(item['output']['goal_state']['state'])
        
        example = dspy.Example(
            example_id=example_id,
            domain=item_domain,
            domain_description=domain_desc,
            context_and_question=context_and_question,
            initial_state=initial_state,
            goal_state=goal_state
        )
        examples.append(example.with_inputs("domain_description", "context_and_question"))
    
    return examples


def convert_training_to_examples(training_data: List[Dict[str, Any]], domain: str) -> List[dspy.Example]:
    """Convert training data to DSPy examples for optimization"""
    examples = []
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, f"This is a {domain} planning domain.")
    
    for item in training_data:
        context = item.get('context', '')
        question = item.get('inputs', '')
        context_and_question = f"{context}\n{question}"
        
        initial_state = json.dumps([{"predicate": "initial", "args": []}])
        goal_state = json.dumps([{"predicate": "goal", "args": []}])
        
        example = dspy.Example(
            domain_description=domain_desc,
            context_and_question=context_and_question,
            initial_state=initial_state,
            goal_state=goal_state
        )
        examples.append(example.with_inputs("domain_description", "context_and_question"))
    
    return examples


def metric_function(example, prediction, trace=None):
    """Metric function for DSPy optimization"""
    try:
        pred_initial = prediction.initial_state.strip()
        pred_goal = prediction.goal_state.strip()
        
        score = 0
        if pred_initial and len(pred_initial) > 10:
            score += 0.5
        if pred_goal and len(pred_goal) > 10:
            score += 0.5
            
        return score
    except:
        return 0


def optimize_prompt_multi_domain(multi_domain_examples: List[dspy.Example], n_train: int, 
                                  enable_thinking: bool = True) -> NL2StateModule:
    """Optimize the prompt using examples from multiple domains
    
    Args:
        multi_domain_examples: Examples from multiple domains
        n_train: Total number of examples to use for optimization
        enable_thinking: Whether to enable thinking mode
        
    Returns:
        Optimized NL2StateModule
    """
    # Use n_train examples for optimization
    if len(multi_domain_examples) > n_train:
        optimization_examples = random.sample(multi_domain_examples, n_train)
    else:
        optimization_examples = multi_domain_examples
    
    print(f"🔧 Optimizing prompt with {len(optimization_examples)} examples from multiple domains...")
    logging.info(f"🔧 Optimizing prompt with {len(optimization_examples)} examples from multiple domains...")
    
    # Show domain distribution
    domain_counts = {}
    for ex in optimization_examples:
        domain = getattr(ex, 'domain', 'unknown')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"📊 Domain distribution: {domain_counts}")
    logging.info(f"📊 Domain distribution: {domain_counts}")
    print(f"🧠 Thinking mode: {'Enabled' if enable_thinking else 'Disabled'}")
    logging.info(f"🧠 Thinking mode: {'Enabled' if enable_thinking else 'Disabled'}")
    
    # Configure the optimizer
    config = dict(
        max_bootstrapped_demos=max(4, len(optimization_examples)), 
        max_labeled_demos=min(2, len(optimization_examples))
    )
    
    optimizer = BootstrapFewShot(metric=metric_function, **config)
    
    # Initialize and optimize the module
    nl2state_module = NL2StateModule(enable_thinking=enable_thinking)
    
    try:
        optimized_module = optimizer.compile(nl2state_module, trainset=optimization_examples)
        print("✅ Multi-domain prompt optimization completed!")
        logging.info("✅ Multi-domain prompt optimization completed!")
        return optimized_module
    except Exception as e:
        print(f"⚠️  Prompt optimization failed: {e}")
        logging.error(f"⚠️  Prompt optimization failed: {e}")
        print("📝 Using base module without optimization")
        logging.info("📝 Using base module without optimization")
        return nl2state_module


class NL2StateProcessor:
    """Main class for Natural Language to State Conversion with Multi-Domain Optimization"""
    
    def __init__(self, domain: str, model: str = "gpt5.1", model_name: str = "gpt-5.1-mini", 
                 src_path: str = "./src", enable_thinking: bool = None):
        """Initialize the NL2State processor
        
        Args:
            domain: Target domain for validation and processing
            model: Model type (gpt5.1 or gemini)
            model_name: Model name
            src_path: Path to source directory
            enable_thinking: Override thinking mode detection
        """
        self.domain = domain
        self.model = model
        self.model_name = model_name
        self.src_path = src_path
        self.optimized_module = None
        
        # Validate domain
        if domain not in DOMAIN_DESCRIPTIONS:
            print(f"⚠️  Warning: Domain '{domain}' not found in DOMAIN_DESCRIPTIONS. Available domains:")
            logging.warning(f"⚠️  Warning: Domain '{domain}' not found in DOMAIN_DESCRIPTIONS. Available domains:")
            for d in DOMAIN_DESCRIPTIONS.keys():
                print(f"   - {d}")
                logging.info(f"   - {d}")
        
        # Setup model
        self.enable_thinking = setup_model(model, model_name)
        if enable_thinking is not None:
            self.enable_thinking = enable_thinking
            print("🧠 Thinking mode explicitly overridden")
            logging.info("🧠 Thinking mode explicitly overridden")
        
        print(f"✅ NL2StateProcessor initialized for domain '{domain}' with {model}/{model_name}")
        logging.info(f"✅ NL2StateProcessor initialized for domain '{domain}' with {model}/{model_name}")
    
    def load_and_optimize_multi_domain(self, n_train_per_domain: int = 1, 
                                        n_validation: int = 0, 
                                        domains: List[str] = None):
        """Load test data from multiple domains and optimize the prompt
        
        Args:
            n_train_per_domain: Number of examples per domain for optimization
            n_validation: Number of examples for validation (from target domain only)
            domains: List of domains to use for optimization (None = all domains)
        
        Returns:
            dict: Validation results if n_validation > 0, None otherwise
        """
        try:
            # Load multi-domain test data for optimization
            print(f"📥 Loading test data from multiple domains for optimization...")
            logging.info(f"📥 Loading test data from multiple domains for optimization...")
            multi_domain_data = load_multi_domain_test_data(
                self.src_path, 
                domains=domains, 
                n_per_domain=n_train_per_domain
            )
            multi_domain_examples = convert_test_to_examples(multi_domain_data)
            print(f"✅ Loaded {len(multi_domain_examples)} examples from {len(set(ex.domain for ex in multi_domain_examples))} domains")
            logging.info(f"✅ Loaded {len(multi_domain_examples)} examples from {len(set(ex.domain for ex in multi_domain_examples))} domains")
            
            # Optimize with multi-domain examples
            total_train = len(multi_domain_examples)
            self.optimized_module = optimize_prompt_multi_domain(
                multi_domain_examples, 
                total_train, 
                self.enable_thinking
            )
            
            # Validation on target domain only
            validation_results = None
            if n_validation > 0:
                print(f"\n🧪 Validating on target domain '{self.domain}'...")
                logging.info(f"\n🧪 Validating on target domain '{self.domain}'...")
                test_data = load_test_data(self.domain, self.src_path)
                
                # Use separate examples for validation
                if len(test_data) > n_validation:
                    validation_data = random.sample(test_data, n_validation)
                else:
                    validation_data = test_data[:n_validation]
                
                validation_examples = convert_test_to_examples(validation_data, self.domain)
                validation_results = self._evaluate_validation(validation_data, validation_examples)
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Failed to load and optimize: {e}")
            logging.error(f"❌ Failed to load and optimize: {e}")
            raise
    
    def _evaluate_validation(self, validation_data: List[Dict], validation_examples: List[dspy.Example]) -> Dict:
        """Internal method to evaluate validation set"""
        print(f"🧪 Evaluating on {len(validation_examples)} validation examples...")
        logging.info(f"🧪 Evaluating on {len(validation_examples)} validation examples...")
        
        validation_results = []
        correct = 0

        for i, (item, example) in enumerate(zip(validation_data, validation_examples)):
            print(f"Processing example {i+1}/{len(validation_examples)}...", end=' ')
            # log to file, but no end=' ' in logger, so we just log "processing..."
            logging.info(f"Processing example {i+1}/{len(validation_examples)}...")
            prediction = self.optimized_module(
                domain_description=example.domain_description,
                context_and_question=example.context_and_question
            )
            
            pred_initial = prediction.initial_state.strip()
            pred_goal = prediction.goal_state.strip()
            
            validation_results.append({
                "example_id": example.example_id if hasattr(example, 'example_id') else None,
                "domain_description": example.domain_description,
                "context_and_question": example.context_and_question,
                "predicted_initial_state": pred_initial,
                "predicted_goal_state": pred_goal,
                "expected_initial_state": example.initial_state,
                "expected_goal_state": example.goal_state
            })
            
            if pred_initial and pred_goal and len(pred_initial) > 10 and len(pred_goal) > 10:
                correct += 1
                print("✅")
                logging.info("✅")
            else:
                print("❌")
                logging.info("❌")
        
        accuracy = correct / len(validation_examples) if validation_examples else 0
        print(f"📊 Validation accuracy: {accuracy:.2%}")
        logging.info(f"📊 Validation accuracy: {accuracy:.2%}")
        
        return {
            "results": validation_results,
            "accuracy": accuracy
        }
    
    def process_training_examples(self, train_path: str, n_examples: int) -> List[Dict[str, Any]]:
        """Process N examples from training data (domain-specific)
        
        Args:
            train_path: Path to training data directory
            n_examples: Number of examples to process
            
        Returns:
            List of results with predicted states
        """
        if self.optimized_module is None:
            raise ValueError("Model not optimized yet. Call load_and_optimize_multi_domain() first.")
        
        try:
            print(f"📥 Loading {n_examples} examples from {self.domain} train data...")
            logging.info(f"📥 Loading {n_examples} examples from {self.domain} train data...")
            train_data = load_training_data(train_path, self.domain, n_examples)
            train_examples = convert_training_to_examples(train_data, self.domain)
            print(f"✅ Loaded {len(train_examples)} train examples")
            logging.info(f"✅ Loaded {len(train_examples)} train examples")
            
            results = []
            print(f"🔄 Processing {len(train_examples)} training examples...")
            logging.info(f"🔄 Processing {len(train_examples)} training examples...")
            
            for i, (item, example) in enumerate(zip(train_data, train_examples), 1):
                print(f"Processing example {i}/{len(train_examples)}...", end=' ')
                logging.info(f"Processing example {i}/{len(train_examples)}...")
                
                prediction = self.optimized_module(
                    domain_description=example.domain_description,
                    context_and_question=example.context_and_question
                )
                
                results.append({
                    "example_id": item.get("id", None),
                    "domain_description": example.domain_description,
                    "context": item.get("context", ""),
                    "inputs": item.get("inputs", ""),
                    "predicted_initial_state": prediction.initial_state.strip(),
                    "predicted_goal_state": prediction.goal_state.strip()
                })
                print("✅")
                logging.info("✅")
            
            return results
            
        except Exception as e:
            print(f"❌ Failed to process training examples: {e}")
            logging.error(f"❌ Failed to process training examples: {e}")
            raise
    
    def process_example(self, context: str, inputs: str, example_id: str = None) -> Dict[str, Any]:
        """Process a single example with context and inputs
        
        Args:
            context: Context description
            inputs: Input question/query
            example_id: Optional example identifier
            
        Returns:
            Dict with predicted states
        """
        if self.optimized_module is None:
            raise ValueError("Model not optimized yet. Call load_and_optimize_multi_domain() first.")
        
        domain_desc = DOMAIN_DESCRIPTIONS.get(self.domain, f"This is a {self.domain} planning domain.")
        context_and_question = f"{context}\n{inputs}"
        
        prediction = self.optimized_module(
            domain_description=domain_desc,
            context_and_question=context_and_question
        )
        
        return {
            "example_id": example_id,
            "domain_description": domain_desc,
            "context": context,
            "inputs": inputs,
            "predicted_initial_state": prediction.initial_state.strip(),
            "predicted_goal_state": prediction.goal_state.strip()
        }
    
    def save_optimized_module(self, module_file: str = "multi_domain_optimized_module.json"):
        """Save the optimized DSPy module to src/ directory (not domain-specific)
        
        Args:
            module_file: Output filename for the optimized module
        """
        if self.optimized_module is None:
            raise ValueError("No optimized module to save. Call load_and_optimize_multi_domain() first.")
        
        # Save to src/ directory (not domain-specific)
        output_path = os.path.join(self.src_path, module_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the module using DSPy's save method
        self.optimized_module.save(output_path)
        print(f"💾 Multi-domain optimized module saved to {output_path}")
        logging.info(f"💾 Multi-domain optimized module saved to {output_path}")
        
        # Also save the prompt in a human-readable text format
        prompt_file = module_file.replace('.json', '_prompt.txt')
        prompt_path = os.path.join(self.src_path, prompt_file)
        
        with open(prompt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-DOMAIN OPTIMIZED DSPY PROMPT FOR NL2STATE CONVERSION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Target Domain: {self.domain}\n")
            f.write(f"Model: {self.model}/{self.model_name}\n")
            f.write(f"Thinking Mode: {'Enabled' if self.enable_thinking else 'Disabled'}\n")
            f.write(f"Optimization: Cross-domain (one example per domain)\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("MODULE CONFIGURATION\n")
            f.write("=" * 80 + "\n\n")
            
            try:
                if hasattr(self.optimized_module, 'generate'):
                    predictor = self.optimized_module.generate
                    f.write(f"Predictor Type: {type(predictor).__name__}\n\n")
                    
                    # Capture actual prompt
                    f.write("=" * 80 + "\n")
                    f.write("ACTUAL PROMPTS SENT TO LLM\n")
                    f.write("=" * 80 + "\n\n")
                    
                    try:
                        # Create a sample input
                        domain_desc = DOMAIN_DESCRIPTIONS.get(self.domain, f"This is a {self.domain} planning domain.")
                        sample_context = "Sample: Ferry is at location A with car C1 on board. Can we get car C1 to location B?"
                        
                        # Enable history tracking
                        dspy.settings.configure(trace=[])
                        
                        # Make prediction to capture prompt
                        _ = self.optimized_module(
                            domain_description=domain_desc,
                            context_and_question=sample_context
                        )
                        
                        # Get the history
                        if hasattr(dspy.settings, 'trace') and dspy.settings.trace:
                            history = dspy.settings.trace
                            f.write(f"Captured {len(history)} LM call(s)\n\n")
                            
                            for idx, entry in enumerate(history, 1):
                                f.write(f"--- LM Call {idx} ---\n")
                                f.write(str(entry))
                                f.write("\n\n")
                        else:
                            f.write("No history captured in trace\n\n")
                            
                            # Try alternative
                            lm = dspy.settings.lm
                            if hasattr(lm, 'history') and lm.history:
                                f.write(f"Found {len(lm.history)} entries in LM history\n\n")
                                for idx, entry in enumerate(lm.history[-3:], 1):
                                    f.write(f"--- History Entry {idx} ---\n")
                                    f.write(str(entry))
                                    f.write("\n\n")
                            else:
                                f.write("No history available\n\n")
                    
                    except Exception as e:
                        f.write(f"Could not capture history: {e}\n\n")
                        f.write(f"Traceback:\n{traceback.format_exc()}\n\n")
                    
                    # Write demonstrations
                    if hasattr(predictor, 'demos') and predictor.demos:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write(f"DEMONSTRATIONS ({len(predictor.demos)} examples)\n")
                        f.write("=" * 80 + "\n\n")
                        for i, demo in enumerate(predictor.demos, 1):
                            f.write(f"\n--- Demo {i} ---\n")
                            if hasattr(demo, 'toDict'):
                                f.write(json.dumps(demo.toDict(), indent=2))
                            elif hasattr(demo, '__dict__'):
                                f.write(str(demo.__dict__))
                            else:
                                f.write(str(demo))
                            f.write("\n")
                    
                    # Write extended signature
                    if hasattr(predictor, 'extended_signature'):
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("EXTENDED SIGNATURE\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(str(predictor.extended_signature))
                        f.write("\n")
                    
                    # Write signature instructions
                    if hasattr(predictor, 'signature'):
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("SIGNATURE INSTRUCTIONS\n")
                        f.write("=" * 80 + "\n\n")
                        sig = predictor.signature
                        if hasattr(sig, 'instructions'):
                            f.write(f"Instructions: {sig.instructions}\n")
                        if hasattr(sig, 'input_fields'):
                            f.write(f"Input Fields: {sig.input_fields}\n")
                        if hasattr(sig, 'output_fields'):
                            f.write(f"Output Fields: {sig.output_fields}\n")
                        f.write("\n")
                    
                    # Write full predictor state
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("FULL PREDICTOR STATE\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(str(predictor))
                    f.write("\n")
                    
            except Exception as e:
                f.write(f"\n⚠️  Could not extract detailed prompt information: {e}\n")
                f.write(f"\nTraceback:\n{traceback.format_exc()}\n")
                f.write(f"\nModule type: {type(self.optimized_module)}\n")
                f.write(f"Module representation:\n{str(self.optimized_module)}\n")
        
        print(f"📝 Human-readable prompt saved to {prompt_path}")
        return output_path, prompt_path
    
    def load_optimized_module(self, module_file: str = "multi_domain_optimized_module.json"):
        """Load a previously saved optimized module from src/ directory
        
        Args:
            module_file: Filename of the saved module
        """
        module_path = os.path.join(self.src_path, module_file)
        
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Optimized module file not found: {module_path}")
        
        # Initialize a base module
        base_module = NL2StateModule(enable_thinking=self.enable_thinking)
        
        # Load the saved module
        base_module.load(module_path)
        self.optimized_module = base_module
        
        print(f"✅ Multi-domain optimized module loaded from {module_path}")
        logging.info(f"✅ Multi-domain optimized module loaded from {module_path}")
        
        # Check if prompt text file exists
        prompt_file = module_file.replace('.json', '_prompt.txt')
        prompt_path = os.path.join(self.src_path, prompt_file)
        if os.path.exists(prompt_path):
            print(f"📝 Human-readable prompt available at {prompt_path}")
            logging.info(f"📝 Human-readable prompt available at {prompt_path}")
        
        return self.optimized_module
    
    def save_results(self, train_results: List[Dict] = None, validation_results: Dict = None, 
                     output_file: str = "results.json"):
        """Save results to domain-specific folder
        
        Args:
            train_results: Results from training data processing
            validation_results: Validation results
            output_file: Output filename
        """
        output_data = {
            "metadata": {
                "domain": self.domain,
                "model": self.model,
                "model_name": self.model_name,
                "thinking_mode_enabled": self.enable_thinking,
                "optimization_type": "multi_domain",
                "total_train_examples_processed": len(train_results) if train_results else 0,
                "total_validation_examples_processed": len(validation_results["results"]) if validation_results else 0,
                "validation_accuracy": f"{validation_results['accuracy']:.2%}" if validation_results else "N/A"
            }
        }
        
        if train_results:
            output_data["train_results"] = train_results
        
        if validation_results:
            output_data["validation_results"] = validation_results["results"]
        
        # Save to domain-specific folder
        output_path = os.path.join(self.src_path, self.domain, output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to {output_path}")
        logging.info(f"💾 Results saved to {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Natural Language to State Conversion with Multi-Domain Optimization")
    parser.add_argument("--train", required=True, help="Path to training data directory")
    parser.add_argument("--N", type=int, required=True, help="Number of examples to process from train data (domain-specific)")
    parser.add_argument("--N_train", type=int, default=1, help="Number of examples PER DOMAIN for cross-domain optimization (0 to skip)")
    parser.add_argument("--N_validation", type=int, default=0, help="Number of examples for validation (from target domain)")
    parser.add_argument("--domain", required=True, help="Target domain for validation and processing")
    parser.add_argument("--out", required=True, help="Output file name for results")
    parser.add_argument("--model", default="openai", choices=["openai", "gemini"], help="Model provider to use")
    parser.add_argument("--model_name", default="gpt-5.1", help="Model name (e.g., gpt-5.1, gemini-2.5-flash)")
    parser.add_argument("--src", default="./src", help="Path to source directory")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable DSPy thinking mode")
    parser.add_argument("--load_module", type=str, help="Load pre-optimized module (e.g., multi_domain_optimized_module.json)")
    parser.add_argument("--domains", nargs='+', help="Domains to use for optimization (default: all available)")

    args = parser.parse_args()

    setup_logging()

    print("🚀 Starting Multi-Domain Natural Language to State Conversion")
    logging.info("🚀 Starting Multi-Domain Natural Language to State Conversion")
    print(f"📁 Training data: {args.train}")
    logging.info(f"📁 Training data: {args.train}")
    print(f"🎯 Target domain: {args.domain}")
    logging.info(f"🎯 Target domain: {args.domain}")
    print(f"📊 Train data examples to process: {args.N}")
    logging.info(f"📊 Train data examples to process: {args.N}")
    print(f"🏋️  N_train per domain (optimization): {args.N_train}")
    logging.info(f"🏋️  N_train per domain (optimization): {args.N_train}")
    print(f"🧪 N_validation: {args.N_validation}")
    logging.info(f"🧪 N_validation: {args.N_validation}")
    print(f"🤖 Model: {args.model}")
    logging.info(f"🤖 Model: {args.model}")
    print(f"📝 Model name: {args.model_name}")
    logging.info(f"📝 Model name: {args.model_name}")
    if args.load_module:
        print(f"📂 Loading pre-optimized module: {args.load_module}")
        logging.info(f"📂 Loading pre-optimized module: {args.load_module}")
    if args.domains:
        print(f"🌐 Optimization domains: {', '.join(args.domains)}")
        logging.info(f"🌐 Optimization domains: {', '.join(args.domains)}")
    else:
        print(f"🌐 Optimization domains: All available")
        logging.info(f"🌐 Optimization domains: All available")

    try:
        # Initialize the processor
        processor = NL2StateProcessor(
            domain=args.domain,
            model=args.model,
            model_name=args.model_name,
            src_path=args.src,
            enable_thinking=args.enable_thinking if args.enable_thinking else None
        )
        
        validation_results = None
        
        # Either load pre-optimized module or optimize from scratch
        if args.load_module:
            print(f"\n📂 Loading pre-optimized module from {args.load_module}...")
            logging.info(f"\n📂 Loading pre-optimized module from {args.load_module}...")
            processor.load_optimized_module(args.load_module)
            print("✅ Pre-optimized module loaded successfully!")
            logging.info("✅ Pre-optimized module loaded successfully!")
        else:
            if args.N_train == 0:
                raise ValueError("Either provide --load_module or set --N_train > 0 to train a new module")
            
            # Load and optimize with multi-domain data
            validation_results = processor.load_and_optimize_multi_domain(
                n_train_per_domain=args.N_train,
                n_validation=args.N_validation,
                domains=args.domains
            )
            
            # Save the optimized module
            print("\n💾 Saving multi-domain optimized module...")
            logging.info("\n💾 Saving multi-domain optimized module...")
            module_file = "multi_domain_optimized_module.json"
            processor.save_optimized_module(module_file)
        
        # Process training examples (domain-specific)
        train_results = processor.process_training_examples(
            train_path=args.train,
            n_examples=args.N
        )
        
        # Save results (domain-specific)
        processor.save_results(
            train_results=train_results,
            validation_results=validation_results,
            output_file=args.out
        )
        
        print("🎉 Processing completed successfully!")
        logging.info("🎉 Processing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
