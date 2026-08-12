#!/usr/bin/env python3
"""
Natural Language to State Conversion using DSPy
This script converts natural language descriptions of planning problems into formal state representations
using large language models via the DSPy framework. It supports prompt optimization and evaluation.
Domains supported: blocksworld, ferry, logistics, grippers, rovers, visitall, grid, floortile

Usage examples:
1. Command line - Train and save optimized module:
python src/nl2state.py --train ./data/train --N 20 --N_train 4 --N_validation 10 --domain ferry --out nl2state_result.json --model gpt5.1 --model_name gpt-5.1-mini
python src/nl2state.py --train ./data/train --N 20 --N_train 4 --N_validation 10 --domain ferry --out nl2state_result.json --model gemini --model_name gemini-2.5-flash

   Output files:
   - src/ferry/nl2state_result.json (results)
   - src/ferry/ferry_optimized_module.json (optimized DSPy module)
   - src/ferry/ferry_optimized_module_prompt.txt (human-readable prompt)

2. Command line - Load pre-optimized module:
python src/nl2state.py --train ./data/train --N 20 --domain ferry --out nl2state_result.json --model gemini --model_name gemini-2.5-flash --load_module ferry_optimized_module.json

3. Programmatic usage - Train and Save:
from nl2state import NL2StateProcessor

# Initialize processor
processor = NL2StateProcessor(domain="ferry", model="gpt5.1", model_name="gpt-5.1-mini")

# Load and optimize with validation
validation_results = processor.load_and_optimize(
    train_path="./data/train", 
    n_train=4, 
    n_validation=10
)

# Save the optimized module for later use
processor.save_optimized_module("ferry_optimized_module.json")

# Process training examples
train_results = processor.process_training_examples(
    train_path="./data/train", 
    n_examples=20
)

# Save results
processor.save_results(train_results, validation_results, "results.json")

4. Programmatic usage - Load and Use:
from nl2state import NL2StateProcessor

# Initialize processor
processor = NL2StateProcessor(domain="ferry", model="gpt5.1", model_name="gpt-5.1-mini")

# Load previously optimized module (skip optimization step)
processor.load_optimized_module("ferry_optimized_module.json")

# Process single example
result = processor.process_example(
    context="Ferry is at l1 with car c0 on board. Car c1 is at l0.",
    inputs="Is it possible to reach a state where car c0 is at l0?",
    example_id="custom_1"
)

# Process training examples
train_results = processor.process_training_examples(
    train_path="./data/train", 
    n_examples=20
)
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple, Any
import dspy
from dspy.teleprompt import BootstrapFewShot
from pathlib import Path
import dotenv
dotenv.load_dotenv()

# Import domain descriptions from utils
from src.utils import DOMAIN_DESCRIPTIONS


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
        # Disable think mode for models that have built-in thinking capabilities
        result = self.generate(
            domain_description=domain_description,
            context_and_question=context_and_question,
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
        base_url = os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL')
        if base_url:
            lm = dspy.LM(f'openai/{model_name}', api_key=openai_api_key, temperature=1.0, max_tokens=64000, cache=False, api_base=base_url)
        else:
            lm = dspy.LM(f'openai/{model_name}', api_key=openai_api_key, temperature=1.0, max_tokens=64000, cache=False)
        print(f"✓ Configured GPT model ({model_name})")
        has_built_in_thinking = False  # GPT does have built-in thinking mode

    elif model_type.lower() == "gemini":
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY', 'GEMINI_API_KEY')
        lm = dspy.LM(f'gemini/{model_name}', api_key=gemini_api_key, max_tokens=32000,cache=False)
        print(f"✓ Configured Gemini model ({model_name})")
        has_built_in_thinking = False  # Gemini doesn't have built-in thinking mode
    elif model_type.lower() == "rits":
        rits_url = os.getenv('RITS_API_URL', None) 
        rits_key = os.getenv('RITS_API_KEY', None) 
    
        lm = dspy.LM(f"openai/{model_name}",api_key="NotRequired",api_base=f"{rits_url}/v1",headers={"RITS_API_KEY": rits_key}, max_tokens=64000)
        print(f"✓ Configured RITS model ({model_name})")
        has_built_in_thinking = False # Only for OSS

    elif model_type.lower() == "grok":
        xai_api_key = os.getenv('XAI_API_KEY')
        if not xai_api_key:
             raise ValueError("XAI_API_KEY not found in environment variables")
        xai_api_key = xai_api_key.strip()
        lm = dspy.LM(f'openai/{model_name}', api_key=xai_api_key, api_base="https://api.x.ai/v1")
        print(f"✓ Configured Grok model ({model_name})")
        has_built_in_thinking = False
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'gpt5.1' or 'gemini'")
    
    dspy.configure(lm=lm)
    return not has_built_in_thinking  # Return False for thinking models (disable DSPy thinking)



def load_test_data(domain: str, src_path: str) -> List[Dict[str, Any]]:
    """Load test data from the nl2state_tests.json file"""
    test_file = os.path.join(src_path, domain, "nl2state_tests.json")
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    # Extract examples from the domain key
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
    
    # Randomly sample n_examples
    if len(data) > n_examples:
        data = random.sample(data, n_examples)
    
    return data


def convert_training_to_examples(training_data: List[Dict[str, Any]], domain: str) -> List[dspy.Example]:
    """Convert training data to DSPy examples for optimization"""
    examples = []
    
    # Get domain description from utils.py
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, f"This is a {domain} planning domain.")
    
    for item in training_data:
        # Use context as input
        context = item.get('context', '')
        question = item.get('inputs', '')
        context_and_question = f"{context}\n{question}"
        
        # For training examples, we need to create mock initial and goal states
        # Since training data doesn't have the exact format, we'll create simplified versions
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


def convert_test_to_examples(test_data: List[Dict[str, Any]], domain: str) -> List[dspy.Example]:
    """Convert test data to DSPy examples"""
    examples = []
    
    # Get domain description from utils.py instead of using the one in test data
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, f"This is a {domain} planning domain.")
    
    for item in test_data:
        context_and_question = item.get('context_and_question', '')
        example_id = item.get('example_id', None)
        
        # Convert output states to JSON strings
        initial_state = json.dumps(item['output']['initial_state']['state'])
        goal_state = json.dumps(item['output']['goal_state']['state'])
        
        example = dspy.Example(
            example_id=example_id,
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
        # Check if prediction has both required outputs
        pred_initial = prediction.initial_state.strip()
        pred_goal = prediction.goal_state.strip()
        
        # Simple scoring based on output quality
        score = 0
        if pred_initial and len(pred_initial) > 10:
            score += 0.5
        if pred_goal and len(pred_goal) > 10:
            score += 0.5
            
        return score
    except:
        return 0


def optimize_prompt(train_examples: List[dspy.Example], n_train: int, enable_thinking: bool = True) -> NL2StateModule:
    """Optimize the prompt using DSPy's BootstrapFewShot"""
    
    # Use only n_train examples for optimization
    if len(train_examples) > n_train:
        optimization_examples = random.sample(train_examples, n_train)
    else:
        optimization_examples = train_examples
    
    print(f"🔧 Optimizing prompt with {len(optimization_examples)} examples...")
    print(f"🧠 Thinking mode: {'Enabled' if enable_thinking else 'Disabled'}")
    
    # Configure the optimizer
    config = dict(max_bootstrapped_demos=min(4, len(optimization_examples)), max_labeled_demos=min(2, len(optimization_examples)))
    
    optimizer = BootstrapFewShot(
        metric=metric_function,
        **config
    )
    
    # Initialize and optimize the module
    nl2state_module = NL2StateModule(enable_thinking=enable_thinking)
    
    try:
        optimized_module = optimizer.compile(nl2state_module, trainset=optimization_examples)
        print("✅ Prompt optimization completed!")
        return optimized_module
    except Exception as e:
        print(f"⚠️  Prompt optimization failed: {e}")
        print("📝 Using base module without optimization")
        return nl2state_module

class NL2StateProcessor:
    """Main class for Natural Language to State Conversion using DSPy"""
    
    def __init__(self, domain: str, model: str = "gpt5.1", model_name: str = "gpt-5.1-mini", 
                 src_path: str = "./src", enable_thinking: bool = None):
        """Initialize the NL2State processor
        
        Args:
            domain: Domain name (e.g., ferry, blocksworld)
            model: Model type (gpt5.1 or gemini)
            model_name: Model name
            src_path: Path to source directory containing test data
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
            for d in DOMAIN_DESCRIPTIONS.keys():
                print(f"   - {d}")
        
        # Setup model
        self.enable_thinking = setup_model(model, model_name)
        if enable_thinking is not None:
            self.enable_thinking = enable_thinking
            print("🧠 Thinking mode explicitly overridden")
        
        print(f"✅ NL2StateProcessor initialized for domain '{domain}' with {model}/{model_name}")
    
    def load_and_optimize(self, train_path: str, n_train: int, n_validation: int = 0):
        """Load test data and optimize the prompt
        
        Args:
            train_path: Path to training data directory (not used for optimization, just for context)
            n_train: Number of examples from nl2state_tests.json for optimization
            n_validation: Number of examples for validation (optional)
        
        Returns:
            dict: Validation results if n_validation > 0, None otherwise
        """
        try:
            # Load test data for optimization
            print(f"📥 Loading test data for domain '{self.domain}'...")
            test_data = load_test_data(self.domain, self.src_path)
            test_examples = convert_test_to_examples(test_data, self.domain)
            print(f"✅ Loaded {len(test_examples)} test examples")
            
            # Split for optimization and validation
            random.shuffle(test_examples)
            optimization_examples = test_examples[:n_train]
            
            print(f"📈 Using {len(optimization_examples)} examples for optimization")
            
            # Optimize the prompt
            self.optimized_module = optimize_prompt(optimization_examples, n_train, self.enable_thinking)
            
            validation_results = None
            if n_validation > 0 and len(test_examples) > n_train:
                validation_data = test_data[n_train:n_train + n_validation]
                validation_examples = test_examples[n_train:n_train + n_validation]
                validation_results = self._evaluate_validation(validation_data, validation_examples)
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Failed to load and optimize: {e}")
            raise
    
    def _evaluate_validation(self, validation_data: List[Dict], validation_examples: List[dspy.Example]) -> Dict:
        """Internal method to evaluate validation set"""
        print(f"🧪 Evaluating on {len(validation_examples)} validation examples...")
        
        validation_results = []
        correct = 0

        for i, (item, example) in enumerate(zip(validation_data, validation_examples)):
            print(f"Processing example {i}/{len(validation_examples)}...", end=' ')
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
        
        accuracy = correct / len(validation_examples) if validation_examples else 0
        print(f"📊 Validation accuracy: {accuracy:.2%}")
        
        return {
            "results": validation_results,
            "accuracy": accuracy
        }
    
    def process_training_examples(self, train_path: str, n_examples: int) -> List[Dict[str, Any]]:
        """Process N examples from training data
        
        Args:
            train_path: Path to training data directory
            n_examples: Number of examples to process
            
        Returns:
            List of results with predicted states only
        """
        if self.optimized_module is None:
            raise ValueError("Model not optimized yet. Call load_and_optimize() first.")
        
        try:
            print(f"📥 Loading {n_examples} examples from train data...")
            train_data = load_training_data(train_path, self.domain, n_examples)
            train_examples = convert_training_to_examples(train_data, self.domain)
            print(f"✅ Loaded {len(train_examples)} train examples")
            
            results = []
            print(f"🔄 Processing {len(train_examples)} training examples...")
            
            for i, (item, example) in enumerate(zip(train_data, train_examples), 1):
                print(f"Processing example {i}/{len(train_examples)}...", end=' ')
                
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
            
            return results
            
        except Exception as e:
            print(f"❌ Failed to process training examples: {e}")
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
            raise ValueError("Model not optimized yet. Call load_and_optimize() first.")
        
        # Get domain description
        domain_desc = DOMAIN_DESCRIPTIONS.get(self.domain, f"This is a {self.domain} planning domain.")
        
        # Combine context and inputs
        context_and_question = f"{context}\n{inputs}"
        
        # Make prediction
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
    
    def save_optimized_module(self, module_file: str = "optimized_module.json"):
        """Save the optimized DSPy module to a file for later use
        
        Args:
            module_file: Output filename for the optimized module
        """
        if self.optimized_module is None:
            raise ValueError("No optimized module to save. Call load_and_optimize() first.")
        
        output_path = f"{self.src_path}/{self.domain}/{module_file}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the module using DSPy's save method
        self.optimized_module.save(output_path)
        print(f"💾 Optimized module saved to {output_path}")
        
        # Also save the prompt in a human-readable text format
        prompt_file = module_file.replace('.json', '_prompt.txt')
        prompt_path = f"{self.src_path}/{self.domain}/{prompt_file}"
        
        with open(prompt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("OPTIMIZED DSPY PROMPT FOR NL2STATE CONVERSION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Domain: {self.domain}\n")
            f.write(f"Model: {self.model}/{self.model_name}\n")
            f.write(f"Thinking Mode: {'Enabled' if self.enable_thinking else 'Disabled'}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("MODULE CONFIGURATION\n")
            f.write("=" * 80 + "\n\n")
            
            # Extract and write the prompt structure
            try:
                # Get the generate predictor from the module
                if hasattr(self.optimized_module, 'generate'):
                    predictor = self.optimized_module.generate
                    f.write(f"Predictor Type: {type(predictor).__name__}\n\n")
                    
                    # CAPTURE ACTUAL PROMPT USING DSPY
                    f.write("=" * 80 + "\n")
                    f.write("ACTUAL PROMPTS SENT TO LLM\n")
                    f.write("=" * 80 + "\n\n")
                    
                    try:
                        import dspy
                        
                        # Create a sample input
                        domain_desc = DOMAIN_DESCRIPTIONS.get(self.domain, f"This is a {self.domain} planning domain.")
                        sample_context = "Sample: Ferry is at location A with car C1 on board. Can we get car C1 to location B?"
                        
                        # Enable history tracking and make a prediction
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
                                f.write(f"\n{'='*80}\n")
                                f.write(f"LM CALL #{idx}\n")
                                f.write(f"{'='*80}\n\n")
                                f.write(json.dumps(entry, indent=2, default=str))
                                f.write("\n\n")
                        else:
                            f.write("No history captured in trace\n\n")
                            
                            # Try alternative: access LM history directly
                            lm = dspy.settings.lm
                            if hasattr(lm, 'history') and lm.history:
                                f.write(f"Found {len(lm.history)} calls in LM history\n\n")
                                last_call = lm.history[-1]
                                f.write("LAST LM CALL:\n")
                                f.write("-" * 80 + "\n")
                                f.write(json.dumps(last_call, indent=2, default=str))
                                f.write("\n\n")
                            else:
                                f.write("No LM history available\n\n")
                    
                    except Exception as e:
                        f.write(f"Could not capture history: {e}\n\n")
                        import traceback
                        f.write(f"Traceback:\n{traceback.format_exc()}\n\n")
                    
                    # Write demonstrations if available
                    if hasattr(predictor, 'demos') and predictor.demos:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write(f"DEMONSTRATIONS ({len(predictor.demos)} examples)\n")
                        f.write("=" * 80 + "\n\n")
                        for i, demo in enumerate(predictor.demos, 1):
                            f.write(f"\n--- Demo {i} ---\n")
                            if hasattr(demo, 'toDict'):
                                f.write(json.dumps(demo.toDict(), indent=2))
                            elif hasattr(demo, '__dict__'):
                                f.write(json.dumps(demo.__dict__, indent=2, default=str))
                            else:
                                f.write(str(demo))
                            f.write("\n")
                    
                    # Write extended signature if available
                    if hasattr(predictor, 'extended_signature'):
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("EXTENDED SIGNATURE\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(str(predictor.extended_signature))
                        f.write("\n")
                    
                    # Write the signature instructions
                    if hasattr(predictor, 'signature'):
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("SIGNATURE INSTRUCTIONS\n")
                        f.write("=" * 80 + "\n\n")
                        sig = predictor.signature
                        if hasattr(sig, 'instructions'):
                            f.write(f"Instructions: {sig.instructions}\n\n")
                        if hasattr(sig, 'input_fields'):
                            f.write("Input Fields:\n")
                            for name, field in sig.input_fields.items():
                                f.write(f"  - {name}: {field.json_schema_extra.get('desc', 'No description')}\n")
                        if hasattr(sig, 'output_fields'):
                            f.write("\nOutput Fields:\n")
                            for name, field in sig.output_fields.items():
                                f.write(f"  - {name}: {field.json_schema_extra.get('desc', 'No description')}\n")
                        f.write("\n")
                    
                    # Write the full predictor state
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("FULL PREDICTOR STATE\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(str(predictor))
                    f.write("\n")
                    
            except Exception as e:
                f.write(f"\n⚠️  Could not extract detailed prompt information: {e}\n")
                import traceback
                f.write(f"\nTraceback:\n{traceback.format_exc()}\n")
                f.write(f"\nModule type: {type(self.optimized_module)}\n")
                f.write(f"Module representation:\n{str(self.optimized_module)}\n")
        
        print(f"📝 Human-readable prompt saved to {prompt_path}")
        return output_path, prompt_path
    
    def load_optimized_module(self, module_file: str = "optimized_module.json"):
        """Load a previously saved optimized module
        
        Args:
            module_file: Path to the saved module file
        """
        module_path = f"{self.src_path}/{self.domain}/{module_file}"
        
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Optimized module file not found: {module_path}")
        
        # Initialize a base module with the same configuration
        base_module = NL2StateModule(enable_thinking=self.enable_thinking)
        
        # Load the saved module
        base_module.load(module_path)
        self.optimized_module = base_module
        
        print(f"✅ Optimized module loaded from {module_path}")
        
        # Check if prompt text file exists and display info
        prompt_file = module_file.replace('.json', '_prompt.txt')
        prompt_path = f"{self.src_path}/{self.domain}/{prompt_file}"
        if os.path.exists(prompt_path):
            print(f"📝 Human-readable prompt available at {prompt_path}")
        
        return self.optimized_module
    
    def save_results(self, train_results: List[Dict] = None, validation_results: Dict = None, 
                     output_file: str = "results.json"):
        """Save results to file
        
        Args:
            train_results: Results from training data processing
            validation_results: Validation results from load_and_optimize
            output_file: Output filename
        """
        output_data = {
            "metadata": {
                "domain": self.domain,
                "model": self.model,
                "model_name": self.model_name,
                "thinking_mode_enabled": self.enable_thinking,
                "total_train_examples_processed": len(train_results) if train_results else 0,
                "total_validation_examples_processed": len(validation_results["results"]) if validation_results else 0,
                "validation_accuracy": f"{validation_results['accuracy']:.2%}" if validation_results else "N/A"
            }
        }
        
        if train_results:
            output_data["train_results"] = train_results
        
        if validation_results:
            output_data["validation_results"] = validation_results["results"]
        
        output_path = f"{self.src_path}/{self.domain}/{output_file}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Natural Language to State Conversion using DSPy")
    parser.add_argument("--train", required=True, help="Path to training data directory")
    parser.add_argument("--N", type=int, required=True, help="Number of examples to convert from natural language to state (from train data)")
    parser.add_argument("--N_train", type=int, default=0, help="Number of examples from nl2state_tests.json to train the DSPy prompt optimizer (0 to skip optimization)")
    parser.add_argument("--N_validation", type=int, default=0, help="Number of examples from nl2state_tests.json to use for validation")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., ferry, blocksworld)")
    parser.add_argument("--out", required=True, help="Output file name for results")
    parser.add_argument("--model", default="openai", choices=["openai", "gemini"], help="Model provider to use")
    parser.add_argument("--model_name", default="gpt-5.1", help="Model name (e.g., gpt-5.1, gemini-2.5-flash)")
    parser.add_argument("--src", default="./src", help="Path to source directory containing test data")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable DSPy thinking mode (auto-disabled for thinking models like Qwen)")
    parser.add_argument("--load_module", type=str, help="Load a previously saved optimized module (e.g., ferry_optimized_module.json) instead of optimizing")

    args = parser.parse_args()

    print("🚀 Starting Natural Language to State Conversion")
    print(f"📁 Training data: {args.train}")
    print(f"🎯 Domain: {args.domain}")
    print(f"📊 Train data examples to process: {args.N}")
    print(f"🏋️  N_train (prompt optimization): {args.N_train}")
    print(f"🧪 N_validation: {args.N_validation}")
    print(f"🤖 Model: {args.model}")
    print(f"📝 Model name: {args.model_name}")
    if args.load_module:
        print(f"📂 Loading pre-optimized module: {args.load_module}")

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
            processor.load_optimized_module(args.load_module)
            print("✅ Pre-optimized module loaded successfully!")
        else:
            if args.N_train == 0:
                raise ValueError("Either provide --load_module to load a pre-optimized module, or set --N_train > 0 to train a new module")
            
            # Load and optimize
            validation_results = processor.load_and_optimize(
                train_path=args.train,
                n_train=args.N_train,
                n_validation=args.N_validation
            )
            
            # Save the optimized module for future use
            print("\n💾 Saving optimized module...")
            module_file = f"{args.domain}_optimized_module.json"
            processor.save_optimized_module(module_file)
        
        # Process training examples
        train_results = processor.process_training_examples(
            train_path=args.train,
            n_examples=args.N
        )
        # for single example processing, you can use:
        # result = processor.process_example(context="Ferry is at l1 with car c0 on board. Car c1 is at l0.", inputs="Is it possible to reach a state where car c0 is at l0?", example_id="custom_1")
        # print(result)
        
        # Save results
        processor.save_results(
            train_results=train_results,
            validation_results=validation_results,
            output_file=args.out
        )
        
        print("🎉 Processing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
