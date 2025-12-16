from collections import OrderedDict
from typing import Dict
from vllm import AsyncLLMEngine
from vllm.config import VllmConfig
from dataclasses import asdict
from vllm.engine.arg_utils import AsyncEngineArgs


class LLMObjectManager:
    """
    Manages multiple vLLM.AsyncLLMEngine objects to enable multi-model serving on a 
    single GPU by swapping models between GPU (active) and CPU (cached).

    This manager orchestrates the lifecycle of each AsyncLLMEngine instance, using the
    modified sleep/wake methods to control GPU memory usage.
    
    Note: This uses AsyncLLMEngine for asynchronous, online serving with continuous batching.
    """
    def __init__(self, max_active_models: int = 5):
        """
        Initializes the manager.

        Args:
            max_active_models: The maximum number of models to keep active on
                               the GPU at the same time.
        """
        if max_active_models < 1:
            raise ValueError("max_active_models must be at least 1.")

        self.max_active_models = max_active_models
        
        # Registry to hold all created AsyncLLMEngine objects, whether active or sleeping.
        self.llms: Dict[str, AsyncLLMEngine] = {}
        
        # An ordered dictionary to track active LLMs for the LRU (Least
        # Recently Used) eviction policy. The rightmost item is the most
        # recently used.
        self.lru_active_llms = OrderedDict()

    async def add_model(self, model_name: str, engine_args: AsyncEngineArgs):
        """
        Creates and initializes an AsyncLLMEngine object for a new model, then immediately
        puts it to sleep to conserve GPU memory.
        
        Args:
            model_name: A unique name to identify this model instance.
            engine_args: The AsyncEngineArgs configuration object for this model.
        
        Note:
            Multiple AsyncLLMEngine instances CANNOT run in parallel on the same GPU.
            They must be used sequentially with sleep/wake cycles. For parallel request
            processing, use a single engine with continuous batching instead.
        """
        if model_name in self.llms:
            print(f"[Manager] Model '{model_name}' is already managed.")
            return

        print(f"[Manager] Initializing AsyncLLMEngine for '{model_name}'")
        
        # Convert engine_args to dict and set model_tag
        # model_tag is required for sleep/wake functionality
        engine_args_dict = asdict(engine_args)
        engine_args_dict['model_tag'] = model_name
        
        # Recreate AsyncEngineArgs with model_tag
        engine_args_with_tag = AsyncEngineArgs(**engine_args_dict)
        
        # Create the AsyncLLMEngine using from_engine_args() class method
        # This is the canonical initialization method for AsyncLLMEngine
        engine = AsyncLLMEngine.from_engine_args(engine_args_with_tag)
        
        print(f"[Manager] Initialization complete. Caching '{model_name}' to CPU...")

        # Immediately sleep the new engine to free VRAM for the next one.
        # Level 1 sleep offloads weights to CPU and discards the KV cache.
        await engine.sleep(level=1)
        
        self.llms[model_name] = engine
        print(f"[Manager] Model '{model_name}' is now loaded and cached.")

    async def get_llm(self, model_name: str) -> AsyncLLMEngine:
        """
        Retrieves an AsyncLLMEngine object, ensuring it is active on the GPU.

        If the requested model is sleeping, this method will wake it up. If the
        GPU is at full capacity, it will first evict the least recently used
        model to make space.

        Args:
            model_name: The unique name of the model instance to retrieve.
        
        Returns:
            The active vLLM.AsyncLLMEngine object, ready for inference.
        """
        print("-" * 50)
        print(f"[Manager] Request received for AsyncLLMEngine '{model_name}'.")

        if model_name not in self.llms:
            raise ValueError(f"AsyncLLMEngine for '{model_name}' not found. Please add it first.")

        # Case 1: The requested AsyncLLMEngine is already active on the GPU.
        if model_name in self.lru_active_llms:
            # Move it to the end of the queue to mark it as most recently used.
            self.lru_active_llms.move_to_end(model_name)
            print(f"[Manager] '{model_name}' is already active on the GPU.")
            return self.llms[model_name]

        # Case 2: The AsyncLLMEngine is sleeping and needs to be woken up.
        # First, check if we need to evict another AsyncLLMEngine to make space.
        if len(self.lru_active_llms) >= self.max_active_models:
            # Evict the least recently used AsyncLLMEngine (the first item in the OrderedDict).
            lru_model_name, _ = self.lru_active_llms.popitem(last=False)
            print(f"[Manager] GPU capacity of {self.max_active_models} reached. "
                  f"Evicting '{lru_model_name}' to CPU cache.")
            lru_llm_object = self.llms[lru_model_name]
            await lru_llm_object.sleep(level=1)

        # Now, wake up the requested AsyncLLMEngine.
        print(f"[Manager] Waking up '{model_name}' and loading to GPU...")
        llm_to_activate = self.llms[model_name]
        await llm_to_activate.wake_up()
        
        # Add the newly activated AsyncLLMEngine to the LRU tracker.
        self.lru_active_llms[model_name] = True
        print(f"[Manager] '{model_name}' is now active on the GPU.")
        print(f"[Manager] Active models: {list(self.lru_active_llms.keys())}")
        
        return llm_to_activate

    async def sleep_all(self, level: int = 1):
        """
        Puts all currently active AsyncLLMEngine models to sleep to free GPU memory.
        
        Args:
            level: The sleep level to use (default 1).
                   Level 1: Offloads weights to CPU and discards KV cache.
                   Level 2: Full offload (if implemented in vLLM).
        """
        print(f"[Manager] Sleeping all active models (level={level})...")
        
        # Get a list of active model names to avoid modifying dict during iteration
        active_model_names = list(self.lru_active_llms.keys())
        
        for model_name in active_model_names:
            print(f"[Manager] Putting '{model_name}' to sleep...")
            llm_object = self.llms[model_name]
            await llm_object.sleep(level=level)
        
        # Clear the LRU tracker since no models are active anymore
        self.lru_active_llms.clear()
        
        print(f"[Manager] All models are now sleeping. Active models: {list(self.lru_active_llms.keys())}")