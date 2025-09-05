import os
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class Config:
    """Configuration class for the Resume Analyzer"""
    
    # Model provider: 'huggingface' or 'openai'
    provider: Literal['huggingface', 'openai'] = 'huggingface'
    
    # OpenAI Configuration (commented out by default)
    # api_key: Optional[str] = None
    # openai_model: str = "gpt-3.5-turbo"
    # openai_temperature: float = 0.3
    # openai_max_tokens: int = 1500
    
    # HuggingFace Configuration (active by default)
    hf_model_name: str = "microsoft/DialoGPT-medium"  # Fast and good for conversation
    # Alternative models:
    # "google/flan-t5-large"     # Good for instruction following
    # "microsoft/DialoGPT-large" # Better quality, slower
    # "facebook/blenderbot-400M-distill" # Fast and efficient
    
    hf_temperature: float = 0.3
    hf_max_length: int = 1000
    hf_device: str = "auto"  # Will use GPU if available, CPU otherwise
    
    def __post_init__(self):
        if self.provider == 'openai':
            # Uncomment these lines if you want to use OpenAI
            # if not self.api_key:
            #     raise ValueError("OpenAI API key is required when using OpenAI provider")
            pass
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        provider = os.getenv("MODEL_PROVIDER", "huggingface").lower()
        
        config = cls(provider=provider)
        
        if provider == 'huggingface':
            config.hf_model_name = os.getenv("HF_MODEL_NAME", "microsoft/DialoGPT-medium")
            config.hf_temperature = float(os.getenv("HF_TEMPERATURE", "0.3"))
            config.hf_max_length = int(os.getenv("HF_MAX_LENGTH", "1000"))
            config.hf_device = os.getenv("HF_DEVICE", "auto")
        
        # OpenAI configuration (commented out)
        # elif provider == 'openai':
        #     config.api_key = os.getenv("OPENAI_API_KEY")
        #     config.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        #     config.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
        #     config.openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))
        
        return config