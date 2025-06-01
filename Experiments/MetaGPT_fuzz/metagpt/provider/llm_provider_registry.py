#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19 17:26
@Author  : alexanderwu
@File    : llm_provider_registry.py
"""
from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.provider.base_llm import BaseLLM
from metagpt.logs import log_llm_stream, logger

class LLMProviderRegistry:
    def __init__(self):
        self.providers = {}

    def register(self, key, provider_cls):
        self.providers[key] = provider_cls

    def get_provider(self, enum: LLMType):
        """get provider instance according to the enum"""
        return self.providers[enum]


def register_provider(keys):
    """register provider to registry"""

    def decorator(cls):
        if isinstance(keys, list):
            for key in keys:
                LLM_REGISTRY.register(key, cls)
        else:
            LLM_REGISTRY.register(keys, cls)
        return cls

    return decorator


def create_llm_instance(config: LLMConfig,args:dict) -> BaseLLM:
    """get the default llm provider"""
    # config.api_type = LLMType.AZURE
    # logger.error('in llm_provider_registry')
    # logger.error(args)
    if not args:
        llm = LLM_REGISTRY.get_provider(LLMType.AZURE)(config)
    else:
        if 'gpt' in args['model']:
            llm = LLM_REGISTRY.get_provider(LLMType.AZURE)(config)
        elif 'deepseek' in args['model']:
            llm = LLM_REGISTRY.get_provider(LLMType.OPENAI)(config) 
        else:
            raise NotImplementedError
    if llm.use_system_prompt and not config.use_system_prompt:
        # for models like o1-series, default openai provider.use_system_prompt is True, but it should be False for o1-*
        llm.use_system_prompt = config.use_system_prompt
    return llm


# Registry instance
LLM_REGISTRY = LLMProviderRegistry()
