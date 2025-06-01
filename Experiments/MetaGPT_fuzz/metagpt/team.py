#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:30
@Author  : alexanderwu
@File    : team.py
@Modified By: mashenquan, 2023/11/27. Add an archiving operation after completing the project, as specified in
        Section 2.2.3.3 of RFC 135.
"""

import warnings
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from metagpt.actions import UserRequirement
from metagpt.const import MESSAGE_ROUTE_TO_ALL, SERDESER_PATH
from metagpt.context import Context
from metagpt.environment import Environment
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.utils.common import (
    NoMoneyException,
    read_json_file,
    serialize_decorator,
    write_json_file,
)

import argparse
# def rewrite_workspace_config(args):
#     lines = []
#     with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/const.py', 'r+') as f:
#         lines = f.readlines()
#     for i in range(len(lines)):
#         if 'DEFAULT_WORKSPACE_ROOT = METAGPT_ROOT' in lines[i]:
#             sep = ' / '
#             temp = lines[i].split(sep)
#             temp[1] = '"'+ args.workspace+'"\n'
#             lines[i] = sep.join(temp)
#     with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/const.py', 'w+') as f:  
#         f.writelines(lines)
# def rewrite_write_prd_an(args):
#     if 'human' in args.dataset:
#         idx = 0
#     if 'mbpp' in args.dataset:
#         idx = 1
#     if 'codecontest' in args.dataset:
#         idx = 2
#     lines = []
#     with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/actions/write_prd_an.py', 'r+') as f:
#         lines = f.readlines()
#     for i in range(len(lines)):
#         split_str = 'ORIGINAL_REQUIREMENTS_example = ORIGINAL_REQUIREMENTS_examples'
#         if split_str in lines[i]:
#             lines[i] = lines[i][:lines[i].find('[')]+f'[{idx}]'+'\n'
#     with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/actions/write_prd_an.py', 'w+') as f:  
#         f.writelines(lines)
# def rewrite_design_api_an(args):
#     idx = 0
#     if 'codecontest' in args.dataset:
#         idx = 1
#     lines = []
#     with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/actions/design_api_an.py', 'r+') as f:
#         lines = f.readlines()
#     for i in range(len(lines)):
#         split_str = 'instruction = instructions'
#         if split_str in lines[i]:
#             lines[i] = lines[i][:lines[i].find('[')]+f'[{idx}]'+'\n'
#         split_str = 'example = examples'
#         if split_str in lines[i]:
#             lines[i] = lines[i][:lines[i].find('[')]+f'[{idx}]'+'\n'
#     with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/actions/design_api_an.py', 'w+') as f:  
#         f.writelines(lines)
# def rewrite_model_config(args):
    
    # lines = []
    # with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/provider/openai_api.py', 'r+') as f:
    #     lines = f.readlines()
    # for i in range(len(lines)):
    #     if 'self.model = "' in lines[i] and 'V2-Lite' not in lines[i]:
    #         sep = 'self.model = '
    #         temp = lines[i].split(sep)
    #         temp[1] = '"'+ args.model +'"\n'
    #         lines[i] = sep.join(temp)
    # with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/provider/openai_api.py', 'w+') as f:  
    #     f.writelines(lines)

    # lines = []
    # with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/provider/llm_provider_registry.py', 'r+') as f:
    #     lines = f.readlines()
    # for i in range(len(lines)):
    #     if 'config.api_type = LLMType.' in lines[i]:
    #         sep = 'config.api_type = LLMType.'
    #         temp = lines[i].split(sep)
    #         temp[1] = 'AZURE'
    #         if 'deepseek' in args.model:
    #             temp[1] = 'OPENAI'
    #         temp[1] += '\n'
    #         lines[i] = sep.join(temp)
    # with open ('/home/zlyuaj/muti-agent/MetaGPT/metagpt/provider/llm_provider_registry.py', 'w+') as f:  
    #     f.writelines(lines)

class Team(BaseModel):
    """
    Team: Possesses one or more roles (agents), SOP (Standard Operating Procedures), and a env for instant messaging,
    dedicated to env any multi-agent activity, such as collaboratively writing executable code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    args: dict = {}

    env: Optional[Environment] = None
    investment: float = Field(default=10.0)
    idea: str = Field(default="")

    def __init__(self, context: Context = None, **data: Any):
        super(Team, self).__init__(**data)
        ctx = context or Context()
        if not self.env:
            self.env = Environment(context=ctx)
        else:
            self.env.context = ctx  # The `env` object is allocated by deserialization
        if "roles" in data:
            self.hire(data["roles"])
        if "env_desc" in data:
            self.env.desc = data["env_desc"]

    def serialize(self, stg_path: Path = None):
        stg_path = SERDESER_PATH.joinpath("team") if stg_path is None else stg_path
        team_info_path = stg_path.joinpath("team.json")
        serialized_data = self.model_dump()
        serialized_data["context"] = self.env.context.serialize()
        # rewrite_workspace_config(self.args)
        # rewrite_write_prd_an(self.args)
        # rewrite_design_api_an(self.args)
        # rewrite_model_config(self.args)

        write_json_file(team_info_path, serialized_data)

    @classmethod
    def deserialize(cls, stg_path: Path, context: Context = None) -> "Team":
        """stg_path = ./storage/team"""
        # recover team_info
        team_info_path = stg_path.joinpath("team.json")
        if not team_info_path.exists():
            raise FileNotFoundError(
                "recover storage meta file `team.json` not exist, " "not to recover and please start a new project."
            )

        team_info: dict = read_json_file(team_info_path)
        ctx = context or Context()
        ctx.deserialize(team_info.pop("context", None))
        team = Team(**team_info, context=ctx)
        return team

    def hire(self, roles: list[Role]):
        """Hire roles to cooperate"""
        # logger.error('in team.hire')
        self.env.add_roles(roles)

    @property
    def cost_manager(self):
        """Get cost manager"""
        return self.env.context.cost_manager

    def invest(self, investment: float):
        """Invest company. raise NoMoneyException when exceed max_budget."""
        self.investment = investment
        self.cost_manager.max_budget = investment
        logger.info(f"Investment: ${investment}.")

    def _check_balance(self):
        if self.cost_manager.total_cost >= self.cost_manager.max_budget:
            raise NoMoneyException(self.cost_manager.total_cost, f"Insufficient funds: {self.cost_manager.max_budget}")

    def run_project(self, idea, send_to: str = ""):
        """Run a project from publishing user requirement."""
        self.idea = idea

        # Human requirement.
        self.env.publish_message(
            Message(role="Human", content=idea, cause_by=UserRequirement, send_to=send_to or MESSAGE_ROUTE_TO_ALL),
            peekable=False,
        )

    def start_project(self, idea, send_to: str = ""):
        """
        Deprecated: This method will be removed in the future.
        Please use the `run_project` method instead.
        """
        warnings.warn(
            "The 'start_project' method is deprecated and will be removed in the future. "
            "Please use the 'run_project' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_project(idea=idea, send_to=send_to)

    # run in env
    @serialize_decorator
    async def run(self, args, n_round=3, idea="", send_to="", auto_archive=True):

        self.args = args
        # logger.error('in team.run')
        """Run company until target round or no money"""
        if idea:
            self.run_project(idea=idea, send_to=send_to)

        while n_round > 0:
            if self.env.is_idle:
                logger.debug("All roles are idle.")
                break
            n_round -= 1
            self._check_balance()
            # 在env中跑
            await self.env.run(args=args)

            logger.debug(f"max {n_round=} left.")
        self.env.archive(auto_archive)
        return self.env.history
