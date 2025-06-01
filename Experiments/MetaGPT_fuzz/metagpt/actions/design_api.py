#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:26
@Author  : alexanderwu
@File    : design_api.py
@Modified By: mashenquan, 2023/11/27.
            1. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
            2. According to the design in Section 2.2.3.5.3 of RFC 135, add incremental iteration functionality.
@Modified By: mashenquan, 2023/12/5. Move the generation logic of the project name to WritePRD.
"""
import json
from pathlib import Path
from typing import Optional

from metagpt.actions import Action, ActionOutput
from metagpt.monitor import monitor_plan
from metagpt.actions.design_api_an import (
    DATA_STRUCTURES_AND_INTERFACES_HUMAN_MBPP,
    DESIGN_API_NODE_CODECONTEST,
    DESIGN_API_NODE_CODECONTEST_4O,
    DESIGN_API_NODE_CODECONTEST_35,
    DESIGN_API_NODE_HUMAN_MBPP,
    PROGRAM_CALL_FLOW,
    REFINED_DATA_STRUCTURES_AND_INTERFACES,
    REFINED_DESIGN_NODE,
    REFINED_PROGRAM_CALL_FLOW,
)
from metagpt.const import DATA_API_DESIGN_FILE_REPO, SEQ_FLOW_FILE_REPO
from metagpt.logs import logger
from metagpt.schema import Document, Documents, Message
from metagpt.utils.mermaid import mermaid_to_file

NEW_REQ_TEMPLATE = """
### Legacy Content
{old_design}

### New Requirements
{context}
"""


class WriteDesign(Action):
    name: str = ""
    i_context: Optional[str] = None
    desc: str = (
        "Based on the PRD, think about the system design, and design the corresponding APIs, "
        "data structures, library tables, processes, and paths. Please provide your design, feedback "
        "clearly and in detail."
    )

    async def run(self, with_messages: Message, schema: str = None):
        # Use `git status` to identify which PRD documents have been modified in the `docs/prd` directory.
        changed_prds = self.repo.docs.prd.changed_files
        # Use `git status` to identify which design documents in the `docs/system_designs` directory have undergone
        # changes.
        changed_system_designs = self.repo.docs.system_design.changed_files

        # For those PRDs and design documents that have undergone changes, regenerate the design content.
        changed_files = Documents()
        for filename in changed_prds.keys():
            doc = await self._update_system_design(filename=filename)
            changed_files.docs[filename] = doc

        for filename in changed_system_designs.keys():
            if filename in changed_files.docs:
                continue
            doc = await self._update_system_design(filename=filename)
            changed_files.docs[filename] = doc
        if not changed_files.docs:
            # logger.info("Nothing has changed.")
            pass
        # Wait until all files under `docs/system_designs/` are processed before sending the publish message,
        # leaving room for global optimization in subsequent steps.
        return ActionOutput(content=changed_files.model_dump_json(), instruct_content=changed_files)

    async def _new_system_design(self, context):
        prd_node=''
        if 'human' in self.args['dataset']:
            prd_node = DESIGN_API_NODE_HUMAN_MBPP
        if 'mbpp' in self.args['dataset']:
            prd_node = DESIGN_API_NODE_HUMAN_MBPP
        if 'codecontest' in self.args['dataset']:
            prd_node = DESIGN_API_NODE_CODECONTEST
            if '35' in self.args['model']:
                prd_node = DESIGN_API_NODE_CODECONTEST_35
            # if '4o' in self.args['model']:
            #     prd_node = DESIGN_API_NODE_CODECONTEST_4O
            # if 'deepseek' in self.args['model']:
            #     prd_node = DESIGN_API_NODE_CODECONTEST_DS 
        # logger.error(type(context))
        # logger.error(context)
        def get_content_from_context(context,key="Original Requirements"):
            try:
                data = json.loads(context)
                original_requirements=data[key]
                return original_requirements
            except:
                print('can not get req from context, try splitting...')
                # 查找原始需求的位置
                try:
                    start_index = context.find(key) + len(key)
                    end_index = context.find('",', start_index)

                    # 提取原始需求内容
                    original_requirements = context[start_index:end_index].strip().replace('\\n', '\n').replace('\\"', '"')

                    return original_requirements
                except:
                    logger.error('can not get original requirement')
                    return ''



        node = await prd_node.fill(context=context, llm=self.llm,args = self.args)

        
        

        # logger.error('node.instruct_content before repair plan')
        # logger.error(node.instruct_content)
        interpreted_plan=''
        if self.args['repair_plan']==1:

            key="Original Requirements"
            original_requirements = get_content_from_context(context,key)

            key='Implementation approach'
            original_plan = get_content_from_context(node.instruct_content.model_dump_json(),key)

            if not original_requirements:
                logger.error('could not find original_requirements')

            if not original_plan:
                logger.error('could not find original_plan')

            # logger.error('original_plan')
            # logger.error(original_plan)      
            # logger.error('original_requirements')
            # logger.error(original_requirements)
            interpreted_plan = await monitor_plan(plan =original_plan,requirement=original_requirements,model=self.args['model'] )
            if type(interpreted_plan)!=str or not interpreted_plan:
                logger.error('fail to generate interpreted_plan, using original plan!')
            # logger.error('interpreted_plan')
            # logger.error(interpreted_plan)
            node.refresh_instruct_content(key=key,changed_content=interpreted_plan)

        
        return node

    async def _merge(self, prd_doc, system_design_doc):
        context = NEW_REQ_TEMPLATE.format(old_design=system_design_doc.content, context=prd_doc.content)
        node = await REFINED_DESIGN_NODE.fill(context=context, llm=self.llm,args = self.args)
        system_design_doc.content = node.instruct_content.model_dump_json()
        return system_design_doc

    async def _update_system_design(self, filename) -> Document:
        prd = await self.repo.docs.prd.get(filename)
        old_system_design_doc = await self.repo.docs.system_design.get(filename)
        
        if not old_system_design_doc:
            system_design = await self._new_system_design(context=prd.content)
            doc = await self.repo.docs.system_design.save(
                filename=filename,
                content=system_design.instruct_content.model_dump_json(),
                dependencies={prd.root_relative_path},
            )
        else:
            doc = await self._merge(prd_doc=prd, system_design_doc=old_system_design_doc)
            await self.repo.docs.system_design.save_doc(doc=doc, dependencies={prd.root_relative_path})
        await self._save_data_api_design(doc)
        await self._save_seq_flow(doc)
        await self.repo.resources.system_design.save_pdf(doc=doc)
        return doc

    async def _save_data_api_design(self, design_doc):
        m = json.loads(design_doc.content)
        data_api_design = m.get(DATA_STRUCTURES_AND_INTERFACES_HUMAN_MBPP.key) or m.get(REFINED_DATA_STRUCTURES_AND_INTERFACES.key)
        if not data_api_design:
            return
        pathname = self.repo.workdir / DATA_API_DESIGN_FILE_REPO / Path(design_doc.filename).with_suffix("")
        await self._save_mermaid_file(data_api_design, pathname)
        # logger.info(f"Save class view to {str(pathname)}")

    async def _save_seq_flow(self, design_doc):
        m = json.loads(design_doc.content)
        seq_flow = m.get(PROGRAM_CALL_FLOW.key) or m.get(REFINED_PROGRAM_CALL_FLOW.key)
        if not seq_flow:
            return
        pathname = self.repo.workdir / Path(SEQ_FLOW_FILE_REPO) / Path(design_doc.filename).with_suffix("")
        await self._save_mermaid_file(seq_flow, pathname)
        # logger.info(f"Saving sequence flow to {str(pathname)}")

    async def _save_mermaid_file(self, data: str, pathname: Path):
        pathname.parent.mkdir(parents=True, exist_ok=True)
        await mermaid_to_file(self.config.mermaid.engine, data, pathname)
