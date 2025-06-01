#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/12 22:24
@Author  : alexanderwu
@File    : design_api_an.py
"""
from typing import List, Optional

from metagpt.actions.action_node import ActionNode
from metagpt.utils.mermaid import MMC1, MMC2

IMPLEMENTATION_APPROACH = ActionNode(
    key="Implementation approach",
    expected_type=str,
    instruction="Analyze the difficult points of the requirements, select the appropriate open-source framework",
    example="We will ...",
)

REFINED_IMPLEMENTATION_APPROACH = ActionNode(
    key="Refined Implementation Approach",
    expected_type=str,
    instruction="Update and extend the original implementation approach to reflect the evolving challenges and "
    "requirements due to incremental development. Outline the steps involved in the implementation process with the "
    "detailed strategies.",
    example="We will refine ...",
)

PROJECT_NAME = ActionNode(
    key="Project name", expected_type=str, instruction="The project name with underline", example="game_2048"
)

FILE_LIST = ActionNode(
    key="File list",
    expected_type=List[str],
    instruction="Only need relative paths. ALWAYS only write a main.py",
    example=["main.py"],
)

REFINED_FILE_LIST = ActionNode(
    key="Refined File list",
    expected_type=List[str],
    instruction="Update the original file list including only relative paths. Do not add other files."
    "Ensure that the refined file list reflects the evolving structure of the project.",
    example=["main.py"],
)

instruction_codecontest = " List the function signature with type and input output format. Do not implement the function Provide input interface using input(),  DO NOT use stdin, Provide output interface using print(). The data structures SHOULD BE VERY DETAILED."
instruction_codecontest_4o = " List the function with type and input output format. Provide input interface using input(),  DO NOT use stdin, Provide output interface using print(). The data structures SHOULD BE VERY DETAILED."
instruction_codecontest_ds = "Provide input interface using input() to read input, List the function with type and input output format, Provide output interface using print(). The data structures SHOULD BE VERY DETAILED."

# DATA_STRUCTURES_AND_INTERFACES_instruction = instructions[0]
example_human_mbpp=''
example_codecontest = '''
# List the function signature, do not implement 
def func()->:
    pass
# Provide the Interface 
n = input()
result = func(n)
print(result)
'''



instruction_human_mbpp = "List the function with type and input output format. The data structures SHOULD BE VERY DETAILED."
# instruction_codecontest = "Provide input interface using input() to read input, List the function with type and input output format, Provide output interface using print(). The data structures SHOULD BE VERY DETAILED."

instruction_codecontest = " List the function signature with type and input output format. Do not implement the function Provide input interface using input(),  DO NOT use stdin, Provide output interface using print(). The data structures SHOULD BE VERY DETAILED."
instruction_codecontest_35 =  " List the function signature with type and input output format. Do not implement the function Provide input interface using input(),  DO NOT use stdin, Provide output interface using print(). The data structures SHOULD BE VERY DETAILED."


instruction_codecontest_4o = " List the function with type and input output format. Provide input interface using input(),  DO NOT use stdin, Provide output interface using print(). The data structures SHOULD BE VERY DETAILED."
instruction_codecontest_ds = "Provide input interface using input() to read input, List the function with type and input output format, Provide output interface using print(). The data structures SHOULD BE VERY DETAILED."

# DATA_STRUCTURES_AND_INTERFACES_instruction = instructions[0]
example_human_mbpp=''
# example_codecontest = 'def function_name(n: int, m: int, books: List[str]) -> List[int]:\n    # Function to ...\n    # Input: ...\n    # Output: ...\n    \n    # Implementation logic here\n    \n# Input reading and function call\nn, m = map(int, input().split())\nbooks = [input().strip() for _ in range(n)]\nresult = funtion_name(n, m, books)\nprint(*result)"'

example_codecontest = '''
# List the function signature, do not implement 
def func()->:
    pass
# Provide the Interface using input() and print(), DO NOT use stdin
n = input()
result = func(n)
print(result)
'''
example_codecontest_35 = '''
# List the function signature, do not implement 
def func()->:
    pass
# Provide the Interface using input() and print()
n = input()
result = func(n)
print(result)
'''

# # optional,because low success reproduction of class diagram in non py project.
example_codecontest_4o = 'def function_name(n: int, m: int, books: List[str]) -> List[int]:\n    # Function to ...\n    # Input: ...\n    # Output: ...\n    \n    # Implementation logic here\n    \n# Input reading and function call\nn, m = map(int, input().split())\nbooks = [input().strip() for _ in range(n)]\nresult = funtion_name(n, m, books)\nprint(*result)"'
example_codecontest_ds = 'def function_name(n: int, m: int, books: List[str]) -> List[int]:\n    # Function to ...\n    # Input: ...\n    # Output: ...\n    \n    # Implementation logic here\n    \n# Input reading and function call\nn, m = map(int, input().split())\nbooks = [input().strip() for _ in range(n)]\nresult = funtion_name(n, m, books)\nprint(*result)"'

# DATA_STRUCTURES_AND_INTERFACES = ActionNode(
#     key="Data structures and interfaces",
#     expected_type=Optional[str],
#     instruction=DATA_STRUCTURES_AND_INTERFACES_instruction,
#     example=DATA_STRUCTURES_AND_INTERFACES_example,
# )


DATA_STRUCTURES_AND_INTERFACES_HUMAN_MBPP = ActionNode(
    key="Data structures and interfaces",
    expected_type=Optional[str],
    instruction=instruction_human_mbpp,
    example=example_human_mbpp,
)

DATA_STRUCTURES_AND_INTERFACES_CODECONTEST = ActionNode(
    key="Data structures and interfaces",
    expected_type=Optional[str],
    instruction=instruction_codecontest,
    example=example_codecontest,
)

DATA_STRUCTURES_AND_INTERFACES_CODECONTEST_35 = ActionNode(
    key="Data structures and interfaces",
    expected_type=Optional[str],
    instruction=instruction_codecontest_35,
    example=example_codecontest_35,
)

DATA_STRUCTURES_AND_INTERFACES_CODECONTEST_4O = ActionNode(
    key="Data structures and interfaces",
    expected_type=Optional[str],
    instruction=instruction_codecontest_4o,
    example=example_codecontest_4o,
)

DATA_STRUCTURES_AND_INTERFACES_CODECONTEST_DS = ActionNode(
    key="Data structures and interfaces",
    expected_type=Optional[str],
    instruction=instruction_codecontest,
    example=example_codecontest,
)

REFINED_DATA_STRUCTURES_AND_INTERFACES = ActionNode(
    key="Refined Data structures and interfaces",
    expected_type=str,
    instruction="Update and extend the existing mermaid classDiagram code syntax to incorporate new classes, "
    "methods , and functions with precise type annotations. Delineate additional "
    "relationships between classes, ensuring clarity and adherence to PEP8 standards."
    "Retain content that is not related to incremental development but important for consistency and clarity.",
    example='',
)

PROGRAM_CALL_FLOW = ActionNode(
    key="Program call flow",
    expected_type=Optional[str],
    instruction="Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED ABOVE "
    "accurately, covering the CRUD AND INIT of each object, SYNTAX MUST BE CORRECT.",
    example='',
)

REFINED_PROGRAM_CALL_FLOW = ActionNode(
    key="Refined Program call flow",
    expected_type=str,
    instruction="Extend the existing sequenceDiagram code syntax with detailed information, accurately covering the"
    "CRUD and initialization of each object. Ensure correct syntax usage and reflect the incremental changes introduced"
    "in the classes and API defined above. "
    "Retain content that is not related to incremental development but important for consistency and clarity.",
    example='',
)

ANYTHING_UNCLEAR = ActionNode(
    key="Anything UNCLEAR",
    expected_type=str,
    instruction="Mention unclear project aspects, then try to clarify it.",
    example="Clarification needed on third-party API integration, ...",
)

ANYTHING_UNCLEAR_CODECONTEST = ActionNode(
    key="Original Requirements",
    expected_type=str,
    instruction="Place whole original user's requirements here including every part of the input. If there are quotation marks in user requirement, please use backslashes before quotation marks to avoid decode error",
    example="",
)

ANYTHING_UNCLEAR_CODECONTEST_4o = ActionNode(
    key="Original Requirements",
    expected_type=str,
    instruction="Place whole original user's requirements here",
    example="",
)

NODES_HUMAN_MBPP = [
    IMPLEMENTATION_APPROACH,
    # PROJECT_NAME,
    FILE_LIST,
    DATA_STRUCTURES_AND_INTERFACES_HUMAN_MBPP,
    PROGRAM_CALL_FLOW,
    ANYTHING_UNCLEAR,
]

NODES_CODECONTEST = [
    IMPLEMENTATION_APPROACH,
    # PROJECT_NAME,
    FILE_LIST,
    DATA_STRUCTURES_AND_INTERFACES_CODECONTEST,
    PROGRAM_CALL_FLOW,
    ANYTHING_UNCLEAR_CODECONTEST_4o,
]

NODES_CODECONTEST_35 = [
    IMPLEMENTATION_APPROACH,
    # PROJECT_NAME,
    FILE_LIST,
    DATA_STRUCTURES_AND_INTERFACES_CODECONTEST_35,
    PROGRAM_CALL_FLOW,
    ANYTHING_UNCLEAR_CODECONTEST,
]

NODES_CODECONTEST_4O = [
    IMPLEMENTATION_APPROACH,
    # PROJECT_NAME,
    FILE_LIST,
    DATA_STRUCTURES_AND_INTERFACES_CODECONTEST_4O,
    PROGRAM_CALL_FLOW,
    ANYTHING_UNCLEAR,
]

NODES_CODECONTEST_DS = [
    IMPLEMENTATION_APPROACH,
    # PROJECT_NAME,
    FILE_LIST,
    DATA_STRUCTURES_AND_INTERFACES_CODECONTEST_DS,
    PROGRAM_CALL_FLOW,
    ANYTHING_UNCLEAR,
]



REFINED_NODES = [
    REFINED_IMPLEMENTATION_APPROACH,
    REFINED_FILE_LIST,
    REFINED_DATA_STRUCTURES_AND_INTERFACES,
    REFINED_PROGRAM_CALL_FLOW,
    ANYTHING_UNCLEAR,
]

DESIGN_API_NODE_HUMAN_MBPP = ActionNode.from_children("DesignAPI", NODES_HUMAN_MBPP)
DESIGN_API_NODE_CODECONTEST= ActionNode.from_children("DesignAPI", NODES_CODECONTEST)
DESIGN_API_NODE_CODECONTEST_35= ActionNode.from_children("DesignAPI", NODES_CODECONTEST_35)
DESIGN_API_NODE_CODECONTEST_4O= ActionNode.from_children("DesignAPI", NODES_CODECONTEST_4O)
DESIGN_API_NODE_CODECONTEST_DS= ActionNode.from_children("DesignAPI", NODES_CODECONTEST_DS)
REFINED_DESIGN_NODE = ActionNode.from_children("RefinedDesignAPI", REFINED_NODES)
