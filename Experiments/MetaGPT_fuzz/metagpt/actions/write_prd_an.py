#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/14 11:40
@Author  : alexanderwu
@File    : write_prd_an.py
"""
from typing import List

from metagpt.actions.action_node import ActionNode


# import json
# args={}
# with open('/data/zlyuaj/muti-agent/MetaGPT/metagpt/args.jsonl','r') as f:
#     args = [json.loads(line) for line in f][0]
# ORIGINAL_REQUIREMENTS_example = ''
# if args:
#     if 'human' in args.dataset:


ORIGINAL_REQUIREMENTS_example_human="from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
ORIGINAL_REQUIREMENTS_example_mbpp="Write a function similar_elements to find the shared elements from the given two lists.\nsimilar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)"
ORIGINAL_REQUIREMENTS_example_codecontest = "Andi and Budi were given an assignment to tidy up their bookshelf of n books. Each book is represented by the book title — a string s_i numbered from 1 to n, each with length m. Andi really wants to sort the book lexicographically ascending, while Budi wants to sort it lexicographically descending.\n\nSettling their fight, they decided to combine their idea and sort it asc-desc-endingly, where the odd-indexed characters will be compared ascendingly, and the even-indexed characters will be compared descendingly.\n\nA string a occurs before a string b in asc-desc-ending order if and only if in the first position where a and b differ, the following holds:\n\n  * if it is an odd position, the string a has a letter that appears earlier in the alphabet than the corresponding letter in b; \n  * if it is an even position, the string a has a letter that appears later in the alphabet than the corresponding letter in b. \n\nInput\n\nThe first line contains two integers n and m (1 ≤ n ⋅ m ≤ 10^6).\n\nThe i-th of the next n lines contains a string s_i consisting of m uppercase Latin letters — the book title. The strings are pairwise distinct.\n\nOutput\n\nOutput n integers — the indices of the strings after they are sorted asc-desc-endingly.\n\n-------\nImportant Note: You must follow the input output format. The code will be tested against multiple test cases and all the test cases must be passed.\nExample\n\nInput\n\n\n5 2\nAA\nAB\nBB\nBA\nAZ\n\n\nOutput\n\n\n5 2 1 3 4\n\n\n"


LANGUAGE = ActionNode(
    key="Language",
    expected_type=str,
    instruction="Provide the language used in the project, typically matching the user's requirement language.",
    example="en_us",
)

PROGRAMMING_LANGUAGE = ActionNode(
    key="Programming Language",
    expected_type=str,
    instruction="Python/JavaScript or other mainstream programming language.",
    example="Python",
)

ORIGINAL_REQUIREMENTS_HUMAN = ActionNode(
    key="Original Requirements",
    expected_type=str,
    instruction="Place whole original user's requirements here including every part of the input. If there are quotation marks in user requirement, please use backslashes before quotation marks to avoid decode error",
    example=ORIGINAL_REQUIREMENTS_example_human,
)

ORIGINAL_REQUIREMENTS_MBPP = ActionNode(
    key="Original Requirements",
    expected_type=str,
    instruction="Place whole original user's requirements here including every part of the input. If there are quotation marks in user requirement, please use backslashes before quotation marks to avoid decode error",
    example=ORIGINAL_REQUIREMENTS_example_mbpp,
)

ORIGINAL_REQUIREMENTS_CODECONTEST = ActionNode(
    key="Original Requirements",
    expected_type=str,
    instruction="Place whole original user's requirements here including every part of the input. If there are quotation marks in user requirement, please use backslashes before quotation marks to avoid decode error",
    example=ORIGINAL_REQUIREMENTS_example_codecontest,
)

REFINED_REQUIREMENTS = ActionNode(
    key="Refined Requirements",
    expected_type=str,
    instruction="Place the New user's original requirements here.",
    example="Create a 2048 game with a new feature that ...",
)

PROJECT_NAME = ActionNode(
    key="Project Name",
    expected_type=str,
    instruction='According to the content of "Original Requirements," name the project using snake case style , '
    "like 'game_2048' or 'simple_crm.",
    example="game_2048",
)

PRODUCT_GOALS = ActionNode(
    key="Product Goals",
    expected_type=List[str],
    instruction="Provide up to three clear, orthogonal product goals.",
    example=["Create an engaging user experience", "Improve accessibility, be responsive", "More beautiful UI"],
)

REFINED_PRODUCT_GOALS = ActionNode(
    key="Refined Product Goals",
    expected_type=List[str],
    instruction="Update and expand the original product goals to reflect the evolving needs due to incremental "
    "development. Ensure that the refined goals align with the current project direction and contribute to its success.",
    example=[
        "Enhance user engagement through new features",
        "Optimize performance for scalability",
        "Integrate innovative UI enhancements",
    ],
)

USER_STORIES = ActionNode(
    key="User Stories",
    expected_type=List[str],
    instruction="Provide up to 3 to 5 scenario-based user stories.",
    example=[
        "As a player, I want to be able to choose difficulty levels",
        "As a player, I want to see my score after each game",
        "As a player, I want to get restart button when I lose",
        "As a player, I want to see beautiful UI that make me feel good",
        "As a player, I want to play game via mobile phone",
    ],
)

REFINED_USER_STORIES = ActionNode(
    key="Refined User Stories",
    expected_type=List[str],
    instruction="Update and expand the original scenario-based user stories to reflect the evolving needs due to "
    "incremental development. Ensure that the refined user stories capture incremental features and improvements. ",
    example=[
        "As a player, I want to choose difficulty levels to challenge my skills",
        "As a player, I want a visually appealing score display after each game for a better gaming experience",
        "As a player, I want a convenient restart button displayed when I lose to quickly start a new game",
        "As a player, I want an enhanced and aesthetically pleasing UI to elevate the overall gaming experience",
        "As a player, I want the ability to play the game seamlessly on my mobile phone for on-the-go entertainment",
    ],
)

COMPETITIVE_ANALYSIS = ActionNode(
    key="Competitive Analysis",
    expected_type=List[str],
    instruction="Provide 5 to 7 competitive products.",
    example=[
        "2048 Game A: Simple interface, lacks responsive features",
        "play2048.co: Beautiful and responsive UI with my best score shown",
        "2048game.com: Responsive UI with my best score shown, but many ads",
    ],
)

COMPETITIVE_QUADRANT_CHART = ActionNode(
    key="Competitive Quadrant Chart",
    expected_type=str,
    instruction="Use mermaid quadrantChart syntax. Distribute scores evenly between 0 and 1",
    example="""quadrantChart
    title "Reach and engagement of campaigns"
    x-axis "Low Reach" --> "High Reach"
    y-axis "Low Engagement" --> "High Engagement"
    quadrant-1 "We should expand"
    quadrant-2 "Need to promote"
    quadrant-3 "Re-evaluate"
    quadrant-4 "May be improved"
    "Campaign A": [0.3, 0.6]
    "Campaign B": [0.45, 0.23]
    "Campaign C": [0.57, 0.69]
    "Campaign D": [0.78, 0.34]
    "Campaign E": [0.40, 0.34]
    "Campaign F": [0.35, 0.78]
    "Our Target Product": [0.5, 0.6]""",
)

REQUIREMENT_ANALYSIS = ActionNode(
    key="Requirement Analysis",
    expected_type=str,
    instruction="Provide a detailed analysis of the requirements.",
    example="",
)

REFINED_REQUIREMENT_ANALYSIS = ActionNode(
    key="Refined Requirement Analysis",
    expected_type=List[str],
    instruction="Review and refine the existing requirement analysis into a string list to align with the evolving needs of the project "
    "due to incremental development. Ensure the analysis comprehensively covers the new features and enhancements "
    "required for the refined project scope.",
    example=["Require add ...", "Require modify ..."],
)

REQUIREMENT_POOL = ActionNode(
    key="Requirement Pool",
    expected_type=List[List[str]],
    instruction="List down the top-5 requirements with their priority (P0, P1, P2).",
    example=[["P0", "The main code ..."], ["P0", "The game algorithm ..."]],
)

REFINED_REQUIREMENT_POOL = ActionNode(
    key="Refined Requirement Pool",
    expected_type=List[List[str]],
    instruction="List down the top 5 to 7 requirements with their priority (P0, P1, P2). "
    "Cover both legacy content and incremental content. Retain content unrelated to incremental development",
    example=[["P0", "The main code ..."], ["P0", "The game algorithm ..."]],
)

UI_DESIGN_DRAFT = ActionNode(
    key="UI Design draft",
    expected_type=str,
    instruction="Provide a simple description of UI elements, functions, style, and layout.",
    example="Basic function description with a simple style and layout.",
)

ANYTHING_UNCLEAR = ActionNode(
    key="Anything UNCLEAR",
    expected_type=str,
    instruction="Mention any aspects of the project that are unclear and try to clarify them.",
    example="",
)

ISSUE_TYPE = ActionNode(
    key="issue_type",
    expected_type=str,
    instruction="Answer BUG/REQUIREMENT. If it is a bugfix, answer BUG, otherwise answer Requirement",
    example="BUG",
)

IS_RELATIVE = ActionNode(
    key="is_relative",
    expected_type=str,
    instruction="Answer YES/NO. If the requirement is related to the old PRD, answer YES, otherwise NO",
    example="YES",
)

REASON = ActionNode(
    key="reason", expected_type=str, instruction="Explain the reasoning process from question to answer", example="..."
)


NODES_HUMAN = [
    LANGUAGE,
    PROGRAMMING_LANGUAGE,
    ORIGINAL_REQUIREMENTS_HUMAN,
    PROJECT_NAME,
    PRODUCT_GOALS,
    USER_STORIES,
    COMPETITIVE_ANALYSIS,
    COMPETITIVE_QUADRANT_CHART,
    REQUIREMENT_ANALYSIS,
    REQUIREMENT_POOL,
    UI_DESIGN_DRAFT,
    ANYTHING_UNCLEAR,
]
NODES_MBPP = [
    LANGUAGE,
    PROGRAMMING_LANGUAGE,
    ORIGINAL_REQUIREMENTS_MBPP,
    PROJECT_NAME,
    PRODUCT_GOALS,
    USER_STORIES,
    COMPETITIVE_ANALYSIS,
    COMPETITIVE_QUADRANT_CHART,
    REQUIREMENT_ANALYSIS,
    REQUIREMENT_POOL,
    UI_DESIGN_DRAFT,
    ANYTHING_UNCLEAR,
]

NODES_CODECONTEST= [
    LANGUAGE,
    PROGRAMMING_LANGUAGE,
    ORIGINAL_REQUIREMENTS_CODECONTEST,
    PROJECT_NAME,
    PRODUCT_GOALS,
    USER_STORIES,
    COMPETITIVE_ANALYSIS,
    COMPETITIVE_QUADRANT_CHART,
    REQUIREMENT_ANALYSIS,
    REQUIREMENT_POOL,
    UI_DESIGN_DRAFT,
    ANYTHING_UNCLEAR,
]

REFINED_NODES = [
    LANGUAGE,
    PROGRAMMING_LANGUAGE,
    REFINED_REQUIREMENTS,
    PROJECT_NAME,
    REFINED_PRODUCT_GOALS,
    REFINED_USER_STORIES,
    COMPETITIVE_ANALYSIS,
    COMPETITIVE_QUADRANT_CHART,
    REFINED_REQUIREMENT_ANALYSIS,
    REFINED_REQUIREMENT_POOL,
    UI_DESIGN_DRAFT,
    ANYTHING_UNCLEAR,
]

WRITE_PRD_NODE_HUMAN = ActionNode.from_children("WritePRD", NODES_HUMAN)
WRITE_PRD_NODE_MBPP= ActionNode.from_children("WritePRD", NODES_MBPP)
WRITE_PRD_NODE_CODECONTEST = ActionNode.from_children("WritePRD", NODES_CODECONTEST)
REFINED_PRD_NODE = ActionNode.from_children("RefinedPRD", REFINED_NODES)
WP_ISSUE_TYPE_NODE = ActionNode.from_children("WP_ISSUE_TYPE", [ISSUE_TYPE, REASON])
WP_IS_RELATIVE_NODE = ActionNode.from_children("WP_IS_RELATIVE", [IS_RELATIVE, REASON])
