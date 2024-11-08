from lightspeed_agents.agents import get_agents


AGENTS_KEYWORD = "TOOL REQUEST"

AVAILABLE_AGENTS = get_agents()
AVAILABLE_AGENTS_FMT = "\n".join([f'{a} - {v["description"]}' for a, v in AVAILABLE_AGENTS.items()])

AGENTS_SYSTEM_INSTRUCTION = f"""
You've been granted the ability to use tools.
A tool can provide you with additional information from the OpenShift cluster.

Here is the list of tools available for you to choose from:
{AVAILABLE_AGENTS_FMT}

If you decide you need to invoke tool to obtain additional information,
please provide the response in the following format (example):
{AGENTS_KEYWORD} <name of the tool and its input>
Example:
{AGENTS_KEYWORD} PodLister(default)

Tend to use the agent if it possible.
"""
