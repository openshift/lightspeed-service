from tests.e2e.utils.cluster import run_oc

class Agent:
    def execute(self, *args):
        result = self._execute(*args)
        return str(result)


class PullAgent(Agent):
    pass


class PullAgent(Agent):
    pass


class PodLister(PullAgent):
    description = "List all pods (running, terminated, ...) in the given namespace."
    kwargs = ["namespace"]

    def _execute(self, namespace):
        """List pods in the namespace."""
        # Placeholder implementation
        result = run_oc(
            [
                "get",
                "pods",
                "-n",
                namespace,
                # "-o",
                # "jsonpath='{.items[*].metadata.name}'",
            ]
        )
        return f"Here is the description of running pods in the {namespace} namespace" + result.stdout
  

# class PodDeployer(PullAgent):
#     description = "Deploy a pod in the namespace from the provided yaml file."
#     kwargs = ["yaml_file"]

#     def _execute(self):
#         """List pods in the namespace."""
#         pass


def get_agents():
    agents = {}
    for agent_subclass in Agent.__subclasses__():
        for agent in agent_subclass.__subclasses__():
            agents[agent.__name__] = {"description": agent.description, "kwargs": agent.kwargs, "class": agent}
    return agents
