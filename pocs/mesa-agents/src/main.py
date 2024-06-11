from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa import Agent, Model, datacollection

class RandomWalker(Agent):
    """ An agent which randomly walks around the grid."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.steps_taken = 0

    def step(self):
        """Take a step to a random neighboring cell."""
        self.model.grid.move_agent(self, self.next_step)
        self.next_step = self.random.choice(list(self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)))
        self.steps_taken += 1

class RandomWalkModel(Model):
    """A model with randomly walking agents."""
    def __init__(self, N, width, height):
        super().__init__()  # Initialize the superclass
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.datacollector = datacollection.DataCollector(
            agent_reporters={"Steps": "steps_taken"}
        )

        # Create agents
        for i in range(self.num_agents):
            a = RandomWalker(i, self)

            # Place agent on the grid
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

            # Now that the agent has a position, set its next_step
            a.next_step = self.random.choice(list(self.grid.get_neighborhood(a.pos, moore=True, include_center=False)))

            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.datacollector.collect(self)

from mesa.visualization.modules import TextElement
from mesa.visualization.ModularVisualization import ModularServer

class StepsElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Steps taken: " + ", ".join(str(agent.steps_taken) for agent in model.schedule.agents)

model_params = {
    "N": 10,
    "width": 10,
    "height": 10
}

server = ModularServer(RandomWalkModel, [StepsElement()], "Random Walk Model", model_params)
server.launch()