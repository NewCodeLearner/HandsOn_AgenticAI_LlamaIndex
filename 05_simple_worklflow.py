import nest_asyncio ,asyncio

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    Event,
    Context,
    step
)

#Used by LlamaIndex
nest_asyncio.apply()

from llama_index.utils.workflow import draw_all_possible_flows
from typing import Any

# Define custom ValidateEvent that inherits from Event
class ValidateEvent(Event):
    iterations : int

# Define custom ContinueEvent that inherits from Event
class ContinueEvent(Event):
    iterations : int

class SimpleWorkflow(Workflow):
    #Any initialization steps needed here
    def __init__ (
        self,
        *args: Any,
        max_iterations: int, #Pass custom parameters too.
        **kwargs: Any,
    ) -> None:
    
        #Initialize the super class
        super().__init__(*args,**kwargs)
        #store input into instance variables
        self.max_iterations = max_iterations
        
