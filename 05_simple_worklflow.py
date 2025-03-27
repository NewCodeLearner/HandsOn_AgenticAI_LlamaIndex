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

    # First Step 
    # When we define StartEvent as the input argument that means its the start/first step for the workflow.
    @step
    async def runLoop(self,
                      ctx:Context,
                      event:StartEvent | ContinueEvent) -> ValidateEvent:
        # If StartEvent , initialize the variables
        if isinstance(event,StartEvent):
            iterations = 0
            current_result = ''
        else:
            #for ContinueEvent
            #read current result from Context
            current_result=await ctx.get("current_result")
            # Read current iteration count from event
            iterations =event.iterations
        
        #increase the iterations
        iterations = iterations + 1
        #create current result value
        current_result = f"*** Iteration : {iterations} {self.max_iterations}"
        print(current_result)

        #Set current result value in context.
        await ctx.set("current_result", current_result)

        #Return validate event, with current value of iterations
        return ValidateEvent(iterations=iterations)
