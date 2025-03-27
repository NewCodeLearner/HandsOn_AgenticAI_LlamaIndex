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
from typing import Any,Union

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
                      event:Union[StartEvent, ContinueEvent] # UNION :I am using python 3.9 , The | operator is valid for standard Python types in Python 3.10+ (e.g., int | str), or custom classes that don't have metaclasses
                      ) -> ValidateEvent :
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
    
    @step
    async def checkIterations(self,
                              ctx:Context,
                              event:ValidateEvent) -> Union[StopEvent, ContinueEvent]:
        #Read current iteration count from event
        iterations =event.iterations
        #Read max iterations from instance variable
        max_iterations = self.max_iterations
        #Read current result from context and print details
        current_result = await ctx.get("current_result")

        print(f"*** Current iteration to validate :{iterations} {max_iterations}")
        #Perform check if max iterations is reached.
        if iterations > max_iterations:
            #Return stop event if max iterations is reached, with current result
            return StopEvent(result=current_result)
        else:
            #Return continue event with current iteration count
            return ContinueEvent(iterations=iterations)
        
#Draw a workflow graph.
SimpleWorkFlow = SimpleWorkflow(max_iterations=0)
draw_all_possible_flows(SimpleWorkFlow, filename="SimpleWorkflow.html")




