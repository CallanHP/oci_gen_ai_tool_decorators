import json
from typing import Callable, TypeVar
from warnings import warn
from oci.generative_ai_inference.models import CohereTool, CohereParameterDefinition, CohereToolCall, CohereToolResult, FunctionDefinition, FunctionCall, ToolMessage, TextContent

T = TypeVar("T")

JSON_SCHEMA_TYPE_MAPPINGS = {
    "int":"integer",
    "float":"number",
    "str":"string",
    "bool":"boolean",
    "dict":"object",
    "list":"array",
    "tuple":"array"
}

DEFAULT_OUTPUT_LABEL = "output"

class LLMToolDecorator:
    """Decorator Class for a tool function.

    This Callable class wraps a function which should be an available tool for 
    use with the OCI Gen AI Service. The decorator adds methods to the function
    to enable parameters to be defined and the required objects to be extracted 
    for inclusion in a call to the LLM, and to simplify invocation from a 
    'tool_call' result.
    
    Attributes:
        func (Callable): The function being wrapped
        func_name (str): The name of the function being wrapped
        description (str): A string describing the function
        parameters (dict): A dictionary containing descriptions of the parameters 
            associated with the function as tuples
        output_label (str): A string which sets the name of the attribute 
            containing the method output used in the tool call result.
        

    """
    def __init__(self, func: Callable[..., T]):
        self.func = func
        self.func_name = func.__name__
        self.description = ""
        self.parameter_definitions = {}
        self.output_label = DEFAULT_OUTPUT_LABEL

    def __call__(self, *args, **kwargs) -> T:
        return self.func(*args, **kwargs)
    
    @staticmethod
    def _python_type_to_json_schema_type(type_:str)->str:
        #Default unknown types to strings
        return JSON_SCHEMA_TYPE_MAPPINGS.get(type_, "string")
    
    def call_with_cohere_tool_call(self, tool_call:CohereToolCall, **kwargs) -> CohereToolResult:
        """Calls the underlying method using the parameters supplied in the 
        tool call response from the Gen AI Service.

            The parameters from the tool_call and the kwargs from this call
            are combined and are all supplied to the method as named arguments.
            Arguments supplied in **kwargs are preferred over those from the 
            tool_call, if they have the same name.

            Args:
            tool_call (CohereToolCall): A tool call object from the Gen AI 
                Service which includes a set of named parameters.
            
            Returns:
            CohereToolResult: Containing the result of method call
        """
        tool_call_args = tool_call.parameters.copy()
        tool_call_args.update(kwargs)
        tool_output = dict()
        tool_output[self.output_label] = self.func(**tool_call_args)
        #Transform the output into a list, as this is what the service expects
        return CohereToolResult(call=tool_call, outputs=tool_output if isinstance(tool_output, list) else [tool_output])
    
    def call_with_generic_tool_call(self, tool_call:FunctionCall, **kwargs) -> ToolMessage:
        """Calls the underlying method using the parameters supplied in the 
        tool call response from the Gen AI Service.

            The parameters from the tool_call and the kwargs from this call
            are combined and are all supplied to the method as named arguments.
            Arguments supplied in **kwargs are preferred over those from the 
            tool_call, if they have the same name.

            Args:
            tool_call (ToolCall): A tool call object from the Gen AI 
                Service which includes a set of parameters and a call id.
            
            Returns:
            ToolMessage: Containing the result of method call
        """
        #The tool call arguments are in a JSON object encoded as a string
        tool_call_args = json.loads(tool_call.arguments)
        tool_call_args.update(kwargs)
        tool_result = self.func(**tool_call_args)
        #Tool_result could be of any type - need to coerce to a string using JSON dumps
        result_text = {}
        result_text[self.output_label]=tool_result
        return ToolMessage(content=[TextContent(text=json.dumps(result_text))], tool_call_id=tool_call.id)

    def add_parameter(self, name: str, obj_type: str, description: str="", optional: bool = False):
        """Adds details of a parameter for the method.

            Adds details about a parameter for the tool call, to allow the LLM to
            determine how the tool should be invoked.

            Args:
            name: A string with the parameter name
            obj_type: A python type representing the type expected by the 
                parameter.
            description: A string describing the parameters purpose and expected 
                format or values.
            optional: A boolean stating whether this parameter may be omitted. 
                Default:False
        """
        if self.parameter_definitions.get(name, None) != None:
            warn("Applying the 'parameter' decorator to the same parameter multiple times can result in unexpected state.", SyntaxWarning)
        self.parameter_definitions[name] = {
            "type": obj_type.__name__,
            "description": description,
            "required": not optional,
        }

    def get_cohere_tool_definition(self)->CohereTool:
        """Gets a Cohere tool definition for interacting with Gen AI.

        Uses values added via each of the decorators when they have been 
        supplied.

        Returns:
          CohereTool: A tool definition

        """
        #For Cohere models, we return a CohereTool
        parameters = {}
        for parameter in self.parameter_definitions:
            parameters[parameter] = CohereParameterDefinition()
            parameters[parameter].description = self.parameter_definitions[parameter]["description"]
            parameters[parameter].type = self.parameter_definitions[parameter]["type"]
            parameters[parameter].is_required = self.parameter_definitions[parameter]["required"]
        return CohereTool(
            name=self.func_name,
            description=self.description,
            parameter_definitions=parameters
        )
        
    def get_generic_tool_definition(self)->FunctionDefinition:
        """Gets a Generic tool definition for interacting with Gen AI.

        Uses values added via each of the decorators when they have been 
        supplied.

        Returns:
          FunctionDefinition: A tool definition

        """
        #For a Generic model, we return a FunctionDefinition
        #Need to assemble a JSON Schema for the parameters
        parameters = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {},
            "required":[]
        }
        for parameter in self.parameter_definitions:
            parameters["properties"][parameter] = {}
            parameters["properties"][parameter]["description"] = self.parameter_definitions[parameter]["description"]
            parameters["properties"][parameter]["type"] = self.python_type_to_json_schema_type(self.parameter_definitions[parameter]["type"])
            if self.parameter_definitions[parameter]["required"]:
                parameters["required"].append(parameter)
        return FunctionDefinition(
            name=self.func_name,
            description=self.description,
            parameters = parameters
        )


def parameter(name: str, type_: type, description: str, optional: bool = False):
    """Decorator for a tool function parameter.

    This decorator adds a parameter definition to a tool for use with the OCI 
    Gen AI Service.
    
    Args:
        name (str): The name of the parameter
        type_ (type): The python type of this parameter
        description (str): A string describing the parameter, including 
            default, format, etc.
        optional (bool): Specifies if this parameter is required or not
            default: False
    """
    def decorator(func: Callable[..., T]) -> LLMToolDecorator:
        if isinstance(func, LLMToolDecorator):
            func.add_parameter(name, type_, description, optional)
            return func
        method_decorator = LLMToolDecorator(func)
        method_decorator.add_parameter(name, type_, description, optional)
        return method_decorator
    return decorator

def output_label(label:str):
    """Decorator for the tool output
    
    Tool results in Cohere are of the form:
    [{"output":"result goes here"}]
    This decorator changes how the "output" attribute is labelled. This can 
    aid in how the response is interpreted by the LLM. If not specified, 
    defaults to "output".
    
    Args:
        label (str): A string describing the function
    
    """
    def decorator(func: Callable[..., T]) -> LLMToolDecorator:
        if isinstance(func, LLMToolDecorator):
            if func.output_label is not DEFAULT_OUTPUT_LABEL:
                warn("Applying the 'output_label' decorator to a function multiple times can result in unexpected state.", SyntaxWarning)
            func.output_label = label
            return func
        method_decorator = LLMToolDecorator(func)
        method_decorator.output_label = label
        return method_decorator
    return decorator

def llm_tool(description: str):
    """Decorator for a tool function.

    This decorator declares that this function should be an available tool for 
    use with the OCI Gen AI Service. The decorator adds methods to the function
    to enable definitions to be extracted for inclusion in a call to the LLM, 
    and to simplify invocation from a 'tool_call' result.
    
    Args:
        description (str): A string describing the function
    """
    def decorator(func: Callable[..., T]) -> LLMToolDecorator:
        if isinstance(func, LLMToolDecorator):
            if func.description != "":
                warn("Applying the 'llm_tool' decorator to a function multiple times can result in unexpected state.", SyntaxWarning)
            func.description = description
            return func
        method_decorator = LLMToolDecorator(func)
        method_decorator.description = description
        return method_decorator
    return decorator


# @llm_tool("This method returns an appropriate greeting")
# @parameter("name", str, "The person or object who is to be greeted. Defaults to 'world'", optional=True)
# @parameter("informal", bool, "Determines whether we use formal or informal language")
# @output_label("greeting")
# def hello_world(informal: bool, name: str = "world", **kwargs) -> str:
#     if informal:
#         return "Hi " + name +kwargs.get("suffix", "")
#     return "Hello " + name +kwargs.get("suffix", "")

# test_tool_call = CohereToolCall(name="hello_world", parameters={"informal":True, "name":"Test"})

# test_generic_call = FunctionCall(name="hello_world", id ="chatcmpl-tool-abcd", type="FUNCTION", arguments=f"{{\"informal\": true, \"name\": \"Test\"}}")

# print(hello_world.get_generic_tool_definition())
# print(hello_world.call_with_cohere_tool_call(test_tool_call, suffix="Suffix"))
# print(hello_world.call_with_generic_tool_call(test_generic_call, suffix="Suffix"))