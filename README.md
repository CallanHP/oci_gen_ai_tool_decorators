# Tool Decorators for OCI Generative AI

A super simple utility for simplifying generating tool interactions for the OCI Generative AI service.

## Installation
To install the decorators using pip:
```
pip install git+https://github.com/CallanHP/oci_gen_ai_tool_decorators
```

## Basic Usage

### Defining a Tool

A method can be defined as a tool for use with the Generative AI service by attaching decorators:

```python
@llm_tool(description="A simple addition tool.")
@output_label("sum")
@parameter("a", int, "First number to add")
@parameter("b", int, "Second number to add")
def add_numbers(a: int, b: int) -> int:
    """Returns the sum of two numbers."""
    return a + b
```

### Adding the Tool to the Chat Request

The tool definition can be generated based upon the decorators and added to the Generative AI Chat request:

```python
#Cohere:
chat_request = oci.generative_ai_inference.models.CohereChatRequest()
chat_request.tools = [add_numbers.get_cohere_tool_definition()]
# etc...

#Generic/Llama:
chat_request = oci.generative_ai_inference.models.GenericChatRequest()
chat_request.tools = [add_numbers.get_generic_tool_definition()]
# etc...
```

When using multiple tools, typical usage would be closer to:

```python
available_tools = [add_numbers, subtract_numbers, multiply_numbers]
...
chat_request.tools = [tool.get_cohere_tool_definition() for tool in available_tools]
```

### Invoking the Tool with a `tool_call` from the Chat Response

Helper methods are provided to simplify method calls using the `tool_call` results from an LLM invocation.

```python
#Cohere returns a CohereToolCall object in tool_calls
tool_call = CohereToolCall(name="add_numbers", parameters={"a": 2, "b": 3})
result = add_numbers.call_with_cohere_tool_call(tool_call)
#result is a CohereToolResult which can be passed in tool_results in the next request
print(result.outputs)  # Output: [{'sum': 5}]

#Generic/Llama returns a FunctionCall object in tool_calls
tool_call = FunctionCall(name="add_numbers", id="123", arguments='{"a": 2, "b": 3}')
result = add_numbers.call_with_generic_tool_call(tool_call)
#result is a ToolMessage which is added to chat_history in the next request
print(result.content)  # Output: [TextContent(text='{"sum": 5}')]
```

### Injecting Context Parameters

This decorator model is unopinionated, simply adding helper methods to decorated method to facilitate generating the required definitions for the call to Gen AI. This is not intended to provide a complete framework for a 'tool-using agent'. Instead, it is expected that developers maintain control over how methods are actually invoked.

Most use cases will require context parameters which shouldn't be supplied by the LLM, such as DB connections, identity context, etc. These can simply be provided as named arguments when using the `call_with_cohere_tool_call()` or `call_with_generic_tool_call()` helper methods.

For example:

```python
#Tool Definition
@llm_tool(description="Gets the full details of a person by name")
@parameter("person_name", str, "Identifier of the row to read")
def get_person_from_db(db_connection: oracledb.Connection, person_name: str) -> Person:
    """Returns a Person from the DB"""
    #Do stuff...

tool_call = CohereToolCall(name="get_person_from_db", parameters={"person_name": "John Smith"})
#Inject an existing DB connection into the tool_call
person = get_person_from_db.call_with_cohere_tool_call(tool_call, db_connection=connection)
```

Alternatively, the parameters can simply be extracted from the tool call and used to call the method directly, the decorators simply extend the method with the additional functionality for use with the Generative AI service - they don't modify the behaviour of the base method at all.

## API Reference

### Decorators

#### `@llm_tool(description: str)`
Declares a function as an available tool for use with the OCI Gen AI Service.

*   **Parameters:** `description` (str) - A string describing the function.
*   **Returns:** An instance of `LLMToolDecorator`.

Example:
```python
@llm_tool("My tool description")
def my_tool():
    pass
```

#### `@parameter(name: str, type_: type, description: str, optional: bool = False)`
Adds a parameter definition to a tool for use with the OCI Gen AI Service.

*   **Parameters:**
    *   `name` (str) - The name of the parameter.
    *   `type_` (type) - The Python type of this parameter.
    *   `description` (str) - A string describing the parameter, including default, format, etc.
    *   `optional` (bool) - Specifies if this parameter is required or not (default: False).
    *   `item_type` (type) - If the the parameter takes a list, specifies type of the items in the list (default: None).
*   **Returns:** An instance of `LLMToolDecorator`.

Example:
```python
@llm_tool("My tool description")
@parameter("my_param", int, "My parameter description")
def my_tool(my_param: int):
    pass
```

#### `@output_label(label: str)`
Decorator for the tool output.

Tool results in Cohere are of the form:
\[{"output":"result goes here"}\]
This decorator changes how the "output" attribute is labelled. This can aid in how the response is interpreted by the LLM. If not specified, defaults to "output".

*   **Parameters:** `label` (str) - A string describing the output label.
*   **Returns:** An instance of `LLMToolDecorator`.

Example:
```python
@llm_tool("My tool description")
@output_label("custom_output")
def my_tool():
    pass
```

### Class

#### `LLMToolDecorator`
The decorator class for a tool function. When one of the tool decorators is applied, the method becomes an instance of an `LLMToolDecorator`, with the original method available via the `__call__` attribute.

This Callable class wraps a function which should be an available tool for use with the OCI Gen AI Service. The decorator adds methods to the function to enable parameters to be defined and the required objects to be extracted for inclusion in a call to the LLM, and to simplify invocation from a 'tool_call' result.

*   **Attributes:**
    *   `func` (Callable): The function being wrapped.
    *   `func_name` (str): The name of the function being wrapped.
    *   `description` (str): A string describing the function.
    *   `parameter_definitions` (dict): A dictionary containing descriptions of the parameters associated with the function as tuples.
    *   `output_label` (str): A string setting the name of the attribute containing the method output used in the tool call result.
*   **Methods:**
    *   `__call__(*args, **kwargs)`: Calls the underlying method with the provided arguments.
    *   `add_parameter(name: str, obj_type: str, description: str, optional: bool = False, item_type: type = None)`: Adds details of a parameter for the method.
    *   `get_cohere_tool_definition()`: Gets a Cohere tool definition for interacting with Gen AI.
    *   `get_generic_tool_definition()`: Gets a generic tool definition for interacting with Gen AI.
    *   `call_with_cohere_tool_call(tool_call: CohereToolCall, **kwargs)`: Calls the underlying method using the parameters supplied in the tool call response from the Gen AI Service.
    *   `call_with_generic_tool_call(tool_call: FunctionCall, **kwargs)`: Calls the underlying method using the parameters supplied in the tool call response from the Gen AI Service.

Example:
```python
@llm_tool("My tool description")
def my_tool():
    pass

# Get the Cohere tool definition
cohere_tool_def = my_tool.get_cohere_tool_definition()

# Get the generic tool definition
generic_tool_def = my_tool.get_generic_tool_definition()
```
