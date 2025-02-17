import unittest
from gen_ai_tool_decorators import llm_tool, parameter, output_label
from oci.generative_ai_inference.models import CohereToolCall, CohereToolResult, CohereTool, FunctionDefinition, FunctionCall, ToolMessage

def mock_function(param="mock"):
  return param

class TestLLMTool(unittest.TestCase):

    def test_decorator_is_callable(self):
        decorated_func = (llm_tool("description"))(mock_function)
        self.assertEqual(decorated_func(),"mock", 'Decorated function did not call correctly!')

    def test_decorator_applies_description(self):
        decorated_func = (llm_tool("description"))(mock_function)
        self.assertEqual(decorated_func.description, "description", 'Decorated function did not have description applied.')
        
    def test_description_included_in_cohere_definition(self):
        decorated_func = (llm_tool("description"))(mock_function)
        tool_definiton = decorated_func.get_cohere_tool_definition()
        self.assertIsInstance(tool_definiton, CohereTool)
        self.assertEqual(tool_definiton.description, "description", 'Tool definition did not have description set')
    
    def test_description_included_in_generic_definition(self):
        decorated_func = (llm_tool("description"))(mock_function)
        tool_definiton = decorated_func.get_generic_tool_definition()
        self.assertIsInstance(tool_definiton, FunctionDefinition)
        self.assertEqual(tool_definiton.description, "description", 'Tool definition did not have description set')

    def test_reapplying_decorator_warns(self):
        decorated_func = (llm_tool("description"))(mock_function)
        with self.assertWarns(Warning):
            redecorated_func = (llm_tool("description2"))(decorated_func)
            
class TestParameter(unittest.TestCase):

    def test_decorator_adds_parameter(self):
        decorated_func = (parameter("name", str, "description"))(mock_function)
        self.assertEqual(decorated_func.parameter_definitions["name"], {
            "type": "str",
            "description": "description",
            "required": True,
        }, 'Decorated function did not have parameter assigned.')

    def test_parameter_included_in_cohere_definition(self):
        decorated_func = (parameter("name", str, "description"))(mock_function)
        tool_definiton = decorated_func.get_cohere_tool_definition()
        self.assertEqual(tool_definiton.parameter_definitions["name"].description, "description", 'Tool definition did not have parameter description set')
        self.assertEqual(tool_definiton.parameter_definitions["name"].type, "str", 'Tool definition did not have parameter type set')
        
    def test_parameter_included_in_generic_definition(self):
        decorated_func = (parameter("name", str, "description"))(mock_function)
        tool_definiton = decorated_func.get_generic_tool_definition()
        self.assertEqual(tool_definiton.parameters["properties"]["name"]["description"], "description", 'Tool definition did not have parameter description set')
        self.assertEqual(tool_definiton.parameters["properties"]["name"]["type"], "string", 'Tool definition did not have parameter type set')
        
    def test_multiple_parameters_supported_in_cohere(self):
        decorated_func = (parameter("p1", str, "description"))(mock_function)
        decorated_func = (parameter("p2", str, "description"))(decorated_func)
        tool_definiton = decorated_func.get_cohere_tool_definition()
        self.assertEqual(len(tool_definiton.parameter_definitions), 2, 'Tool definition did not contain two parameters')
    
    def test_multiple_parameters_supported_in_generic(self):
        decorated_func = (parameter("p1", str, "description"))(mock_function)
        decorated_func = (parameter("p2", str, "description"))(decorated_func)
        tool_definiton = decorated_func.get_generic_tool_definition()
        self.assertEqual(len(tool_definiton.parameters["properties"]), 2, 'Tool definition did not contain two parameters')
    
    def test_reapplying_decorator_warns(self):
        decorated_func = (parameter("name", str, "description"))(mock_function)
        with self.assertWarns(Warning):
            redecorated_func = (parameter("name", str, "description2"))(decorated_func)

class TestOutputLabel(unittest.TestCase):

    def test_decorator_applies_description(self):
        decorated_func = (output_label("label"))(mock_function)
        self.assertEqual(decorated_func.output_label, "label", 'Decorated function did not have output label applied.')

    def test_reapplying_decorator_warns(self):
        decorated_func = (output_label("label2"))(mock_function)
        with self.assertWarns(Warning):
            redecorated_func = (output_label("description2"))(decorated_func)
            
class TestCallingBehaviour(unittest.TestCase):
    
    def test_cohere_tool_calling(self):
        decorated_func = (llm_tool("description"))(mock_function)
        decorated_func = (parameter("param", str, "description"))(decorated_func)
        decorated_func = (output_label("label"))(decorated_func)
        tool_call = CohereToolCall(name="hello_world", parameters={"param":"from_tool_call"})
        result = decorated_func.call_with_cohere_tool_call(tool_call)
        self.assertIsInstance(result, CohereToolResult)
        self.assertEqual(result.outputs[0], {"label":"from_tool_call"})
        
    def test_generic_tool_calling(self):
        decorated_func = (llm_tool("description"))(mock_function)
        decorated_func = (parameter("param", str, "description"))(decorated_func)
        decorated_func = (output_label("label"))(decorated_func)
        tool_call = FunctionCall(name="hello_world", id ="chatcmpl-tool-abcd", type="FUNCTION", arguments=f"{{\"param\": \"from_tool_call\"}}")
        result = decorated_func.call_with_generic_tool_call(tool_call)
        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.content[0].text, "{\"label\": \"from_tool_call\"}")
    
    def test_inject_param_from_outside_cohere_tool_call(self):
        decorated_func = (llm_tool("description"))(mock_function)
        decorated_func = (output_label("label"))(decorated_func)
        tool_call = CohereToolCall(name="hello_world", parameters={})
        result = decorated_func.call_with_cohere_tool_call(tool_call, param="injected")
        self.assertIsInstance(result, CohereToolResult)
        self.assertEqual(result.outputs[0], {"label":"injected"})

if __name__ == '__main__':
    unittest.main()