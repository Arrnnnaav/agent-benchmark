from langchain.tools import tool

@tool # above any function turns it into a proper took obejct that agents can call
def calculator(expression: str) -> str: # takes a string and return and string
    """
    Useful for evaluating matehmatical expressions. 
    Input should be a valid python math expression as a string.
    Example inputs: '2 + 2', '100 * 0.05', '(3.5 * 1000) / 1000000'
    """
    #this docstring is critical as langchain reads it and show it to the agent as the toll description 
    #agent uses this descritopm to decide when to use this toll 
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# eval() executes a python expression. second and thrid arguments are the global and local namespaces 
# it can access {"__builtins__": {}} removes all built-in functons - so the agent cant accedentiall call eval("__import__('os').system('rm -rf /')") in can only do math this is call sanboxing.