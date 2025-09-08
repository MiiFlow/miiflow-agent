"""Calculator tools for mathematical operations."""

import math
import operator
from typing import Union
from miiflow_llm.core.tools import tool


@tool("add", "Add two numbers")
def add(a: float, b: float) -> float:
    return a + b


@tool("subtract", "Subtract second number from first")
def subtract(a: float, b: float) -> float:
    return a - b


@tool("multiply", "Multiply two numbers")
def multiply(a: float, b: float) -> float:
    return a * b


@tool("divide", "Divide first number by second")
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


@tool("power", "Raise first number to power of second")
def power(base: float, exponent: float) -> float:
    return base ** exponent


@tool("square_root", "Calculate square root")
def square_root(number: float) -> float:
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(number)


@tool("factorial", "Calculate factorial")
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n > 20:
        raise ValueError("Factorial limited to n <= 20")
    return math.factorial(n)


@tool("calculate", "Evaluate mathematical expression")
def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    expression = expression.strip()
    
    if not expression:
        raise ValueError("Empty expression")
   
    dangerous = ['import', '__', 'exec', 'eval', 'open', 'file']
    for pattern in dangerous:
        if pattern in expression.lower():
            raise ValueError(f"Dangerous pattern: {pattern}")
    
    allowed_names = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }
    
    try:
        result = eval(expression, allowed_names)
        
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a number")
        
        if math.isinf(result):
            raise ValueError("Result is infinite")
        if math.isnan(result):
            raise ValueError("Result is NaN")
        
        return result
        
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")
