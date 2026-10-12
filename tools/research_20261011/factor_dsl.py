"""Restricted arithmetic expression compiler for auditable LLM factor proposals."""
import ast
import numpy as np
OPS={ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,ast.Mult:lambda a,b:a*b,
     ast.Div:lambda a,b:a/(b.replace(0,np.nan) if hasattr(b,'replace') else b)}
def evaluate(expression,frame):
    def walk(n):
        if isinstance(n,ast.Expression):return walk(n.body)
        if isinstance(n,ast.Name):
            if n.id not in frame:raise ValueError('Unknown input '+n.id)
            return frame[n.id]
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return n.value
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -walk(n.operand)
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:return OPS[type(n.op)](walk(n.left),walk(n.right))
        raise ValueError('Disallowed expression node: '+type(n).__name__)
    result=walk(ast.parse(expression,mode='eval'))
    return result.replace([np.inf,-np.inf],np.nan) if hasattr(result,'replace') else result
