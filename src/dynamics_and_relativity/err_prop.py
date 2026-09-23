import numpy as np
import polars as pl
import sympy as sp
from sympy import Basic


class Formula:
    def __init__(self,formula: str|sp.Expr, vars: list[str], consts: dict[str, float] | None) -> None:
        """
            Creert een formule object.
            vars zijn de meetwaardes die per keer veranderen.
            consts zijn de constanten die niet veranderen.
        """
        self.symbols = {
            name : sp.Symbol(name)
            for name in list((consts or {}).keys()) + vars
        }
        self.vars = vars
        self.consts = consts or {}
        if isinstance(formula, str):
            self.formula = sp.parse_expr(formula, local_dict=self.symbols)
            self._formula_str = formula
        elif isinstance(formula, sp.Expr):
            self.formula = formula
            self._formula_str = str(formula)
        else:
            raise TypeError(f"Unsupported formula type: {type(formula)}")
    def __str__(self):
        return self._formula_str + "\n vars:" + self.vars.__str__() + "\n consts:" + self.consts.__str__()


    def get_error_expr(self, vars:list[str]|None = None) -> Formula:
        """
            Gebasseerd op de foutenleer reader:
                sigma_f**2 = (df/dx*delta_x)**2 + (df/dy*delta_y)**2 ...etc
            De functie neemt voor alle vars de partiele afgeleide keer de fout en doet dit in het kwadraad.
            De partiele afgeleiden worden opgesomt en hiervan wordt het kwadraat genomen om sigma_f te krijgen.
        """
        err_sqrd = 0
        delta_vars: dict[str,sp.Symbol] = {}
        variables = vars or self.vars
        for var in variables:
            symbol = self.symbols[var]
            delta = sp.Symbol(f"delta_{var}")
            derivative = sp.diff(self.formula, symbol)
            err_sqrd += (derivative * delta) ** 2
            delta_vars[f"delta_{var}"] =delta

        err: sp.Expr = sp.sqrt(err_sqrd)
        err = err.simplify()
        return Formula(err,list(delta_vars.keys()) + self.vars,consts = self.consts)



    def update_formula(self, new_str:str):
        self.formula = sp.parse_expr(new_str, local_dict=  self.symbols)
        self._formula_str = new_str

    def update_vars(self, vars:list[str]):
        self.vars = vars
        self.symbols = {
            name : sp.Symbol(name)
            for name in list((self.consts or {}).keys()) + vars
        }
        self.formula :sp.Expr= sp.parse_expr(self._formula_str, local_dict = self.symbols)

    def evaluate(self, var_values: dict[str,float]):
        varsdict:dict[Basic, Basic | float] = {
            self.symbols[name] : value
            for name, value in var_values.items() if name in self.symbols
        }
        constsdict:dict[Basic, Basic | float] = {
            self.symbols[name] : value
            for name, value in self.consts.items() if name in self.symbols
        }
        symbolsdict: dict[Basic, Basic | float] = varsdict | constsdict
        return self.formula.evalf(subs= symbolsdict)

    def evaluate_polars(self, df: pl.DataFrame, result_name="result"):
        expr = self.formula.subs({
            self.symbols[name]: value
            for name, value in self.consts.items()
        })

        args = [self.symbols[var] for var in self.vars]

        func = sp.lambdify(
            args,
            expr,
            modules=["numpy"]
        )

        arrays = [
            df[var].cast(pl.Float64).to_numpy()
            for var in self.vars
        ]

        values = func(*arrays)

        return pl.Series(result_name, values)


if __name__ == "__main__":
    formula1 = "( x**2 + sin(y)**2 )**0.5"
    form = Formula(formula1, vars=["x","y"], consts={})
    res = form.evaluate(var_values={"x":10.0,"y":10.0})
    err_func = form.get_error_expr()
    err_func_eval = err_func.evaluate(var_values={
        "x":10,
        "delta_x":2.0,
        "y":10.0,
        "delta_y":0.5,
    })
    print(err_func_eval)
