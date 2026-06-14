"""
Module formulating and solving the Gurzhi (electronic Navier-Stokes) equation.
Utilizes Mixed Finite Elements (Taylor-Hood P2-P1) for the saddle-point problem.
"""
import logging
from dolfin import *


class GurzhiSolver:
    def __init__(self, mesh: Mesh, D_nu: float, inflow_J: float, L: float, W: float):
        self.mesh = mesh
        self.D_nu = Constant(D_nu)
        self.inflow_J = Constant((inflow_J, 0.0))
        self.L = L
        self.W = W

        # Setup przestrzeni funkcyjnych
        self.W_space = self._create_function_space()
        self.bcs = self._define_boundary_conditions()

    def _create_function_space(self):
        # Elementy Taylor-Hood dla stabilności warunku LBB (Ladyzhenskaya-Babuška-Brezzi)
        V_element = VectorElement("CG", self.mesh.ufl_cell(), 2)  # Gęstość prądu J (predkość)
        P_element = FiniteElement("CG", self.mesh.ufl_cell(), 1)  # Potencjał zrelaksowany phi (ciśnienie)
        W_element = MixedElement([V_element, P_element])
        return FunctionSpace(self.mesh, W_element)

    def _define_boundary_conditions(self):
        L = self.L

        # Identyfikacja brzegów
        def inlet(x, on_boundary): return on_boundary and near(x[0], -L / 2.0)

        def outlet(x, on_boundary): return on_boundary and near(x[0], L / 2.0)

        def walls(x, on_boundary):
            # Ściany to wszystko co nie jest wejściem i wyjściem
            return on_boundary and not (near(x[0], -L / 2.0) or near(x[0], L / 2.0))

        W0 = self.W_space.sub(0)  # Podprzestrzeń wektorowa J
        W1 = self.W_space.sub(1)  # Podprzestrzeń skalarna phi

        # Warunki Dirichleta
        bc_in = DirichletBC(W0, self.inflow_J, inlet)
        bc_wall = DirichletBC(W0, Constant((0.0, 0.0)), walls)  # No-slip (lepkość dominuje)
        bc_out = DirichletBC(W1, Constant(0.0), outlet)  # Referencyjny punkt potencjału

        return [bc_in, bc_wall, bc_out]

    def solve(self):
        logging.info("Formulating the Weak Form of the Gurzhi Equation...")

        # Funkcje próbne i testowe
        (J, phi) = TrialFunctions(self.W_space)
        (v, q) = TestFunctions(self.W_space)

        # Sformułowanie słabe (Wariacyjne)
        # Całkowanie przez części operatora Laplace'a: -D_nu^2 * Laplace(J) -> D_nu^2 * inner(grad(J), grad(v))
        a = (
                inner(J, v) * dx
                + self.D_nu ** 2 * inner(grad(J), grad(v)) * dx
                - phi * div(v) * dx
                - q * div(J) * dx
        )
        L_form = inner(Constant((0.0, 0.0)), v) * dx

        # Rozwiązanie
        w = Function(self.W_space)
        logging.info("Solving the linear system...")
        solve(a == L_form, w, self.bcs)

        J_res, phi_res = w.split()
        return J_res, phi_res