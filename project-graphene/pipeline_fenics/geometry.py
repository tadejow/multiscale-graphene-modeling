"""
Module responsible for generating the 2D computational domain and FEM mesh.
Recreates the ETHZ experimental geometry (Main channel + Disk pocket).
"""
import logging
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh


class GrapheneDeviceMesh:
    def __init__(self, length: float, width: float, radius: float, cx: float, cy: float, resolution: int):
        self.length = length
        self.width = width
        self.radius = radius
        self.cx = cx
        self.cy = cy
        self.resolution = resolution
        self.mesh = self._build_mesh()

    def _build_mesh(self) -> Mesh:
        logging.info("Constructing constructive solid geometry (CSG)...")
        # Główny kanał
        channel = Rectangle(Point(-self.length / 2.0, 0.0), Point(self.length / 2.0, self.width))
        # Dysk (kieszeń hydrodynamiczna)
        disk = Circle(Point(self.cx, self.cy), self.radius)

        # Unia geometrii
        domain = channel + disk

        logging.info(f"Generating FEM mesh with resolution {self.resolution}...")
        mesh = generate_mesh(domain, self.resolution)
        logging.info(f"Mesh generated: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells.")
        return mesh

    def get_mesh(self) -> Mesh:
        return self.mesh