from __future__ import annotations
from IPython.core.interactiveshell import Bool
import re
import bisect
import sympy as sp
from IPython.display import display, Latex
import numpy as np
from scipy.optimize import bisect
from matplotlib.patches import Polygon
from matplotlib import colors as mcolors
from sympy.printing.pretty.pretty import xobj



def reset_equations():
  open("Equations.txt", "w").close()


class FancyVector2D:
  def __init__(self, ax, x0=0, y0=0, x1=1, y1=0,
                color="C0", linewidth=3.2,
                head_length=30, head_width=20,
                head_tail= 6,
                previous: FancyVector2D | None = None,
                label: str = '', zorder: int = 4,
                ):
    """
    Representa un vector como línea + polígono (punta estilo quiver).
    """
    self.ax = ax
    self.fig = ax.figure
    self.x0, self.y0 = x0, y0
    self.x1, self.y1 = x1, y1
    self.color = color
    self.linewidth = linewidth
    self.head_length = head_length
    self.head_width = head_width
    self.head_tail = head_tail
    self.label = label
    
    self.previous = previous
    if not self.previous is None:
      self.x1 = self.previous.x1 + x1
      self.y1 = self.previous.y1 + y1

    self._updateXY()
    self.visible = True

    # Este atributo permite mostrar/ocultar la leyenda del vector, 
    # independientemente de si el vector es visible o no
    self.main = True

    # Dibujar por encima de todo lo demás
    self.zorder = zorder
    


    # Asegurar que el renderer existe
    self._ensure_renderer()

    # Dibujar cabeza
    head_coords, xb, yb = self._make_arrowhead(x0, y0, x1, y1)
    self.head = Polygon(head_coords, closed=True, color=color, 
                        zorder = self.zorder)
    ax.add_patch(self.head)

    # Dibujar cuerpo
    (self.body_line,) = ax.plot([x0, xb], [y0, yb],
                                color=color, linewidth=linewidth,
                                label= label, zorder = self.zorder)
    
  def _ensure_renderer(self):
    # Forzar a que la figura tenga un renderer válido
    if self.fig.canvas is not None:
      self.fig.canvas.draw_idle()
      self.fig.canvas.flush_events()
      self.fig.canvas.draw()


  def _make_arrowhead(self, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    if L == 0:
      return np.array([[x1, y1]]), x1, y1

    # Dirección unitaria en datos → convertimos a pantalla
    x0_disp, y0_disp = self.ax.transData.transform((x0, y0))
    x1_disp, y1_disp = self.ax.transData.transform((x1, y1))
    dx_disp, dy_disp = x1_disp - x0_disp, y1_disp - y0_disp
    L_disp = np.hypot(dx_disp, dy_disp)

    ux, uy = dx_disp / L_disp, dy_disp / L_disp     # vector unitario
    px, py = -uy, ux                                # perpendicular


    # Coordenadas de la cabeza en pixeles
    xb_disp = x1_disp - self.head_length * ux
    yb_disp = y1_disp - self.head_length * uy
    p1_disp = (xb_disp + self.head_width/2 * px - self.head_tail * ux,
                 yb_disp + self.head_width/2 * py - self.head_tail * uy)
    p2_disp = (xb_disp - self.head_width/2 * px - self.head_tail * ux,
                 yb_disp - self.head_width/2 * py - self.head_tail * uy)  
    coords_disp = np.array([[x1_disp, y1_disp],
                              p1_disp,
                              [xb_disp,
                              yb_disp],
                              p2_disp])

    # Convertir a coordenadas de datos
    coords_data = self.ax.transData.inverted().transform(coords_disp)
    xb, yb = self.ax.transData.inverted().transform((xb_disp, yb_disp))
    return coords_data, xb, yb

  def _updateXY(self):
    if not self.previous is None:
      self.x0, self.y0 = self.previous.x1, self.previous.y1
    self.X = self.x1 - self.x0
    self.Y = self.y1 - self.y0

  def set_end(self, x1, y1):
    """Actualizar las coordenadas finales del vector."""
    self.x1, self.y1 = x1, y1
    self._updateXY()
    # Cabeza
    new_coords, xb, yb = self._make_arrowhead(self.x0, self.y0, self.x1, self.y1)
    self.head.set_xy(new_coords)
    # Línea
    self.body_line.set_data([self.x0, xb], [self.y0, yb])

  def set_start(self, x0, y0):
    """Actualizar las coordenadas iniciales del vector."""
    self.x0, self.y0 = x0, y0
    self.set_end(self.x1, self.y1)  # recalcular todo

  def set_XY(self, X=None, Y=None):
    if X is None:
      X = self.X
    if Y is None:
      Y = self.Y
    if not self.previous is None:
      self.x0, self.y0 = self.previous.x1, self.previous.y1
    self.set_end(self.x0 + X, self.y0 + Y)
    
  def set_head_shape(self, head_length=None, head_width=None, head_tail=None):
    """Cambiar geometría de la cabeza."""
    if head_length is not None:
      self.head_length = head_length
    if head_width is not None:
      self.head_width = head_width
    if head_tail is not None:
      self.head_tail = head_tail
    self.set_end(self.x1, self.y1)  # actualizar

  def set_line_width(self, line_width=None):
    if line_width is not None:
      self.body_line.set_linewidth(line_width)

  def remove(self):
    """Eliminar de la figura."""
    self.body_line.remove()
    self.head.remove()

  def set_label(self, label: str):
    self.label = label
    self.body_line.set_label(label)

  def set_color(self, color):
    self.color = color
    self.body_line.set_color(color)
    self.head.set_color(color)

  def set_visible(self, visibility: Bool):
    self.visible = visibility
    self.body_line.set_visible(visibility)
    self.head.set_visible(visibility)
  
  def set_main(self, main: Bool | None = None):
    if not main is None:
      self.main = main

  def set_zorder(self, zorder: int = 4):
    self.zorder = 4
    self.body_line.set_zorder(zorder)
    self.head.set_zorder(zorder)











def vector_from_offset(OriginVector: Vector, DX, DY, scale= 1., **kwargs):
  NewVector = Vector(OriginVector.ax, OriginVector.x1, OriginVector.y1, 
                      DX*scale + OriginVector.x1, DY*scale + OriginVector.y1, **kwargs)
  return NewVector


def write_latex_equation(eq: str, eq_num: int, 
label: str = '', tag: int = None) -> None:
  Equation = f"Equation {eq_num}\n"

  s = Equation
  if not tag is None:
    s += f"Tag {tag}"
  else:
    s += "(No Tag)"

  if label:
    s += f" -- {label.upper()}:\n\n<a name=\"{label}\"></a>\n"
  else:
    s += " -- (No Label):\n\n\n"
  s += f"$$ {eq} "
  if not tag is None:
    s += f"\\tag{{{tag}}} "
  s += "$$\n\n\n\n\n"
  
  
  with open("Equations.txt", "r") as f:
    lines = f.readlines()
  Equations = []

  # Remove previous versions
  if Equation in lines:
    idx = lines.index(Equation)
    idxs = list(range(idx, idx+10))
    for i, line in enumerate(s.split('\n')[:-1]):
      line_num = idxs[i]
      lines[line_num] = line+'\n'
    
  else:
    # Find all Equations
    for i, line in enumerate(lines):
      m = re.search(r"Equation (\d+)", line)
      if m:
        Equations.append(int(m.group(1)))

    # If this Equation will be last, append
    Is_Empty = not Equations
    if Is_Empty or (not Is_Empty and eq_num > max(Equations)):
      for i, line in enumerate(s.split('\n')[:-1]):
        lines.append(line+'\n')

    # Else, find position
    else:
      next_eq = bisect.bisect_left(Equations, eq_num)
      Next_Equation = f"Equation {Equations[next_eq]}:\n"
      next_idx = lines.index(Next_Equation)
      for i, line in enumerate(s.split('\n')[:-1]):
        lines.insert(next_idx+i, line+'\n')

  with open("Equations.txt", "w") as f:
    for line in lines:
      f.write(line)

def raw_subs(expr, replacements, rationalize=True):
  """
  Sustituye expresiones en expr de manera literal, sin evaluación ni simplificación.
  Usa Add/Mul/Pow con evaluate=False para reconstruir el árbol.
  """
  expr = sp.sympify(expr)

  # Si el nodo entero está en los reemplazos, devolver directo
  if expr in replacements:
    if rationalize:
      return rationalize_floats(replacements[expr])
    return replacements[expr]
  
  # Recursión sobre hijos
  new_args = [raw_subs(arg, replacements, rationalize) for arg in expr.args]

  # Reconstruir el mismo tipo de nodo, sin evaluar
  if isinstance(expr, sp.Add):
    rs = sp.Add(*new_args, evaluate=False)
  elif isinstance(expr, sp.Mul):
    rs = sp.Mul(*new_args, evaluate=False)
  elif isinstance(expr, sp.Pow):
    rs = sp.Pow(*new_args, evaluate=False)
  elif isinstance(expr, sp.Abs):
    rs = sp.Abs(*new_args, evaluate=False)
  else:
    if new_args:
      rs = expr.func(*new_args)
    else:
      rs = expr
  if rationalize:
    return rationalize_floats(rs)
  return rs


def rationalize_floats(expr):
  """
  Recorre una expresión SymPy y convierte todos los Float a Rational,
  manteniendo la estructura (sin expandir/simplificar).
  """
  expr = sp.sympify(expr)
  # Si es un número flotante -> convertir
  if isinstance(expr, sp.Float):
    return sp.nsimplify(expr, rational= True)

  # Si tiene sub-argumentos, procesarlos recursivamente
  if expr.args:
    new_args = [rationalize_floats(a) for a in expr.args]
    return expr.func(*new_args, evaluate=False)

  # Si es átomo (símbolo, entero, etc.)
  return expr

def eq_chain_latex(*symbols):
  return " = ".join(sp.latex(s) for s in symbols)

def add_unit_latex(expr: str, unit: str, brackets: Bool = False) -> str:
  if brackets:
    return expr + " \\mathrm{[\\," + unit + "\\,]}"
  return expr + " \\mathrm{\\," + unit + "}"

def dLatex(expr):
  display(Latex(expr))

def invert_numeric(f , s_target, a=-10, b=10):
  # asume f(a)-s_target y f(b)-s_target tienen signos opuestos
  return bisect(lambda xx: f(xx) - s_target, a, b)

def d_Eq(symbol: str, expr):
  display(sp.Eq(sp.Symbol(symbol), expr))

def Lamb(expr, var):
  return sp.lambdify(var, expr, "numpy")

def toward_gray(color, amount=0.5, background = 'white'):
  """Mezcla el color con gris (1,1,1)"""
  r, g, b, a = mcolors.to_rgba(color)
  rba, gba, bba, aba = mcolors.to_rgba(background)

  gray = np.array([rba, gba, bba])
  rgb = np.array([r, g, b])
  new_rgb = rgb * (1 - amount) + gray * amount
  return mcolors.to_hex(new_rgb)
