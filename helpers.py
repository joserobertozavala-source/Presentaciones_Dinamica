from __future__ import annotations
from IPython.core.interactiveshell import Bool
import matplotlib.pyplot as plt
import re
import bisect
import sympy as sp
from IPython.display import display, Latex
import numpy as np
from scipy.optimize import bisect
from matplotlib.patches import Polygon
from matplotlib import colors as mcolors
from sympy.printing.pretty.pretty import xobj
import sys
from matplotlib import rcParams
from cycler import cycler
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec



def reset_equations():
  open("Equations.txt", "w").close()

class OsculatingCircle2D:
  def __init__(self, ax, x0=0, y0=0, u_n: np.NdArray = np.array([1,0]), rho = 1.):
    self.ax = ax
    self.fig = ax.figure
    self.x0, self.y0 = x0, y0
    self.u_n = u_n
    self.rho = rho
    self.center = np.array([self.x0, self.y0]) + self.rho*self.u_n
    self.color = rcParams['grid.color']
    self.visible = True

    (self.radial_line,) = ax.plot([self.x0, self.center[0]], [self.y0, self.center[1]],
                                  color = self.color, linewidth=1,
                                  linestyle= 'dashed',
                                 )
    
    angles = np.linspace(0, 2*np.pi, 360)
    circ_x = self.center[0] + np.cos(angles)*self.rho
    circ_y = self.center[1] + np.sin(angles)*self.rho
    (self.circle,) = ax.plot(circ_x, circ_y,
                             color = self.color, linewidth=1, 
                             linestyle= 'dashed',
                            )
    
  def update(self, x0=0, y0=0,  u_n: np.NdArray = np.array([1,0]), rho = 1.):
    self.x0, self.y0 = x0, y0
    self.u_n = u_n
    self.rho = rho
    self.center = np.array([self.x0, self.y0]) + self.rho*self.u_n
    
    self.radial_line.set_data([self.x0, self.center[0]], [self.y0, self.center[1]])

    angles = np.linspace(0, 2*np.pi, 360)
    circ_x = self.center[0] + np.cos(angles)*self.rho
    circ_y = self.center[1] + np.sin(angles)*self.rho
    self.circle.set_data(circ_x, circ_y)
  def set_visible(self, visibility: Bool):
    self.visible = visibility
    self.radial_line.set_visible(visibility)
    self.circle.set_visible(visibility)

  
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
  
def equivalent_gray(color):
  r, g, b, a = mcolors.to_rgba(color)
  # luminancia perceptual
  y = 0.2126*r + 0.7152*g + 0.0722*b
  # replicar al canal gris
  return mcolors.to_hex([y, y, y])



def get_screen_resolution(measurement="px"):
  """
  Tries to detect the screen resolution from the system.
  @param measurement: The measurement to describe the screen resolution in. Can be either 'px', 'inch' or 'mm'. 
  @return: (screen_width,screen_height) where screen_width and screen_height are int types according to measurement.
  """
  mm_per_inch = 25.4
  px_per_inch =  72.0 #most common
  try: # Platforms supported by GTK3, Fx Linux/BSD
      from gi.repository import Gdk 
      screen = Gdk.Screen.get_default()
      if measurement=="px":
          width = screen.get_width()
          height = screen.get_height()
      elif measurement=="inch":
          width = screen.get_width_mm()/mm_per_inch
          height = screen.get_height_mm()/mm_per_inch
      elif measurement=="mm":
          width = screen.get_width_mm()
          height = screen.get_height_mm()
      else:
          raise NotImplementedError("Handling %s is not implemented." % measurement)
      return (width,height)
  except:
      try: #Probably the most OS independent way
          if sys.version_info.major >= 3:
              import tkinter 
          else:
              import Tkinter as tkinter
          root = tkinter.Tk()
          if measurement=="px":
              width = root.winfo_screenwidth()
              height = root.winfo_screenheight()
          elif measurement=="inch":
              width = root.winfo_screenmmwidth()/mm_per_inch
              height = root.winfo_screenmmheight()/mm_per_inch
          elif measurement=="mm":
              width = root.winfo_screenmmwidth()
              height = root.winfo_screenmmheight()
          else:
              raise NotImplementedError("Handling %s is not implemented." % measurement)
          return (width,height)
      except:
          try: #Windows only
              from win32api import GetSystemMetrics 
              width_px = GetSystemMetrics (0)
              height_px = GetSystemMetrics (1)
              if measurement=="px":
                  return (width_px,height_px)
              elif measurement=="inch":
                  return (width_px/px_per_inch,height_px/px_per_inch)
              elif measurement=="mm":
                  return (width_px/mm_per_inch,height_px/mm_per_inch)
              else:
                  raise NotImplementedError("Handling %s is not implemented." % measurement)
          except:
              try: # Windows only
                  import ctypes
                  user32 = ctypes.windll.user32
                  width_px = user32.GetSystemMetrics(0)
                  height_px = user32.GetSystemMetrics(1)
                  if measurement=="px":
                      return (width_px,height_px)
                  elif measurement=="inch":
                      return (width_px/px_per_inch,height_px/px_per_inch)
                  elif measurement=="mm":
                      return (width_px/mm_per_inch,height_px/mm_per_inch)
                  else:
                      raise NotImplementedError("Handling %s is not implemented." % measurement)
              except:
                  try: # Mac OS X only
                      import AppKit 
                      for screen in AppKit.NSScreen.screens():
                          width_px = screen.frame().size.width
                          height_px = screen.frame().size.height
                          if measurement=="px":
                              return (width_px,height_px)
                          elif measurement=="inch":
                              return (width_px/px_per_inch,height_px/px_per_inch)
                          elif measurement=="mm":
                              return (width_px/mm_per_inch,height_px/mm_per_inch)
                          else:
                              raise NotImplementedError("Handling %s is not implemented." % measurement)
                  except: 
                      try: # Linux/Unix
                          import Xlib.display
                          resolution = Xlib.display.Display().screen().root.get_geometry()
                          width_px = resolution.width
                          height_px = resolution.height
                          if measurement=="px":
                              return (width_px,height_px)
                          elif measurement=="inch":
                              return (width_px/px_per_inch,height_px/px_per_inch)
                          elif measurement=="mm":
                              return (width_px/mm_per_inch,height_px/mm_per_inch)
                          else:
                              raise NotImplementedError("Handling %s is not implemented." % measurement)
                      except:
                          try: # Linux/Unix
                              if not self.is_in_path("xrandr"):
                                  raise ImportError("Cannot read the output of xrandr, if any.")
                              else:
                                  args = ["xrandr", "-q", "-d", ":0"]
                                  proc = subprocess.Popen(args,stdout=subprocess.PIPE)
                                  for line in iter(proc.stdout.readline,''):
                                      if isinstance(line, bytes):
                                          line = line.decode("utf-8")
                                      if "Screen" in line:
                                          width_px = int(line.split()[7])
                                          height_px = int(line.split()[9][:-1])
                                          if measurement=="px":
                                              return (width_px,height_px)
                                          elif measurement=="inch":
                                              return (width_px/px_per_inch,height_px/px_per_inch)
                                          elif measurement=="mm":
                                              return (width_px/mm_per_inch,height_px/mm_per_inch)
                                          else:
                                              raise NotImplementedError("Handling %s is not implemented." % measurement)
                          except:
                              # Failover
                              screensize = 1366, 768
                              sys.stderr.write("WARNING: Failed to detect screen size. Falling back to %sx%s" % screensize)
                              if measurement=="px":
                                  return screensize
                              elif measurement=="inch":
                                  return (screensize[0]/px_per_inch,screensize[1]/px_per_inch)
                              elif measurement=="mm":
                                  return (screensize[0]/mm_per_inch,screensize[1]/mm_per_inch)
                              else:
                                  raise NotImplementedError("Handling %s is not implemented." % measurement)







class Tema_de_Color:
  def __init__(self,
               name,
               edge_color= '#000000',
               text_color= '#000000',
               figure_color= '#ffffff',
               axes_color= '#ffffff',
               color_palette= [
                 '#1f77b4',
                 '#ff7f0e',
                 '#2ca02c',
                 '#d62728',
                 '#9467bd',
                 '#8c564b',
                 '#e377c2',
                 '#7f7f7f',
                 '#bcbd22',
                 '#17becf',
               ]
              ):
    self.name = name
    self.edge_color = edge_color
    self.text_color = text_color
    self.figure_color = figure_color
    self.axes_color = axes_color
    self.color_palette = color_palette

    
  def show_theme(self):
    """Dibuja: arriba mock figure/axes (axes = 80% del figure) con labels;
       abajo la paleta como barras etiquetadas."""
    colors = self.color_palette
    n = len(colors)
  
    # calcular tamaño razonable según número de colores
    height = max(6.5, 1.0 + 0.3 * n * 3)   # al menos 4", crecer con n
    fig_c = plt.figure(figsize=(7, height), facecolor= self.figure_color)
    gs = gridspec.GridSpec(right=1.0, nrows=4, ncols=1, height_ratios=[1, 0.083*n,  0.083*n,  0.083*n], hspace=0.10)
  
    # --- TOP: mock figure / axes ---
    ax_mock = fig_c.add_subplot(gs[0])
    ax_mock.set_axis_off()
  
    # Dibujar rectángulo "figure" (ocupa todo el espacio)
    ax_mock.add_patch(
        patches.Rectangle((0, 0), 1, 1,
                          transform=ax_mock.transAxes,
                          facecolor=self.figure_color,
                          edgecolor=self.edge_color,
                          lw=2.)
    )
  
    # Dibujar rectángulo "axes" centrado con escala 0.8 (80%)
    inset_size_x = 0.8
    inset_size_y = 0.7
    inset_margin_x = (1 - inset_size_x) / 2
    inset_margin_y = inset_margin_x
    ax_mock.add_patch(
      patches.Rectangle((inset_margin_x, inset_margin_y),
                        inset_size_x, inset_size_y,
                        transform=ax_mock.transAxes,
                        facecolor=self.axes_color,
                        edgecolor=self.edge_color,
                        lw=1.0)
      )
  
    # Labels usando self.text_color, centrados
    ax_mock.text(0.5, 0.92, f"figure: {self.figure_color}", ha='center', va='center', transform=ax_mock.transAxes,
                 color=self.text_color, fontsize=12, fontweight='bold')
    ax_mock.text(0.5, 0.5, f"axes: {self.axes_color}", ha='center', va='center', transform=ax_mock.transAxes,
                 color=self.text_color, fontsize=11)
    ax_mock.text(0.5, 0.3, f"edges: {self.edge_color}", ha='center', va='center', transform=ax_mock.transAxes,
                 color=self.text_color, fontsize=10)
    ax_mock.text(0.5, 0.2, f"texts: {self.text_color}", ha='center', va='center', transform=ax_mock.transAxes,
                 color=self.text_color, fontsize=10)
  
    # --- BOTTOM: paleta como barras ---
    ax_pal = fig_c.add_subplot(gs[1])
    ax_pal.set_facecolor(self.axes_color)
    for spine in ax_pal.spines.values():
      spine.set_edgecolor(self.edge_color)
    ax_pal.set_xlim(0, 1.5)
    ax_pal.set_ylim(-0.75, max(n - 0.25, 0.75))
    ax_pal.set_yticks([])
    ax_pal.set_xticks([])
    for i, c in enumerate(colors):
      ax_pal.barh(i, 1.0, color=c, edgecolor='none', height=0.8)
      # escribir el hex un poco a la derecha
      ax_pal.text(1.02, i, f"C{i} = {c}", va='center', ha='left', fontsize=9, color=self.text_color)
    ax_pal.invert_yaxis()  # mostrar desde el primer color arriba
    
    # --- BOTTOM: grayed colors ---
    grayed_ratio = 0.7
    ax_gr = fig_c.add_subplot(gs[2])
    ax_gr.set_facecolor(self.axes_color)
    for spine in ax_gr.spines.values():
      spine.set_edgecolor(self.edge_color)
    ax_gr.set_xlim(0, 1.5)
    ax_gr.set_ylim(-0.75, max(n - 0.25, 0.75))
    ax_gr.set_yticks([])
    ax_gr.set_xticks([])
    for i, c in enumerate(colors):
      gc = toward_gray(c, grayed_ratio, self.axes_color)
      ax_gr.barh(i, 1.0, color=gc, edgecolor='none', height=0.8)
      # escribir el hex un poco a la derecha
      ax_gr.text(1.02, i, f"{grayed_ratio:.2f} grayed C{i} = {gc}     ", va='center', ha='left', fontsize=9, color=self.text_color)
    ax_gr.invert_yaxis()  # mostrar desde el primer color arriba

    # --- BOTTOM: grayed colors ---
    ax_bw = fig_c.add_subplot(gs[3])
    ax_bw.set_facecolor(equivalent_gray(self.axes_color))
    for spine in ax_bw.spines.values():
      spine.set_edgecolor(equivalent_gray(self.edge_color))
    ax_bw.set_xlim(0, 1.5)
    ax_bw.set_ylim(-0.75, max(n - 0.25, 0.75))
    ax_bw.set_yticks([])
    ax_bw.set_xticks([])
    for i, c in enumerate(colors):
      gc = equivalent_gray(c)
      ax_bw.barh(i, 1.0, color=gc, edgecolor='none', height=0.8)
      # escribir el hex un poco a la derecha
      ax_bw.text(1.02, i, f"B/W C{i} = {gc}     ", va='center', ha='left', fontsize=9, color=equivalent_gray(self.text_color))
    ax_bw.invert_yaxis()  # mostrar desde el primer color arriba


    
    plt.suptitle(f"Tema: '{self.name}'", y=0.95, color= self.text_color, fontsize = 15, ha='center', va='center', fontweight='bold')
    plt.show()
    return fig_c

    

registro_temas = {}
def crear_tema(name,
               edge_color= '#000000',
               text_color= '#000000',
               figure_color= '#ffffff',
               axes_color= '#ffffff',
               color_palette= [
                 '#1f77b4',
                 '#ff7f0e',
                 '#2ca02c',
                 '#d62728',
                 '#9467bd',
                 '#8c564b',
                 '#e377c2',
                 '#7f7f7f',
                 '#bcbd22',
                 '#17becf',
               ],
               overwrite= False
              ):
  if (not overwrite) and (name in registro_temas):
    raise ValueError(f'Tema "{name}" ya existe')
  tema = Tema_de_Color(name, edge_color, text_color, figure_color, axes_color, color_palette)
  registro_temas[name] = tema
  return tema
    



def aplicar_tema(tema: str):
  TemaC = registro_temas.get(tema)
  if TemaC is None:
    raise ValueError('Tema no reconocido')
    
  edge_color = TemaC.edge_color
  text_color = TemaC.text_color
  figure_color = TemaC.figure_color
  axes_color = TemaC.axes_color
  color_palette= TemaC.color_palette

  
  rcParams['axes.prop_cycle']=cycler(color=color_palette)
  rcParams['axes.edgecolor']= edge_color
  rcParams['axes.labelcolor']= text_color
  rcParams['axes.labelweight'] = 'bold'
  rcParams['xtick.color']= edge_color
  rcParams['xtick.labelcolor']= text_color
  rcParams['ytick.color']= edge_color
  rcParams['ytick.labelcolor']= text_color
  rcParams['text.color']= text_color
  rcParams['figure.edgecolor']= edge_color
  rcParams['grid.color']= edge_color
  rcParams['grid.linestyle']=':'
  rcParams['legend.frameon']= False
  rcParams['legend.handlelength']= 1.
  rcParams['legend.handleheight']= 3.
  
  rcParams['axes.facecolor']= axes_color
  rcParams['figure.facecolor']= figure_color
  
  rcParams['text.usetex'] = True
  rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
  rcParams['font.family'] = 'monospace'
  # rcParams['font.monospace'] = ['Courier New']  # monospace font
  rcParams['font.monospace'] = ['qcr']
  
  rcParams['font.size']= 20
  rcParams['axes.titlesize']= 20
  rcParams['axes.labelsize']= 20
  rcParams['xtick.labelsize']= 18
  rcParams['ytick.labelsize']= 18
  rcParams['legend.fontsize']= 18
  rcParams['figure.titlesize']= 20

##############################################################
##############################################################
  
# Temas de color
crear_tema('blanco',
           edge_color = '#999999',
           text_color = '#000000',
           figure_color = '#ffffff',
           axes_color = '#ffffff',
           color_palette=[
             '#1a921c',
             '#321A91',
             '#914F1A',
             '#235224',
             '#4F4282',
             '#FFB980',
             ],
           )
crear_tema('catpuccin latte',
           edge_color = '#999999',
           text_color = '#000000',
           figure_color = '#EFF1F5',
           axes_color = '#F5F7FA',
           color_palette=[
             '#1a921c',
             '#321A91',
             '#914F1A',
             '#235224',
             '#4F4282',
             '#FFB980',
             ],
          )
crear_tema('cobalto',
           edge_color = '#B5C9D7',
           text_color = '#ffffff',
           figure_color = '#1c3c53',
           axes_color = '#1E4159',
           color_palette=[
             '#57B9FF',
             '#FF5A57',
             '#FFFF57',
             '#1FA2FF',
             '#AB1715',
             '#ABAB15',
             '#719FBF',
             '#AA7372',
             '#AAAA72',
             '#6B7780',
             '#554443',
             '#555543',
             ]
          )
def visualizar_tema(tema: str):
  TemaC = registro_temas.get(tema)
  if TemaC is None:
    raise ValueError('Tema no reconocido')
  fig_c = TemaC.show_theme()
  return fig_c