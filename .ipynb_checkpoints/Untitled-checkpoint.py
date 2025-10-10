#!/usr/bin/env python
# coding: utf-8

# In[3]:


from IPython.display import IFrame

IFrame(src='Aceleracion_normal_tangente_Q4.slides.slides.html', width='100%', height=600)


# In[8]:


from nbconvert import PythonExporter
import nbformat

nombre = 'Aceleracion_normal_tangente_Q4'
# Cargar el notebook
with open(f"{nombre}.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

# Exportar a código Python
exporter = PythonExporter()
source, _ = exporter.from_notebook_node(nb)

# Guardar como .py
with open(f"{nombre}.py", "w") as f:
    f.write(source)



# In[9]:


Aceleracion_normal_tangente_Q4.py


# In[ ]:




