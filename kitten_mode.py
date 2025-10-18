# kitten_mode.py
from IPython.display import Javascript, display
from IPython import get_ipython

def disable_kitten_mode():
    # Limpiar cualquier hook anterior
    get_ipython().events.callbacks['post_run_cell'].clear()

def enable_kitten_mode():
    """
    Activa el modo gatito flotante en JupyterLab.
    """
    # Limpiar cualquier hook anterior
    get_ipython().events.callbacks['post_run_cell'].clear()

    # Script que define la función del gatito en el documento principal con pool aleatorio
    js = """
    (function() {
        if (!window._kittenModeActive) {
            window._kittenModeActive = true;

            // Lista de gatitos (puedes agregar más URLs)
            const kittens = [
                'https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif',
                'https://media.giphy.com/media/VbnUQpnihPSIgIXuZv/giphy.gif',
                'https://media.giphy.com/media/H4DjXQXamtTiIuCcRU/giphy.gif',
                
            ];

            window.showFloatingKitten = function() {
                const doc = window.parent.document;
                const img = doc.createElement('img');

                // Elegir un gatito aleatorio del pool
                const kitty = kittens[Math.floor(Math.random() * kittens.length)];
                img.src = kitty;

                Object.assign(img.style, {
                    position: 'fixed',
                    top: '10px',
                    right: '10px',
                    width: '100px',
                    borderRadius: '12px',
                    zIndex: 9999,
                    boxShadow: '0 0 10px rgba(0,0,0,0.4)',
                    transition: 'opacity 0.3s ease',
                    opacity: '1'
                });

                doc.body.appendChild(img);
                setTimeout(() => {
                    img.style.opacity = '0';
                    setTimeout(() => img.remove(), 300);
                }, 2000);
            };
        }
    })();
    """
    display(Javascript(js))

    # Registrar el hook para gatito tras cada ejecución de celda
    def show_kitten_after_cell():
        display(Javascript("if (window.parent.showFloatingKitten) window.parent.showFloatingKitten();"))

    get_ipython().events.register('post_run_cell', show_kitten_after_cell)

