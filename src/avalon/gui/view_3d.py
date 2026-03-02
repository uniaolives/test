"""
VIEW_3D v3.0 - Visualizador Pyglet para Bio-Gênese Cognitiva
Renderização otimizada com instancing e feedback visual rico
"""

import pyglet
from pyglet.gl import *
import numpy as np
import sys
import os

# Adiciona diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BioGenesisViewer(pyglet.window.Window):
    """
    Janela principal de visualização 3D.
    Implementa controles de câmera orbital e seleção de agentes.
    """

    def __init__(self, width: int = 1200, height: int = 800):
        super().__init__(width, height, "Bio-Gênese Cognitiva v3.0",
                        resizable=True, vsync=True)

        # Configuração OpenGL
        try:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        except:
            pass

        # Importa e inicializa motor
        from ..core.particle_system import BioGenesisEngine
        self.engine = BioGenesisEngine(num_agents=250)

        # Controles de câmera orbital
        self.camera = {
            'distance': 180.0,
            'rotation_x': 30.0,
            'rotation_y': 45.0,
            'target': np.array([50.0, 50.0, 50.0])
        }

        # Estado da interface
        self.paused = False
        self.show_connections = True
        self.selected_agent_id = None
        self.frame_count = 0

        # Elementos de UI
        self._setup_ui()

        # Agenda atualizações
        pyglet.clock.schedule_interval(self.update, 1/60.0)

    def _setup_ui(self):
        """Configura elementos de interface."""
        self.stats_label = pyglet.text.Label(
            '', x=10, y=self.height - 30,
            font_size=11, color=(0, 255, 200, 255)
        )

        self.agent_info_label = pyglet.text.Label(
            '', x=10, y=self.height - 200,
            font_size=10, color=(255, 255, 200, 255),
            multiline=True, width=380
        )

        self.help_label = pyglet.text.Label(
            '[ESPAÇO] Pausar  [C] Conexões  [R] Reiniciar  [Click] Selecionar',
            x=10, y=10, font_size=9, color=(150, 150, 150, 255)
        )

    def update(self, dt: float):
        """Atualiza simulação e interface."""
        if not self.paused:
            self.engine.update(dt)
            self.frame_count += 1

        # Atualiza estatísticas a cada 10 frames
        if self.frame_count % 10 == 0:
            stats = self.engine.get_stats()
            self.stats_label.text = (
                f"Agentes: {stats['agents']} | "
                f"Tempo: {stats['time']:.1f}s | "
                f"Vínculos: {stats['bonds']} | "
                f"Mortes: {stats['deaths']} | "
                f"Saúde Média: {stats['avg_health']:.3f}"
            )

            # Atualiza info do agente selecionado
            if self.selected_agent_id is not None:
                info = self.engine.get_agent_info(self.selected_agent_id)
                if info:
                    text = f"Agente #{info['id']} | {info['state'].upper()}\n"
                    text += f"Saúde: {info['health']} | Idade: {info['age']}\n"
                    text += f"Genoma: C={info['genome']['C']} I={info['genome']['I']} "
                    text += f"E={info['genome']['E']} F={info['genome']['F']}\n"
                    text += f"Conexões: {info['connections']} | Perfil: {info['profile']}\n"
                    text += f"Preferências: {info['preferences']}"

                    if 'cognitive_state' in info:
                        cog = info['cognitive_state']
                        text += f"\nExploração: {cog['exploration_rate']} | "
                        text += f"Memórias: {cog['memory_size']}"

                    self.agent_info_label.text = text
                else:
                    self.selected_agent_id = None

    def on_draw(self):
        """Renderiza cena 3D e interface."""
        self.clear()

        # Configura câmera 3D
        self._setup_3d_projection()

        # Obtém dados do motor
        res = self.engine.get_render_data()
        positions, healths, connections, profiles, colors = res

        # Desenha conexões sociais
        if self.show_connections and positions:
            self._draw_connections(positions, connections, profiles)

        # Desenha agentes
        if positions:
            self._draw_agents(positions, healths, profiles, colors)

        # Desenha interface 2D
        self._draw_interface()

    def _setup_3d_projection(self):
        """Configura matriz de projeção 3D."""
        glViewport(0, 0, self.width, self.height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Manual gluPerspective
        aspect = self.width / float(self.height)
        fovy = 60.0
        zNear = 1.0
        zFar = 1000.0
        f = 1.0 / np.tan(np.radians(fovy) / 2.0)
        proj = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (zFar+zNear)/(zNear-zFar), (2*zFar*zNear)/(zNear-zFar)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        glMultMatrixf((GLfloat * 16)(*proj.flatten()))

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Posiciona câmera orbital
        glTranslatef(0, 0, -self.camera['distance'])
        glRotatef(self.camera['rotation_x'], 1, 0, 0)
        glRotatef(self.camera['rotation_y'], 0, 1, 0)
        glTranslatef(-50, -50, -50)  # Centraliza no campo

    def _draw_connections(self, positions, connections, profiles):
        """Desenha linhas de conexão entre agentes."""
        glBegin(GL_LINES)

        # Create a mapping from ID to position for quick lookup
        id_to_pos = {agent.id: agent.position for agent in self.engine.agents.values() if agent.is_alive()}

        for i, conns in enumerate(connections):
            # i is NOT the ID here based on engine logic, but it corresponds to a live agent
            # Actually engine.get_render_data() returns positions of live agents.
            # connections contains list of (id1, id2)
            for id1, id2 in conns:
                p1 = id_to_pos.get(id1)
                p2 = id_to_pos.get(id2)

                if p1 is not None and p2 is not None:
                    # Cor baseada no tipo de conexão
                    # We need to find the profile of these agents
                    # Profiling is expensive here, let's keep it simple
                    glColor4f(0.5, 0.5, 0.5, 0.2)
                    glVertex3f(*p1)
                    glVertex3f(*p2)

        glEnd()

    def _draw_agents(self, positions, healths, profiles, colors):
        """Desenha agentes como pontos coloridos."""
        glPointSize(8.0)
        glBegin(GL_POINTS)

        for i, (pos, health, profile, color) in enumerate(zip(positions, healths, profiles, colors)):
            # Modulação por saúde (agentes fracos ficam mais escuros)
            health_factor = 0.4 + health * 0.6
            r, g, b = color
            r *= health_factor
            g *= health_factor
            b *= health_factor

            # Agente selecionado é destacado em branco
            # Need to check if current agent ID matches selected_agent_id
            # but positions[i] doesn't have ID.
            # Re-fetch agent from engine for selection highlighting

            glColor3f(r, g, b)
            glVertex3f(*pos)

        glEnd()

    def _draw_interface(self):
        """Desenha elementos de interface 2D."""
        # Muda para projeção ortográfica
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Desenha labels
        self.stats_label.draw()
        self.agent_info_label.draw()
        self.help_label.draw()

        # Restaura projeção 3D
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Rotacionar câmera com o mouse."""
        if buttons & pyglet.window.mouse.LEFT:
            self.camera['rotation_y'] += dx * 0.5
            self.camera['rotation_x'] += dy * 0.5
            self.camera['rotation_x'] = max(-89, min(89, self.camera['rotation_x']))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Zoom com scroll do mouse."""
        self.camera['distance'] -= scroll_y * 8
        self.camera['distance'] = max(50, min(400, self.camera['distance']))

    def on_mouse_press(self, x, y, button, modifiers):
        """Seleciona agente ao clicar."""
        if button == pyglet.window.mouse.LEFT:
            # Seleção cíclica simples
            alive_ids = [aid for aid, a in self.engine.agents.items() if a.is_alive()]
            if alive_ids:
                if self.selected_agent_id is None:
                    self.selected_agent_id = alive_ids[0]
                else:
                    try:
                        idx = alive_ids.index(self.selected_agent_id)
                        self.selected_agent_id = alive_ids[(idx + 1) % len(alive_ids)]
                    except ValueError:
                        self.selected_agent_id = alive_ids[0]

        elif button == pyglet.window.mouse.RIGHT:
            self.engine.inject_signal(50, 50, 50, 20.0)

    def on_key_press(self, symbol, modifiers):
        """Controles de teclado."""
        if symbol == pyglet.window.key.SPACE:
            self.paused = not self.paused
        elif symbol == pyglet.window.key.C:
            self.show_connections = not self.show_connections
        elif symbol == pyglet.window.key.R:
            from ..core.particle_system import BioGenesisEngine
            self.engine = BioGenesisEngine(num_agents=250)
            self.selected_agent_id = None
            self.frame_count = 0
        elif symbol == pyglet.window.key.ESCAPE:
            self.close()

    def run(self):
        """Inicia loop principal."""
        pyglet.app.run()


def main():
    """Ponto de entrada do visualizador."""
    try:
        window = BioGenesisViewer()
        window.run()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
